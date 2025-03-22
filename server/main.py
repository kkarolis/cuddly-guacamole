from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Any
from enum import Enum
import collections
import aiosqlite
import functools
import asyncio
import os
import structlog
import sys
import logging
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from structlog.stdlib import ProcessorFormatter

from fastapi import HTTPException
import argparse
import uvicorn
import json
import pathlib

logger = structlog.get_logger()


class GroupByField(str, Enum):
    # below must be ordered by "cardinality"
    # lowest "cardinality" items first.
    DIRECTION = "direction"
    MUID = "muid"
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    DAY_OF_WEEK = "day_of_week"
    HOUR = "hour"


# we maintain an index of the fields in order to later verify the order of
# passed in through group key
GROUP_BY_FIELDS_CARDINALITY_ORDERED = collections.OrderedDict(
    [(v, i) for i, v in enumerate(GroupByField)]
)
# with below we get: {
# hour: (direction, muid, year, month, day, day_of_week, hour)
# day_of_week: (direction, muid, year, month, day, day_of_week)
# }
GROUP_BY_FIELDS = {
    v.value: tuple(GROUP_BY_FIELDS_CARDINALITY_ORDERED.keys())[: i + 1]
    for i, v in enumerate(GROUP_BY_FIELDS_CARDINALITY_ORDERED)
}


def process_json_file(input_file):
    for record in json.load(input_file)["data"]:
        timestamp_str = record["timestamp"]
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        logger.debug("processing_record", dt=timestamp, record=record)
        yield (timestamp, record)


def get_value_type(record: dict) -> (float, str):
    if "0100011D00FF" in record:
        return record["0100011D00FF"], "import"
    elif "0100021D00FF" in record:
        return record["0100021D00FF"], "export"
    else:
        logger.warning("no_value_type", record=record)
        return None, None


async def init_db():
    """Initialize the database and create required tables."""
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.cursor()

        # Create main energy table
        await cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS energy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT,
                value FLOAT,
                direction TEXT,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                hour INTEGER,
                minute INTEGER,
                second INTEGER,
                day_of_week INTEGER,
                muid TEXT,
                timestamp DATETIME
            );
        """
        )

        # Process JSON files from data directory
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        for json_file in pathlib.Path(data_dir).glob("*.json"):
            logger.info("load_single_file_start", file=json_file.name)
            with open(json_file) as f:
                for dt, record in process_json_file(f):
                    await cursor.execute(
                        """
                        INSERT INTO energy
                        (data, value, direction, year, month, day, hour, 
                         minute, second, day_of_week, muid, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            json.dumps(record),
                            *get_value_type(record),
                            dt.year,
                            dt.month,
                            dt.day,
                            dt.hour,
                            dt.minute,
                            dt.second,
                            dt.weekday(),
                            record.get("tags", {}).get("muid", None),
                            dt,
                        ),
                    )
            logger.info("load_single_file_end", file=json_file.name)

        await db.commit()
    logger.info("database_initialized_successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG)
    )

    # Startup
    yield


app = FastAPI(lifespan=lifespan)

# Mount static files directory
app.mount("/static", StaticFiles(directory="server/static"), name="static")


@app.get("/")
async def welcome_to_analytics_api():
    return FileResponse("server/static/index.html")


async def get_db():
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        yield db


class AggregateFunction(str, Enum):
    # has to match aggregate functions available in sql (and sqlite for the demo)
    AVERAGE = "avg"
    SUM = "sum"
    MINIMUM = "min"
    MAXIMUM = "max"


class MeasurementDirection(str, Enum):
    IN = "import"
    OUT = "export"


class MeasurementDataPoint(BaseModel):
    group_key: List[Any]
    value: float


class ResponseMetadata(BaseModel):
    records: int
    totalRecords: int
    unit: str
    nextToken: Optional[str] = None


class GetMeasurementDataOutput(BaseModel):
    data: List[MeasurementDataPoint]
    metadata: Optional[ResponseMetadata] = None


def _check_group_key_order(group_key: List[GroupByField]) -> bool:
    group_key_cardinality_index_values = [
        GROUP_BY_FIELDS_CARDINALITY_ORDERED[k] for k in group_key
    ]
    # check each element with previous one making sure it's increasing
    is_correct_order, _ = functools.reduce(
        (lambda x, y: (x[0] and y > x[1], y)),
        group_key_cardinality_index_values,
        (True, -1),
    )
    return is_correct_order


def _add_where_clause(
    query: List[str],
    query_params: List[str],
    muid: Optional[str],
    direction: Optional[MeasurementDirection],
    start: Optional[datetime],
    end: Optional[datetime],
):
    filtering_clauses = []

    if muid:
        filtering_clauses.append("muid = ?")
        query_params.append(muid)
    if direction:
        filtering_clauses.append("direction = ?")
        query_params.append(direction.value)

    if start and end:
        filtering_clauses.append("timestamp BETWEEN ? AND ?")
        query_params.append(start)
        query_params.append(end)
    elif start:
        filtering_clauses.append("timestamp >= ?")
        query_params.append(start)
    elif end:
        filtering_clauses.append("timestamp <= ?")
        query_params.append(end)

    # for the time being this only works with AND support
    if filtering_clauses:
        where_clause = " AND ".join([f"({clause})" for clause in filtering_clauses])
        query.append(f"WHERE {where_clause}")


@app.get("/analytics/energy/measurements", response_model=GetMeasurementDataOutput)
async def get_measurement_data(
    group_key: List[GroupByField] = Query(
        ...,
        description="""Group key fields to aggregate data by. Must be specified in order of increasing cardinality.

Fields (in order):
- direction: Energy flow direction (import/export)
- muid: Measurement device identifier
- year: Year of measurement
- month: Month of measurement
- day: Day of measurement
- day_of_week: Day of week (0-6, where 0 is Monday)
- hour: Hour of day (0-23)""",
    ),
    aggregate_function: AggregateFunction = Query(
        AggregateFunction.SUM,
        description="""Aggregation function to apply to the measurements.

Available functions:
- sum: Sum of all values
- avg: Average of all values
- min: Minimum value
- max: Maximum value""",
    ),
    muid: Optional[str] = Query(
        None, description="Filter measurements by specific device identifier"
    ),
    limit: Optional[int] = Query(
        100,
        gt=0,
        le=300,
        description="""Maximum number of records to return per page.""",
    ),
    page: Optional[int] = Query(
        0,
        ge=0,
        description="""Page number to return (0-based).""",
    ),
    direction: Optional[MeasurementDirection] = Query(
        None, description="Filter measurements by energy flow direction (import/export)"
    ),
    start: Optional[datetime] = Query(
        None, description="Start date for the data range (inclusive)"
    ),
    end: Optional[datetime] = Query(
        None, description="End date for the data range (inclusive)"
    ),
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Retrieve energy measurement data with flexible aggregation and filtering options.

    This endpoint allows you to analyze energy consumption patterns by grouping and
    aggregating measurements based on various time dimensions and other parameters.

    Examples:

    1. Daily energy consumption for a specific month:
    ```
    group_key: ["day"]
    start: "2024-02-01T00:00:00"
    end: "2024-02-29T23:59:59"
    ```

    2. Average energy consumption by day of week:
    ```
    group_key: ["day_of_week"]
    aggregate_function: "avg"
    ```

    3. Hourly energy consumption for a specific device:
    ```
    group_key: ["hour"]
    muid: "device123"
    ```

    4. Monthly energy import/export comparison:
    ```
    group_key: ["month", "direction"]
    ```

    5. Paginated results:
    ```
    group_key: ["day"]
    limit: 50
    page: 1
    ```

    Notes:
    - Day of week is represented as 0-6, where 0 is Monday and 6 is Sunday
    - All energy values are returned in kWh
    - Group keys must be specified in order of increasing cardinality
    - When filtering by date range, both start and end dates are inclusive
    - Pagination is 0-based (first page is 0)
    """
    # convert the input group to unique key ordered list
    group_keys_unique_input = list(collections.OrderedDict.fromkeys(group_key).keys())
    if not _check_group_key_order(group_keys_unique_input):
        raise HTTPException(
            status_code=400,
            detail=(
                "Group keys must always follow the following order: "
                f"{', '.join([k.value for k in GROUP_BY_FIELDS_CARDINALITY_ORDERED])}"
            ),
        )

    output_group_keys = [v.value for v in group_keys_unique_input]
    output_group_by_fields = ", ".join(output_group_keys)
    group_by_fields = ", ".join(GROUP_BY_FIELDS[output_group_keys[-1]])

    query_cte, query_params_cte = [], []
    query, query_params = [], []

    query_cte.append(f"SELECT {group_by_fields}, sum(value) as value")
    query_cte.append("FROM energy")
    _add_where_clause(query_cte, query_params_cte, muid, direction, start, end)
    query_cte.append(f"GROUP BY {group_by_fields}")

    query.append(
        f"SELECT {output_group_by_fields}, {aggregate_function.value}(value) as value"
    )
    query.append("FROM energy_grouped")
    query.append(f"GROUP BY {output_group_by_fields}")

    # TODO: add proper token based pagination, below is pretty not a
    #  inadvisable practice.
    query.append(f"LIMIT {limit}")
    query.append(f"OFFSET {page * limit}")

    # use a CTE to filter and sum up data and then do an aggregation
    query_final = f"""
    WITH energy_grouped AS (
        {" ".join(query_cte)}
    )
    {" ".join(query)}
    """
    query_params_final = [*query_params_cte, *query_params]

    logger.debug(
        "execute_query",
        query=query_final,
        query_params=query_params_final,
    )

    async with db.cursor() as cursor:
        await cursor.execute(query_final, query_params_final)
        rows = await cursor.fetchall()

    return GetMeasurementDataOutput(
        data=[
            MeasurementDataPoint(
                group_key=[row[c] for c in output_group_keys],
                value=row["value"],
            )
            for row in rows
        ],
        metadata=ResponseMetadata(
            # TODO for pagination would be nice to return all records matching.
            records=len(rows),
            totalRecords=len(rows),
            unit="kWh",  # would be dynamic in production application
            nextToken=None,  # TODO: Implement pagination if needed
        ),
    )


def run_server(dev_mode=False):
    """Run the FastAPI server."""
    if dev_mode:
        logger.info("server_starting_development_mode")
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    else:
        logger.info("server_starting_production_mode")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4, reload=False)


def setup_logging(log_level: int, dev_mode: bool):
    processors = []
    processors.append(structlog.processors.add_log_level)
    processors.append(
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        )
    )
    processors.append(structlog.processors.TimeStamper(fmt="iso"))
    processors.append(structlog.processors.StackInfoRenderer())

    if dev_mode:
        processors.append(structlog.dev.set_exc_info)
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Energy Analytics API - CLI interface")

    # Add dev mode and log level arguments to main parser
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with console logging",
    )
    log_level_choices = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument(
        "--log-level",
        choices=log_level_choices,
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Set the logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init DB command
    subparsers.add_parser("init-db", help="Initialize the database")
    subparsers.add_parser("run-server", help="Run the API server")

    # Run server command
    args = parser.parse_args()

    # Configure logging based on the provided level and dev mode
    setup_logging(getattr(logging, args.log_level), args.dev)

    if args.command == "init-db":
        asyncio.run(init_db())
    elif args.command == "run-server":
        run_server(dev_mode=args.dev)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
