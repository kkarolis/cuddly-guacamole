import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import structlog
import pathlib
import argparse
import logging
import sys
from scipy.ndimage import shift
from datetime import datetime
from typing import List, Tuple, Optional, Union, Any

logger: Any = structlog.get_logger()


def read_energy_daily_hourly(
    db_path: Union[str, pathlib.Path], meter_type: str
) -> np.ndarray:
    """Read energy measurements from SQLite database.

    Args:
        db_path: Path to the SQLite database
    """
    with sqlite3.connect(db_path) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            WITH energy_hourly AS (
                SELECT direction, muid, year, month, day, day_of_week, hour, SUM(value) as value
                FROM energy
                GROUP BY direction, muid, year, month, day, day_of_week, hour
            )
            SELECT day, hour, value
            FROM energy_hourly
            WHERE direction = ?
            """,
            (meter_type,),
        )
        records: List[Tuple[int, float]] = cursor.fetchall()
        logger.info("records_found", count=len(records))

        if not records:
            logger.error("no_valid_measurements_found")
            return np.array([])

        return np.array([record[2] for record in records])


def read_avg_energy_hourly(
    db_path: Union[str, pathlib.Path], meter_type: str
) -> np.ndarray:
    """Read energy measurements from SQLite database.

    Args:
        db_path: Path to the SQLite database
    """
    with sqlite3.connect(db_path) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            WITH energy_hourly AS (
                SELECT direction, muid, year, month, day, day_of_week, hour, SUM(value) as value
                FROM energy
                GROUP BY direction, muid, year, month, day, day_of_week, hour
            )
            SELECT hour, AVG(value) as value
            FROM energy_hourly
            WHERE direction = ?
            GROUP BY hour
            ORDER BY hour
            """,
            (meter_type,),
        )
        records: List[Tuple[int, float]] = cursor.fetchall()
        logger.info("records_found", count=len(records))

        if not records:
            logger.error("no_valid_measurements_found")
            return np.array([])

        return np.array([record[1] for record in records])


def read_weekly_totals(
    db_path: Union[str, pathlib.Path], meter_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Read total energy usage by day of week from SQLite database.

    Args:
        db_path: Path to the SQLite database
        meter_type: Type of meter (import/export)

    Returns:
        Tuple of (daily_totals, day_of_week)
    """
    logger: Any = structlog.get_logger()
    with sqlite3.connect(db_path) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            WITH energy_daily AS (
                SELECT direction, muid, year, month, day, day_of_week, SUM(value) as value
                FROM energy
                GROUP BY direction, muid, year, month, day, day_of_week
            )
            SELECT
                day_of_week,
                AVG(value) as total_value
            FROM energy_daily
            WHERE direction = ?
            GROUP BY day_of_week
            ORDER BY day_of_week
            """,
            (meter_type,),
        )
        records: List[Tuple[int, float]] = cursor.fetchall()
        logger.info("records_found", count=len(records))

        if not records:
            logger.error("no_valid_measurements_found")
            return np.array([]), np.array([])

        days = np.array([record[0] for record in records])
        totals = np.array([record[1] for record in records])
        return totals, days


def plot_daily_pattern(import_data: np.ndarray, export_data: np.ndarray) -> None:
    """Plot average energy usage pattern throughout the day.

    Args:
        import_data: Array of import energy measurements
        export_data: Array of export energy measurements
            (will be plotted as negative values)
    """
    logger: Any = structlog.get_logger()

    # Create x-axis based on data length
    x_values: np.ndarray = np.arange(len(import_data))

    # Calculate delta (import - export)
    delta_data: np.ndarray = import_data - export_data

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, import_data, "b-", linewidth=2, label="Import")
    plt.plot(x_values, -export_data, "r-", linewidth=2, label="Export (negative)")
    plt.plot(x_values, delta_data, "g-", linewidth=2, label="Delta (Import - Export)")

    plt.title("Daily Energy Usage Pattern")
    plt.xlabel("Hour of Day")
    plt.ylabel("Energy Usage (kWh)")
    plt.grid(True)
    plt.xticks(x_values[::2])  # Show every other tick
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


def plot_autocorrelation(data: np.ndarray, label: str = "Energy Usage") -> None:
    """Plot correlation matrix for hour-of-day data with different time shifts.

    Args:
        data: Array of hourly energy measurements
        label: Label for the plot title and legend
    """
    # Calculate correlation
    result = np.correlate(data, data, mode="full")
    result = result[result.size // 2 :]

    # Normalize by dividing by the maximum correlation (at lag 0)
    result = result / result[0]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(result, "b-", linewidth=2, label=label)
    plt.title(f"{label} Autocorrelation")
    plt.tick_params(
        axis="both",
        labelbottom=False,
    )
    plt.xlabel("Lag (Each tick is 12 hours)")
    plt.ylabel("Normalized Correlation")
    plt.xticks(range(0, len(result), 12))  # Show every 4 hours
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


def plot_weekly_pattern(import_data: np.ndarray, day_of_week: np.ndarray) -> None:
    """Plot energy import patterns by day of week.

    Args:
        import_data: Array of import energy measurements
        day_of_week: Array of day of week (0-6, where 0 is Monday)
    """
    logger: Any = structlog.get_logger()

    if len(import_data) == 0:
        logger.error("no_data_to_plot")
        return

    # Create a matrix of hours x days
    hours_per_day = 24
    days_in_week = 7
    weekly_pattern = np.zeros((hours_per_day, days_in_week))
    counts = np.zeros((hours_per_day, days_in_week))

    # Fill the matrix with values
    for i, (value, day) in enumerate(zip(import_data, day_of_week)):
        hour = i % hours_per_day
        weekly_pattern[hour, day] += value
        counts[hour, day] += 1

    # Calculate averages
    weekly_pattern = np.divide(
        weekly_pattern, counts, out=np.zeros_like(weekly_pattern), where=counts != 0
    )

    # Create the plot
    plt.figure(figsize=(12, 6))
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    for day in range(days_in_week):
        plt.plot(weekly_pattern[:, day], label=days[day])

    plt.title("Weekly Energy Import Pattern")
    plt.xlabel("Hour of Day")
    plt.ylabel("Energy Usage (kWh)")
    plt.grid(True)
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


def plot_weekly_total(weekly_totals: np.ndarray, day_of_week: np.ndarray) -> None:
    """Plot total energy usage by day of week.

    Args:
        weekly_totals: Array of total energy usage for each day
        day_of_week: Array of day of week (0-6, where 0 is Monday)
    """
    logger: Any = structlog.get_logger()

    if len(weekly_totals) == 0:
        logger.error("no_data_to_plot")
        return

    # Create the plot
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(days, weekly_totals)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.title("Total Energy Usage by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Total Energy Usage (kWh)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show(block=False)


def main() -> int:
    """Main entry point for the autocorrelation analysis."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate autocorrelation plot from energy data.",
    )
    parser.add_argument(
        "db",
        help="Path to the SQLite database file",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG", "ERROR", "WARNING"],
        help="Logging level",
    )

    args: argparse.Namespace = parser.parse_args()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, args.log_level)
        )
    )

    logger: Any = structlog.get_logger()

    if not args.db.exists():
        logger.error("database_not_found", path=args.db)
        return 1

    try:
        logger.info("reading_measurements", db=args.db)
        import_measurements = read_avg_energy_hourly(args.db, "import")
        export_measurements = read_avg_energy_hourly(args.db, "export")
        weekly_totals, day_of_week = read_weekly_totals(args.db, "import")
        logger.info("measurements_read", count=len(import_measurements))

        if len(import_measurements) == 0:
            logger.error("no_measurements_found")
            return 1

        logger.info(
            "plotting_patterns",
            data_points=len(import_measurements),
        )
        plot_daily_pattern(import_measurements, export_measurements)
        plot_weekly_total(weekly_totals, day_of_week)

        daily_measurements_export = read_energy_daily_hourly(args.db, "export")
        plot_autocorrelation(daily_measurements_export, "Export")

        daily_measurements_import = read_energy_daily_hourly(args.db, "import")
        plot_autocorrelation(daily_measurements_import, "Import")

        # Wait for user input before exiting
        input("Press Enter to exit...")
        return 0

    except Exception as e:
        logger.exception("analysis_failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
