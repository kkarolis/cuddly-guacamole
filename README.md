OBIS code: 0100021D00FF ?

OBIS codes are defined in the IEC 62056 standard and are used to identify data items in utility metering systems. Each segment of the code has a specific meaning:

Breaking down "0100021D00FF":

 • "01": This typically represents the energy type (electricity)
 • "00": This could indicate the measurement channel
 • "02": This often represents the quantity being measured
 • "1D": This is the measurement type (hexadecimal 1D = decimal 29)
 • "00": This might be a processing method or tariff
 • "FF": This often indicates "all" values or a summation value


https://github.com/Shagrat2/obis2hex/blob/main/src/extension.ts
https://de.wikipedia.org/wiki/OBIS-Kennzahlen
https://doc.smart-me.com/interfaces/auto-export - 1-1:2.29.0*255: Active Energy Total Export (load profile)

OBIS code: 0100011D00FF - active energy imported
1-1:1.29.0*255: Active Energy Total Import (load profile)

OBIS code: 0100021D00FF - active energy exported
1-1:2.29.0*255: Active Energy Total Export (load profile)

Assuming kWh as the unit.

## Running with Docker

The application can be run using Docker with the following steps:

### Build the Docker image

```bash
docker build -t energy-analytics-api .
```

### Run the Docker container

```bash
docker run -p 8000:8000 energy-analytics-api
```

This will:
1. Initialize the database
2. Start the FastAPI server
3. Make the API available at http://localhost:8000

### Development mode

To run the server in development mode with Docker:

```bash
docker run -p 8000:8000 -v $(pwd)/server:/app/server energy-analytics-api python -m server.main init-db && python -m server.main run-server --dev
```

This mounts the server directory as a volume, allowing you to make changes to the code without rebuilding the Docker image.
