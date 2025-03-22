# Overview

This simple application uses a single endpoint `/analytics/energy/measurement` through `FastAPI` to interact with a demo `SQLite` database. 

In production scenario, you wouldn't use an application database like this because:
- firstly, there is a dataplane service which feeds the data to the system. It would be hard to scale and ensure consistency as we would need to fan out traffic to all the application servers
- secondly, it would make it harder to scale out the web app layer
- thirdly, you would need to perform various backup activities which wouldn't be easily possible with this approach.
- finally, upon crash it would need to fetch data from other replicas, making this set up too complicated. 

Usually for a production set up of this type we would likely go with single-master / multi-slave replica set up and set the web api's to interact with the read replicas for reading analytics data.


As for the API itself, it's essentially just a group by through RestAPI. Based on analysis performed seems customers may like this functionality to track usage across different time slices.


After running the application with steps below, more documentation can be found on the OpenAPI auto-generated documentation [http://localhost:8000/docs](http://localhost:8000/docs). There is a `Try it Out` button there to test the API from the browser.

# Prerequisites

There are 2 ways to run this application, either through `uv` ([installation](https://docs.astral.sh/uv/getting-started/installation/)) or `docker` ([installation](https://docs.docker.com/get-started/get-docker/))

## `uv` based run

(make sure to run it from projects root)
```bash
uv run server/main.py --dev run-server
```

## Docker based run

```
docker build -t joule:1.0 .
docker run -it --rm -p 127.0.0.1:8000:8000 -e LOG_LEVEL=INFO joule:1.0
```
