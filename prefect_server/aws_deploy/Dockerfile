# Prefect server for AWS Elastic Beanstalk (Docker platform), Yue Liang - CA1 Part 2 F3.
# The run history is stored in an AWS RDS PostgreSQL database, not inside the container,
# so it survives container or EC2 instance replacement.
# The database URL and the UI address come from Elastic Beanstalk environment properties
# (PREFECT_SERVER_DATABASE_CONNECTION_URL, PREFECT_UI_API_URL), never from this file.
FROM python:3.12-slim

# Same Prefect version as the laptop that runs the flow, so client and server match
RUN pip install --no-cache-dir prefect==3.9.5

ENV PREFECT_SERVER_ANALYTICS_ENABLED=false

EXPOSE 4200

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4200/api/health')"

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]
