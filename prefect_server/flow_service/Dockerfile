# Prefect flow service for AWS Elastic Beanstalk (Docker platform), Yue Liang - PBI 9 Task 4 (#75).
# A second, separate service: it runs the geographic and account behaviour analytics flow
# (flows/geo_account_risk_flow_yueliang.py) and connects to the Prefect web server deployed
# from prefect_server/aws_deploy. All connection settings (team RDS MySQL, Prefect API URL and
# login) come from Elastic Beanstalk environment properties, never from this file.
FROM python:3.12-slim

WORKDIR /app

# Only the libraries the flow needs, at the team's pinned versions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY flows/ flows/

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    DO_NOT_TRACK=1 \
    PREFECT_RUNNER_SERVER_HOST=0.0.0.0 \
    PREFECT_RUNNER_SERVER_PORT=8080

# The runner's health endpoint; Elastic Beanstalk forwards port 80 to it
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"

# Register the deployment on the Prefect server, then wait for scheduled or manual runs
CMD ["python", "flows/geo_account_risk_flow_yueliang.py", "--serve", "--webserver"]
