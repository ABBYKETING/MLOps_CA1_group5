FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "mlflow[auth]==3.18.0"

RUN mkdir -p /app/data/mlartifacts

EXPOSE 5000
