# Shared Local MLflow Server

## Purpose

This deployment provides a shared MLflow tracking server for the
SP-buy Fraud Detection project.

The local deployment serves as an alternative demonstration environment
after the AWS Elastic Beanstalk environment encountered an AWS Academy
permission restriction. The lecturer confirmed that a local MLflow
server may be used for the demonstration.

## Architecture

Team Members
    -> Tailscale Private HTTPS
    -> Dockerised MLflow Server
    -> SQLite Tracking Database
    -> Persistent Model Artifact Storage
    -> SQLite Authentication Database

## Components

| Component | Implementation |
|---|---|
| MLflow | 3.18.0 |
| Deployment | Docker Compose |
| Tracking backend | SQLite |
| Artifact storage | Docker named volume |
| Authentication | MLflow Basic Authentication |
| Access control | Individual accounts and experiment permissions |
| Remote access | Tailscale Serve with private HTTPS |
| Host port | 127.0.0.1:5006 |
| Container port | 5000 |
| Persistent volume | spbuy-mlflow-shared-data |

## Prerequisites

- Docker Desktop with Docker Compose
- Tailscale for remote access
- Python environment with the project dependencies installed
- Access to the project repository

## Initial Configuration

From the repository root:

```powershell
cd mlflow_server/local_deploy

Copy-Item .env.example .env
Copy-Item auth_config.ini.example auth_config.ini
```

For a new deployment:

1. Replace the placeholder server secret with a strong random secret.
2. Replace the placeholder administrator password with a strong password.
3. Configure allowed hosts and CORS origins for the deployment.
4. Keep the actual configuration files private.

Do not overwrite existing configuration files when restarting an
existing deployment.

## Start the Server

```powershell
docker compose up -d --build
docker compose ps
```

Local MLflow UI:

http://127.0.0.1:5006

## Remote Access

Install and configure Tailscale on the host machine.

Expose the local service privately using:

```powershell
tailscale serve --bg http://127.0.0.1:5006
tailscale serve status
```

Add the assigned Tailscale HTTPS hostname and origin to the allowed
hosts and CORS configuration respectively. Run `docker compose up -d` to apply the updated configuration.

Share the host device only with authorised team members and restrict
network access to HTTPS port 443.

Do not use Tailscale Funnel to expose this service publicly.

The host computer, Docker Desktop and Tailscale must remain running
while remote access is required.

## User Management

Each team member uses an individual non-administrator MLflow account.

The administrator creates user accounts and grants the required
experiment-level permissions.

The shared experiment is:

SP-buy-Fraud-Detection

Team members are granted EDIT permission on this experiment.

Credentials must be distributed privately.

## Connect Training Scripts

Set the following environment variables in the client environment:

- MLFLOW_TRACKING_URI
- MLFLOW_TRACKING_USERNAME
- MLFLOW_TRACKING_PASSWORD

Use the private HTTPS tracking URI and the assigned individual
credentials.

All official models must be logged to the shared experiment using
the common MLflow logging conventions.

Run naming convention:

{member}_{algorithm}_{dataset_version}_{model_version}

## Persistence

The Docker named volume stores:

- MLflow tracking metadata
- Model and evaluation artifacts
- Authentication database

Container recreation does not remove the named volume.

To stop the server:

```powershell
docker compose stop
```

To restart the server:

```powershell
docker compose up -d
```

Do not use `docker compose down -v` unless permanent deletion of
the stored data is intended.

Back up the persistent storage before major configuration changes.

## Security

- The service is bound to localhost on the host computer.
- Remote access is restricted through Tailscale private HTTPS.
- Basic authentication is enabled.
- Individual non-admin accounts are used for experiment logging.
- Allowed hosts and CORS origins are explicitly configured.
- Remote MLflow Assistant access is disabled.
- Passwords, secrets and database files are excluded from Git.
- Actual credentials must never be committed to the repository.

## Verification Completed

- Authenticated and unauthenticated API access checks
- Individual user account and permission checks
- Persistent metadata and artifact verification
- Container recreation and recovery testing
- MLflow version upgrade verification
- Remote HTTPS connectivity
- Remote experiment logging by all three remote team members
- Official model parameter and metric logging
- Trained model download and reload verification for Abby's models

## Limitations

- The local host must remain online during remote access.
- SQLite is suitable for the scope of this coursework demonstration
  but is not intended as a highly available production backend.
- A Docker volume provides persistence but does not replace backups.
- The shared experiment's EDIT permission is collaborative and
  does not provide individual run ownership isolation.
- This deployment is a local demonstration alternative, not an
  AWS production deployment.
