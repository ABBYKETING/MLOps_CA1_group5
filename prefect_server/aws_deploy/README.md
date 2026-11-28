# Prefect server on AWS Elastic Beanstalk (Yue Liang, CA1 Part 2 F3)

A self-hosted Prefect server running as a Docker container on AWS Elastic Beanstalk, with its
run history stored in an AWS RDS **PostgreSQL** database. The flow
`flows/geo_account_risk_flow_yueliang.py` still runs on the analyst's laptop (it reads the team's
RDS MySQL data), but it reports every run, log and artifact to this server, so anyone with the
URL and login can see the results.

```
laptop: python flows/geo_account_risk_flow_yueliang.py
   |  reads data from                      |  reports runs, logs, artifacts to
   v                                       v
RDS MySQL (team data, spbuy)        Elastic Beanstalk: Prefect server (Docker, port 4200)
                                           |  stores run history in
                                           v
                                    RDS PostgreSQL (database "prefect")
```

## Why these choices

- **PostgreSQL, not the team's MySQL:** Prefect server only supports SQLite and PostgreSQL.
- **RDS, not SQLite inside the container:** a container's files are lost when Elastic Beanstalk
  replaces the container or EC2 instance (this is what happened to the team's first MLflow
  deployment). With RDS the run history survives.
- **Basic authentication:** an Elastic Beanstalk URL is public. `PREFECT_SERVER_API_AUTH_STRING`
  makes the API refuse requests without the login (tested: 401 without it).
- **No secrets in files:** the database URL and the login are Elastic Beanstalk environment
  properties, not part of the Dockerfile or this repository.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Installs Prefect 3.9.5 (same as the laptop client) and starts `prefect server` on port 4200 |
| `prefect-eb.zip` | The Dockerfile zipped at the root; this is what is uploaded to Elastic Beanstalk |

## 1. Create the PostgreSQL database (RDS console)

1. RDS -> Create database -> Standard create -> **PostgreSQL** (version 16 or 17).
2. Templates: **Free tier** (or Dev/Test with `db.t4g.micro` / `db.t3.micro`).
3. DB instance identifier: `prefect-db`. Master username: `prefectadmin`, choose a password.
4. Connectivity: **Public access: Yes**; VPC security group: allow inbound **PostgreSQL (5432)**.
5. Additional configuration -> **Initial database name: `prefect`**.
6. Create, wait until **Available**, copy the **Endpoint**.

## 2. Create the Elastic Beanstalk environment

1. Elastic Beanstalk -> Create application. Name: `prefect-yueliang`.
2. Platform: **Docker** (Amazon Linux 2023).
3. Application code: Upload your code -> `prefect-eb.zip`.
4. Presets: **Single instance**.
5. Service access: service role **LabRole**, EC2 key pair **vockey**, EC2 instance profile
   **LabInstanceProfile** (AWS Academy Learner Lab).
6. Instance type: **t3.small** (Prefect server needs about 1 GB of memory).
7. Environment properties (Configure updates, monitoring, and logging -> Environment properties):

| Name | Value |
|---|---|
| `PREFECT_SERVER_DATABASE_CONNECTION_URL` | `postgresql+asyncpg://prefectadmin:<password>@<rds-endpoint>:5432/prefect` |
| `PREFECT_SERVER_API_AUTH_STRING` | `admin:<choose-a-password>` |
| `PREFECT_SERVER_UI_API_URL` | leave out for now (set in step 3) |

8. Create the environment and wait until Health is **Ok**. Copy the domain, for example
   `prefect-yueliang.eba-xxxx.us-east-1.elasticbeanstalk.com`.

## 3. Point the UI at the AWS address

The web UI needs to know the public API address. Configuration -> Updates, monitoring, and
logging -> Edit -> add `PREFECT_SERVER_UI_API_URL` = `http://<your-eb-domain>/api` -> Apply.
Open `http://<your-eb-domain>` and log in with the auth string.

## 4. Run the flow against the AWS server (laptop, PowerShell)

```powershell
$env:DB_HOST = '<team RDS MySQL endpoint>'      # plus DB_USER, DB_PASSWORD, DB_NAME as usual
$env:PREFECT_API_URL = 'http://<your-eb-domain>/api'
$env:PREFECT_API_AUTH_STRING = 'admin:<the password from step 2>'
python flows\geo_account_risk_flow_yueliang.py
```

The run, its logs and its six artifacts appear in the Prefect UI on Elastic Beanstalk.

## Tested before deployment

The same configuration (Prefect 3.9.5 server on PostgreSQL 16 with the auth string) was run
locally: the health check answered, API requests without the login were refused with 401, and a
full flow run was stored in the PostgreSQL `flow_run` table and shown as Completed.

## AWS Academy notes

- Learner Lab stops EC2 and RDS when the lab session ends; start the lab (and wait for both to
  be running) before a demo. The Elastic Beanstalk URL stays the same.
- If deployment fails with a permission error (the team's MLflow deployment once hit an AWS
  Academy policy denial on `s3:ListBucket`), screenshot the error and the events page for the
  report.
