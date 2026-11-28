# Prefect flow service on AWS Elastic Beanstalk (Yue Liang, PBI 9 Task 4 #75)

The second service of the advanced challenge (spec section 2.3.2, step 3): the analytics flow
`flows/geo_account_risk_flow_yueliang.py` runs in its **own Docker container** on Elastic
Beanstalk and connects to the Prefect web server deployed from `../aws_deploy`.

```
Elastic Beanstalk env 1: Prefect web server (aws_deploy)  <--- registers deployment, reports runs ---
        | stores run history in                                                                 |
        v                                                                                       |
RDS PostgreSQL "prefect"                       Elastic Beanstalk env 2: flow service (this folder)
                                                        | reads the latest data from
                                                        v
                                               team RDS MySQL "spbuy"
```

When the container starts it runs `python flows/geo_account_risk_flow_yueliang.py --serve --webserver`:

1. registers the deployment **daily-geo-account-fraud-yueliang** on the Prefect server,
   scheduled every day at **07:30 Asia/Singapore** (before analysts start reviewing);
2. keeps polling the server, and runs the flow on the schedule **or whenever someone clicks
   Run (Quick run) on the deployment in the Prefect UI**;
3. opens a health endpoint `/health` on port 8080, which Elastic Beanstalk uses (through its
   proxy on port 80) to check that the service is alive.

Results are the same as a laptop run: logs and the 6 artifacts appear in the Prefect UI.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.12 slim image with only the flow's libraries; starts the service |
| `requirements.txt` | prefect, pandas, numpy, matplotlib, seaborn, SQLAlchemy, PyMySQL at the team's pinned versions |
| `build_bundle.py` | Builds `prefect-flow-service.zip` from the repository files |
| `prefect-flow-service.zip` | The bundle uploaded to Elastic Beanstalk: Dockerfile and requirements at the root, plus `src/analytics_geo_account_yueliang.py`, `src/database.py`, `flows/geo_account_risk_flow_yueliang.py` |

Rebuild the bundle after changing the flow or analytics code:

```powershell
python prefect_server/flow_service/build_bundle.py
```

## Deploy (second Elastic Beanstalk environment)

Same application (`prefect-yueliang`), **Create new environment**, with the same settings as the
server except the code bundle and the environment properties:

| Setting | Value |
|---|---|
| Tier / platform | Web server environment, Docker (Amazon Linux 2023) |
| Application code | Local file `prefect-flow-service.zip`, version label `flow-v1` |
| Preset | Single instance |
| Service access | Service role `LabRole`, instance profile `LabInstanceProfile`, key pair `vockey` |
| Instance type | `t3.small` |

Environment properties (values are never stored in the repository):

| Name | Value |
|---|---|
| `DB_HOST` | team RDS MySQL endpoint |
| `DB_USER` | team RDS user |
| `DB_PASSWORD` | team RDS password |
| `DB_NAME` | `spbuy` |
| `PREFECT_API_URL` | `http://<prefect server domain>/api` |
| `PREFECT_API_AUTH_STRING` | the Prefect server login (`admin:<password>`) |

After Health is Green: Prefect UI -> **Deployments** -> `daily-geo-account-fraud-yueliang` ->
**Run** -> **Quick run**. The run is picked up by this service within a few seconds.

## Tested before deployment (28 Nov 2026)

The bundle was unpacked into an empty folder and started exactly as the container starts it,
against a Prefect 3.9.5 server on PostgreSQL 16 with the login enabled:

- `/health` answered `{"message":"OK"}` (HTTP 200);
- the deployment was registered with schedule `30 7 * * *` in `Asia/Singapore`, and the next
  runs appeared as Scheduled;
- a run triggered from the server (`prefect deployment run ...`, the same as Quick run in the UI)
  was executed by the service and finished **Completed** with all 9 tasks and 6 artifacts.

## Deployed on AWS (28 Nov 2026)

- Environment `prefect-yueliang-flow` (version `flow-v1`) reached Health **Green**; the deployment
  `daily-geo-account-fraud-yueliang` appeared as **Ready** with the 07:30 daily schedule.
- A Quick run from the AWS Prefect UI (`reliable-chital`) finished **Completed** in 42 s with
  9 task runs, reading the team `spbuy` database from inside AWS.
- Lesson learned: the first start exited with `401 Unauthorized` because
  `PREFECT_API_AUTH_STRING` did not match the server's `PREFECT_SERVER_API_AUTH_STRING`.
  Elastic Beanstalk still showed **Green**, because Basic health only checks the instance, not the
  container, so the container log (`unexpected-quit.log`) is where to look when the deployment is
  missing from the UI.

## Error handling (#71)

The flow was run with a wrong database password (`$env:DB_PASSWORD='wrong'`). `count_orders_in_rds`
was tried 3 times 10 seconds apart (2 retries), then the task and the flow ended in **Failed**
(run `misty-koel` on the AWS Prefect server, 27 s, 1 task run) with MySQL error 1045
"Access denied", and none of the analysis or report tasks ran on missing data.
