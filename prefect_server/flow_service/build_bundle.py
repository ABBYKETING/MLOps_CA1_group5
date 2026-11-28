"""Build prefect-flow-service.zip, the source bundle uploaded to Elastic Beanstalk (#75).

Elastic Beanstalk builds the Docker image from a zip with the Dockerfile at its root, so the
bundle contains the Dockerfile, requirements.txt, and the three code files the flow imports.
Run from anywhere:  python prefect_server/flow_service/build_bundle.py
"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FILES = {
    "Dockerfile": HERE / "Dockerfile",
    "requirements.txt": HERE / "requirements.txt",
    "src/analytics_geo_account_yueliang.py": REPO / "src" / "analytics_geo_account_yueliang.py",
    "src/database.py": REPO / "src" / "database.py",
    "flows/geo_account_risk_flow_yueliang.py": REPO / "flows" / "geo_account_risk_flow_yueliang.py",
}

bundle = HERE / "prefect-flow-service.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for name, path in FILES.items():
        zf.write(path, name)
print(f"Wrote {bundle} with {len(FILES)} files")
