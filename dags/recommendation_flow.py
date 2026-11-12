"""Prefect 3 flow - orchestrates the full recommendation pipeline.

    # one-off run (prints a link to the run in the UI if a server is running)
    python dags/recommendation_flow.py

    # with the UI (take the screenshot from here)
    prefect server start                      # terminal 1 -> http://127.0.0.1:4200
    python dags/recommendation_flow.py        # terminal 2

    # scheduled (daily 02:00) - keeps a worker process serving the deployment
    python dags/recommendation_flow.py --serve

Task dependencies mirror the pipeline DAG:
ingest -> validate -> prepare -> transform -> features -> train -> evaluate
Ingestion retries on network failures; every task result is a small JSON-able
summary so the Prefect UI shows what each step produced.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, get_run_logger, task  # noqa: E402

from src.common.utils import load_config, set_seed  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.validation import validate  # noqa: E402


@task(name="ingest_data", retries=2, retry_delay_seconds=10)
def ingest_task(cfg):
    meta = ingest.run(cfg)
    return {k: v["rows"] for k, v in meta["files"].items()}


@task(name="validate_data")
def validate_task(cfg, _upstream):
    return validate.run(cfg)["summary"]


@task(name="prepare_and_split")
def prepare_task(cfg, _upstream):
    info = prepare.run(cfg)
    return {k: info[k] for k in ("train_rows", "test_rows", "cutoff_timestamp")}


@task(name="transform_derived_datasets")
def transform_task(cfg, _upstream):
    return transform.run(cfg)


@task(name="build_feature_store")
def features_task(cfg, _upstream):
    meta = build_features.run(cfg)
    return {"feature_version": meta["feature_version"], "rows": meta["rows"]}


@task(name="train_models")
def train_task(cfg, _upstream):
    return train.run(cfg)


@task(name="evaluate_models")
def evaluate_task(cfg, _upstream):
    df = evaluate.run(cfg)
    k = cfg["evaluation"]["k"]
    return df[[f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}"]].round(4).to_dict()


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None):
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    log = get_run_logger()
    raw = ingest_task(cfg)
    val = validate_task(cfg, raw)
    prep = prepare_task(cfg, val)
    trans = transform_task(cfg, prep)
    feats = features_task(cfg, trans)
    models = train_task(cfg, feats)
    results = evaluate_task(cfg, models)
    log.info("Pipeline complete. Results: %s", results)
    return results


if __name__ == "__main__":
    if "--serve" in sys.argv:
        recommendation_pipeline.serve(name="daily-recsys", cron="0 2 * * *")
    else:
        recommendation_pipeline()
