"""Stage 6 - Model training (+ experiment tracking with MLflow).

1. Carve a time-based validation window out of TRAIN (last 10% of train time).
2. Tune the hybrid blend weights on that window (features for the window are
   rebuilt in memory from the sub-train data, so nothing leaks from validation
   or test). Every grid point is logged as a nested MLflow run.
3. Re-fit all models on the full train split and save them to models/<name>/.

Each final model gets its own MLflow run with parameters, training time,
feature-store version, data hashes and the pickled model as an artifact.
The evaluation stage later adds test metrics to the same run.
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd

from src.common.utils import PROJECT_ROOT, get_logger, load_config, path, read_json, set_seed, write_json
from src.features.build_features import compute_feature_tables
from src.features.feature_store import FeatureStore
from src.training.models import (HybridRecommender, ImplicitALS, MatrixFactorization,
                                 PopularityRecommender)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def setup_mlflow(cfg: dict):
    import mlflow

    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{(PROJECT_ROOT / uri.replace('sqlite:///', '')).as_posix()}"
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(name)
    return mlflow


def _validation_window(train: pd.DataFrame, frac: float):
    cut = int(train["timestamp"].quantile(1 - frac))
    return train[train["timestamp"] < cut].copy(), train[train["timestamp"] >= cut].copy(), cut


def tune_hybrid(train, users, items, cfg, log, mlflow) -> tuple[dict, pd.DataFrame]:
    from src.evaluation.evaluate import EvalContext

    hcfg = cfg["training"]["hybrid"]
    sub, val, cut = _validation_window(train, hcfg["validation_fraction"])
    as_of = pd.Timestamp(cut, unit="s", tz="UTC")
    uf, itf, _, meta = compute_feature_tables(sub, users, items, cfg, as_of)
    fs_val = FeatureStore.from_frames(uf, itf, meta)
    log.info("Tuning hybrid on validation window: sub-train=%d, val=%d rows", len(sub), len(val))

    k = cfg["evaluation"]["k"]
    ctx = EvalContext(sub, val, fs_val, cfg["preparation"]["relevance_threshold"],
                      max_users=1500, seed=cfg["project"]["seed"])
    cf = ImplicitALS(**cfg["training"]["implicit_als"], seed=cfg["project"]["seed"]).fit(sub, fs_val)
    hybrid = HybridRecommender(w_cf=1.0).fit(sub, fs_val, base=cf)

    rows = []
    grid = hcfg["grid"]
    for wg, wp in itertools.product(grid["w_genre"], grid["w_popularity"]):
        hybrid.set_weights(w_genre=wg, w_popularity=wp)
        m = ctx.ranking_metrics(ctx.recommend(hybrid, k), k)
        rows.append({"w_genre": wg, "w_popularity": wp, **m})
        with mlflow.start_run(run_name=f"tune_g{wg}_p{wp}", nested=True):
            mlflow.log_params({"w_genre": wg, "w_popularity": wp})
            mlflow.log_metrics({f"val_{a}": float(b) for a, b in m.items()})
    res = pd.DataFrame(rows)
    metric = f"{hcfg['tune_metric']}_at_{k}"
    best = res.sort_values(metric, ascending=False).iloc[0]
    log.info("Best hybrid weights: w_genre=%.2f w_popularity=%.2f (val %s=%.4f)",
             best.w_genre, best.w_popularity, metric, best[metric])
    return {"w_cf": 1.0, "w_genre": float(best.w_genre), "w_popularity": float(best.w_popularity)}, res


def _log_and_save(model, name, params, seconds, cfg, mlflow, extra_artifacts=()):
    models_dir = path(cfg, "models_dir") / name
    fmeta = read_json(path(cfg, "features_dir") / "feature_metadata.json")
    raw_meta = read_json(path(cfg, "raw_dir") / "ingestion_metadata.json")
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"stage": "training", "model": name,
                         "feature_version": fmeta["feature_version"],
                         "raw_interactions_sha256": raw_meta["files"]["interactions"]["sha256"][:16]})
        mlflow.log_params({"model": name, **{k: v for k, v in params.items() if not isinstance(v, dict)}})
        mlflow.log_metric("train_seconds", seconds)
        if hasattr(model, "history") and model.history:
            for step, v in enumerate(model.history):
                mlflow.log_metric("train_rmse_residual", v, step=step)
        pkl = model.save(models_dir)
        mlflow.log_artifact(str(pkl), artifact_path="model")
        for a in extra_artifacts:
            mlflow.log_artifact(str(a))
        run_id = run.info.run_id
    write_json({"name": name, "class": type(model).__name__, "params": params,
                "train_seconds": seconds, "mlflow_run_id": run_id,
                "feature_version": fmeta["feature_version"]}, models_dir / "meta.json")
    return run_id


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("training", cfg)
    seed = cfg["project"]["seed"]
    set_seed(seed)
    mlflow = setup_mlflow(cfg)
    proc, reports = path(cfg, "processed_dir"), path(cfg, "reports_dir")

    train = pd.read_parquet(proc / "train.parquet")
    users = pd.read_parquet(proc / "users.parquet")
    items = pd.read_parquet(proc / "items.parquet")
    fs = FeatureStore(path(cfg, "features_dir"))
    tcfg = cfg["training"]
    summary = {}

    # -- 1. tune hybrid weights on a validation window inside train
    with mlflow.start_run(run_name="hybrid_tuning") as tune_run:
        mlflow.set_tag("stage", "tuning")
        best_w, grid = tune_hybrid(train, users, items, cfg, log, mlflow)
        grid_path = reports / "hybrid_tuning.csv"
        grid.to_csv(grid_path, index=False)
        mlflow.log_artifact(str(grid_path))
        mlflow.log_params({f"best_{k}": v for k, v in best_w.items()})

    # -- 2. final models on full train
    t = time.time(); pop = PopularityRecommender(**tcfg["popularity"]).fit(train, fs)
    summary["popularity"] = _log_and_save(pop, "popularity", tcfg["popularity"], time.time() - t, cfg, mlflow)

    mf_params = {**tcfg["mf"], "seed": seed}
    t = time.time(); mf = MatrixFactorization(**mf_params).fit(train, fs)
    summary["matrix_factorization"] = _log_and_save(mf, "matrix_factorization", mf_params,
                                                    time.time() - t, cfg, mlflow)
    log.info("MF trained (final residual RMSE %.4f)", mf.history[-1])

    ials_params = {**tcfg["implicit_als"], "seed": seed}
    t = time.time(); ials = ImplicitALS(**ials_params).fit(train, fs)
    ials_secs = time.time() - t
    summary["implicit_als"] = _log_and_save(ials, "implicit_als", ials_params, ials_secs, cfg, mlflow)

    t = time.time(); hyb = HybridRecommender(**best_w, cf_params=ials_params).fit(train, fs, base=ials)
    hyb_params = {**best_w, **{f"cf_{k}": v for k, v in ials_params.items()},
                  "tuned_on": "last 10% of train period"}
    summary["hybrid"] = _log_and_save(hyb, "hybrid", hyb_params, ials_secs + time.time() - t, cfg, mlflow,
                                      extra_artifacts=[grid_path])
    log.info("Trained models: %s", list(summary))
    write_json({"runs": summary, "best_hybrid_weights": best_w, "tuning_run": tune_run.info.run_id},
               path(cfg, "models_dir") / "training_summary.json")
    return summary


if __name__ == "__main__":
    run()
