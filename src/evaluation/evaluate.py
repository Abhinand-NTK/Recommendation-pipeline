"""Stage 7 - Model evaluation.

Evaluates every trained model on the held-out (future) test period:

    ranking : Precision@K, Recall@K, NDCG@K, HitRate@K   (K = 10 and 5)
    beyond-accuracy : catalog Coverage, intra-list Diversity (genres), Novelty
    rating  : RMSE, MAE (models that predict explicit ratings)

Protocol
    * candidate items = all items seen in training, minus the user's train items
    * evaluated users = test users that also appear in train and have >= 1
      relevant (rating >= 4) test item  (warm-start users)
    * metrics are logged to the model's MLflow run and written to
      reports/model_comparison.{csv,md} plus figures in reports/figures/
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.utils import PROJECT_ROOT, get_logger, load_config, path, read_json, write_json
from src.evaluation import metrics as M
from src.features.feature_store import FeatureStore

MODEL_ORDER = ["popularity", "matrix_factorization", "implicit_als", "hybrid"]


# --------------------------------------------------------------------------- shared
class EvalContext:
    """Pre-computes everything needed to score a model against a hold-out set."""

    def __init__(self, train: pd.DataFrame, holdout: pd.DataFrame, fs: FeatureStore,
                 rel_threshold: float, max_users: int | None = None, seed: int = 42):
        self.train, self.holdout = train, holdout
        self.item_ids = np.sort(train["item_id"].unique())
        i_pos = pd.Series(np.arange(len(self.item_ids)), index=self.item_ids)
        warm = set(train["user_id"])
        h = holdout[holdout["user_id"].isin(warm)]
        self.relevant = h[h["rating"] >= rel_threshold].groupby("user_id")["item_id"].apply(set)
        users = self.relevant.index.to_numpy()
        if max_users and len(users) > max_users:
            users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
        self.users = users
        self.seen = {u: i_pos[g].to_numpy() for u, g in train.groupby("user_id")["item_id"]}
        gcols = fs.genre_cols
        gv = fs.get_item_features(self.item_ids, gcols).fillna(0).to_numpy()
        self.item_vectors = dict(zip(self.item_ids, gv))
        counts = train["item_id"].value_counts()
        self.pop_share = (counts / counts.sum()).to_dict()
        self.activity = train["user_id"].value_counts()

    def recommend(self, model, k: int, batch: int = 512) -> np.ndarray:
        return np.vstack([model.recommend(self.users[s:s + batch], k, self.seen)
                          for s in range(0, len(self.users), batch)])

    def ranking_metrics(self, recs: np.ndarray, k: int) -> dict:
        rows = []
        for r, u in zip(recs, self.users):
            rel, lst = self.relevant[u], list(r[:k])
            rows.append((M.precision_at_k(lst, rel, k), M.recall_at_k(lst, rel, k),
                         M.ndcg_at_k(lst, rel, k), M.hit_rate_at_k(lst, rel, k)))
        p, r, n, h = np.mean(rows, axis=0)
        top = recs[:, :k]
        return {f"precision_at_{k}": p, f"recall_at_{k}": r, f"ndcg_at_{k}": n, f"hit_rate_at_{k}": h,
                f"coverage_at_{k}": M.catalog_coverage(top, len(self.item_ids)),
                f"diversity_at_{k}": M.intra_list_diversity(top, self.item_vectors),
                f"novelty_at_{k}": M.novelty(top, self.pop_share)}

    def per_user_ndcg(self, recs: np.ndarray, k: int) -> pd.Series:
        return pd.Series([M.ndcg_at_k(list(r[:k]), self.relevant[u], k) for r, u in zip(recs, self.users)],
                         index=self.users)

    def rating_metrics(self, model) -> dict:
        known = self.holdout[self.holdout["user_id"].isin(model.user_ids)
                             & self.holdout["item_id"].isin(model.item_ids)]
        pred = model.predict_rating(known["user_id"].to_numpy(), known["item_id"].to_numpy())
        if pred is None:
            return {}
        return {"rmse": M.rmse(known["rating"], pred), "mae": M.mae(known["rating"], pred),
                "rating_eval_pairs": float(len(known))}


# --------------------------------------------------------------------------- reporting
def _plots(df: pd.DataFrame, seg: pd.DataFrame, fig_dir: Path, k: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = df.index.tolist()
    colors = ["#9aa5b1", "#4c78a8", "#f58518", "#54a24b"][: len(names)]
    panels = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"coverage_at_{k}", f"diversity_at_{k}"]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.6))
    for ax, m in zip(axes, panels):
        ax.bar(names, df[m], color=colors)
        ax.set_title(m.replace("_", " ").replace("at ", "@"), fontsize=11)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        for i, v in enumerate(df[m]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Model comparison on the future test period (K={k})", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    seg.plot(kind="bar", ax=ax, color=colors, width=0.8)
    ax.set_ylabel(f"NDCG@{k}")
    ax.set_xlabel("User activity in training (number of ratings)")
    ax.set_title(f"NDCG@{k} by user activity segment")
    ax.tick_params(axis="x", rotation=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "ndcg_by_user_activity.png", dpi=150)
    plt.close(fig)


def _markdown(df: pd.DataFrame, k: int, extra_k: list, ctx: EvalContext, seg: pd.DataFrame) -> str:
    def fmt(v):
        return "n/a" if pd.isna(v) else f"{v:.4f}"

    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"hit_rate_at_{k}",
            f"coverage_at_{k}", f"diversity_at_{k}", f"novelty_at_{k}", "rmse", "mae"]
    head = ["Model", f"P@{k}", f"R@{k}", f"NDCG@{k}", f"HR@{k}", "Coverage", "Diversity", "Novelty",
            "RMSE", "MAE"]
    lines = ["# Model Evaluation Results", "",
             f"Evaluated users: **{len(ctx.users):,}** warm-start users with at least one relevant "
             f"(rating >= 4) item in the future test period. Candidates: {len(ctx.item_ids):,} items "
             "minus each user's training items.", "",
             f"## Main comparison (K = {k})", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, row in df.iterrows():
        lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    for ek in extra_k:
        c2 = [f"precision_at_{ek}", f"recall_at_{ek}", f"ndcg_at_{ek}", f"coverage_at_{ek}", f"diversity_at_{ek}"]
        lines += ["", f"## K = {ek}", "", f"| Model | P@{ek} | R@{ek} | NDCG@{ek} | Coverage | Diversity |",
                  "|---|---|---|---|---|---|"]
        for name, row in df.iterrows():
            lines.append(f"| {name} | " + " | ".join(fmt(row.get(c)) for c in c2) + " |")
    lines += ["", f"## NDCG@{k} by user activity segment", "",
              "| Segment | " + " | ".join(seg.columns) + " |", "|" + "---|" * (len(seg.columns) + 1)]
    for s, row in seg.iterrows():
        lines.append(f"| {s} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    lines += ["", "Metric notes: Coverage = share of catalog recommended to at least one user; "
              "Diversity = mean pairwise genre dissimilarity inside a list (0-1); Novelty = mean "
              "-log2(item popularity share); RMSE/MAE only for models that predict explicit ratings "
              "(implicit ALS and hybrid output preference scores, so they are n/a).", "",
              "![comparison](figures/model_comparison.png)", "",
              "![segments](figures/ndcg_by_user_activity.png)", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- stage
def run(cfg: dict | None = None) -> pd.DataFrame:
    import mlflow

    from src.training.models import BaseRecommender
    from src.training.train import setup_mlflow

    cfg = cfg or load_config()
    log = get_logger("evaluation", cfg)
    proc, models_dir, reports = path(cfg, "processed_dir"), path(cfg, "models_dir"), path(cfg, "reports_dir")
    fig_dir = reports / "figures"
    fig_dir.mkdir(exist_ok=True)
    k, extra_k = cfg["evaluation"]["k"], cfg["evaluation"]["extra_k"]

    train = pd.read_parquet(proc / "train.parquet")
    test = pd.read_parquet(proc / "test.parquet")
    fs = FeatureStore(path(cfg, "features_dir"))
    ctx = EvalContext(train, test, fs, cfg["preparation"]["relevance_threshold"],
                      cfg["evaluation"]["max_users"], cfg["project"]["seed"])
    log.info("Evaluating on %d users", len(ctx.users))
    setup_mlflow(cfg)

    results, per_user = {}, {}
    for name in MODEL_ORDER:
        mdir = models_dir / name
        if not (mdir / "model.pkl").exists():
            continue
        model = BaseRecommender.load(mdir, fs)
        meta = read_json(mdir / "meta.json")
        kmax = max([k, *extra_k])
        recs = ctx.recommend(model, kmax)
        res = {}
        for kk in sorted({k, *extra_k}):
            res.update(ctx.ranking_metrics(recs, kk))
        res.update(ctx.rating_metrics(model))
        res["evaluated_users"] = float(len(ctx.users))
        results[name] = res
        per_user[name] = ctx.per_user_ndcg(recs, k)
        with mlflow.start_run(run_id=meta["mlflow_run_id"]):
            mlflow.log_metrics({f"test_{m}": float(v) for m, v in res.items()})
        log.info("%s: P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.3f div=%.3f", name, k,
                 res[f"precision_at_{k}"], k, res[f"recall_at_{k}"], k, res[f"ndcg_at_{k}"],
                 res[f"coverage_at_{k}"], res[f"diversity_at_{k}"])

    df = pd.DataFrame(results).T
    df.index.name = "model"
    df.to_csv(reports / "model_comparison.csv")
    act = ctx.activity.reindex(ctx.users)
    bins = pd.cut(act, [0, 50, 150, 400, np.inf], labels=["<=50", "51-150", "151-400", ">400"])
    seg = pd.DataFrame(per_user).groupby(bins.to_numpy(), observed=True).mean()
    seg.index.name = "activity"
    seg.to_csv(reports / "ndcg_by_user_activity.csv")
    _plots(df, seg, fig_dir, k)
    (reports / "model_comparison.md").write_text(_markdown(df, k, extra_k, ctx, seg), encoding="utf-8")
    write_json({"k": k, "evaluated_users": len(ctx.users), "results": results}, reports / "evaluation_results.json")

    # log the comparison table to a summary run as well
    with mlflow.start_run(run_name="evaluation_summary"):
        mlflow.set_tag("stage", "evaluation")
        mlflow.log_artifact(str(reports / "model_comparison.md"))
        mlflow.log_artifact(str(fig_dir / "model_comparison.png"))
    log.info("Comparison written to reports/model_comparison.md")
    return df


if __name__ == "__main__":
    run()
