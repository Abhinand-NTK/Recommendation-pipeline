"""Render reports/architecture_diagram.png (requires the Graphviz `dot` binary)."""
from pathlib import Path

import graphviz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "architecture_diagram"

g = graphviz.Digraph("pipeline", format="png")
g.attr(rankdir="LR", fontname="Helvetica", fontsize="12", nodesep="0.35", ranksep="0.45", dpi="160",
       label="End-to-end data management pipeline for a MovieLens recommender  (orchestrated by Prefect)",
       labelloc="t")
g.attr("node", fontname="Helvetica", fontsize="10", style="rounded,filled", shape="box", color="#4c5a67")
g.attr("edge", color="#4c5a67", arrowsize="0.7")

store = dict(shape="cylinder", fillcolor="#eef2f6")
stage = dict(fillcolor="#dbe9f7")
side = dict(fillcolor="#fde9d7")

g.node("src", "MovieLens-1M\n(GroupLens zip /\nGitHub mirror / local CSV)", shape="folder", fillcolor="#f2f2f2")
g.node("ing", "1. Ingestion\nsrc/ingestion\n+ fault injection", **stage)
g.node("raw", "data/raw\nusers / items /\ninteractions .csv\n+ SHA-256 metadata", **store)
g.node("val", "2. Validation\nrules + Pandera\ncontract", **stage)
g.node("err", "data/errors\n*_errors.csv\n(row + reason)", **store)
g.node("vrep", "reports/\nvalidation_report.md", shape="note", fillcolor="#fff7d6")
g.node("prep", "3. Preparation\nclean, impute,\ndedupe, time split", **stage)
g.node("proc", "data/processed\ntrain / test .parquet", **store)
g.node("trans", "4. Transformation\nactivity summary,\nitem popularity,\nuser-item matrix", **stage)
g.node("fe", "5. Feature store\nuser / item /\ninteraction features", **stage)
g.node("fs", "data/features\n.parquet + catalog\n(offline + lookup API)", **store)
g.node("train", "6. Training\nPopularity | MF (ALS)\nImplicit ALS | Hybrid\n(tuned on val window)", **stage)
g.node("mlf", "MLflow\nparams, metrics,\nartifacts", **side)
g.node("models", "models/\n*.pkl + meta.json", **store)
g.node("eval", "7. Evaluation\nP/R/NDCG@K, coverage,\ndiversity, RMSE/MAE", **stage)
g.node("rep", "reports/\nmodel_comparison.md\n+ figures", shape="note", fillcolor="#fff7d6")
g.node("serve", "src/predict.py\nTop-K for a user", **side)

for a, b in [("src", "ing"), ("ing", "raw"), ("raw", "val"), ("val", "prep"), ("prep", "proc"),
             ("proc", "trans"), ("trans", "fe"), ("fe", "fs"), ("fs", "train"), ("train", "models"),
             ("models", "eval"), ("eval", "rep")]:
    g.edge(a, b)
g.edge("val", "err", style="dashed", label="invalid")
g.edge("val", "vrep", style="dashed")
g.edge("train", "mlf", style="dashed")
g.edge("eval", "mlf", style="dashed")
g.edge("fs", "serve", style="dashed", label="same features")
g.edge("models", "serve", style="dashed")
g.render(str(OUT), cleanup=True)
print(f"Wrote {OUT}.png")

# compact two-row version for the PDF report
c = g.copy()
c.attr(rankdir="TB", label="", ranksep="0.35", nodesep="0.25")
with c.subgraph() as s1:
    s1.attr(rank="same")
    for n in ["src", "ing", "raw", "val", "prep", "proc"]:
        s1.node(n)
with c.subgraph() as s2:
    s2.attr(rank="same")
    for n in ["trans", "fe", "fs", "train", "models", "eval", "rep"]:
        s2.node(n)
c.render(str(ROOT / "reports" / "figures" / "architecture_compact"), cleanup=True)
print("Wrote reports/figures/architecture_compact.png")
