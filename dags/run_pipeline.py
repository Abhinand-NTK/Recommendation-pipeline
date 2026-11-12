"""Plain-Python orchestrator (no extra services needed).

Runs the pipeline stages as a dependency graph in topological order, records the
status / duration of every task, and writes reports/pipeline_run.json.

    python dags/run_pipeline.py                     # full run
    python dags/run_pipeline.py --from-stage train  # re-run train and everything downstream
    python dags/run_pipeline.py --only evaluate     # one stage

For the Prefect version (UI, retries, run history) see dags/recommendation_flow.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import traceback
from datetime import datetime, timezone
from graphlib import TopologicalSorter

from src.common.utils import get_logger, load_config, path, set_seed, write_json
from src.evaluation import evaluate
from src.features import build_features
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

STAGES = {
    "ingest": (ingest.run, []),
    "validate": (validate.run, ["ingest"]),
    "prepare": (prepare.run, ["validate"]),
    "transform": (transform.run, ["prepare"]),
    "features": (build_features.run, ["transform"]),
    "train": (train.run, ["features"]),
    "evaluate": (evaluate.run, ["train"]),
}


def downstream(start: str) -> set[str]:
    out, frontier = {start}, [start]
    while frontier:
        cur = frontier.pop()
        for name, (_, deps) in STAGES.items():
            if cur in deps and name not in out:
                out.add(name)
                frontier.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from-stage", choices=list(STAGES))
    g.add_argument("--only", choices=list(STAGES))
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    log = get_logger("orchestrator", cfg)
    order = list(TopologicalSorter({k: v[1] for k, v in STAGES.items()}).static_order())
    selected = ({args.only} if args.only else downstream(args.from_stage) if args.from_stage else set(order))

    run_log = {"started_utc": datetime.now(timezone.utc).isoformat(), "tasks": []}
    for name in order:
        if name not in selected:
            continue
        fn, deps = STAGES[name]
        log.info("==> %s (depends on %s)", name, deps or "-")
        t0 = time.time()
        try:
            fn(cfg)
            run_log["tasks"].append({"task": name, "status": "success", "seconds": round(time.time() - t0, 1)})
        except Exception as exc:
            run_log["tasks"].append({"task": name, "status": "failed", "error": repr(exc)})
            log.error("Task %s failed:\n%s", name, traceback.format_exc())
            write_json(run_log, path(cfg, "reports_dir") / "pipeline_run.json")
            raise SystemExit(1)
    run_log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_log, path(cfg, "reports_dir") / "pipeline_run.json")
    log.info("Pipeline finished: %s", ", ".join(f"{t['task']}={t['seconds']}s" for t in run_log["tasks"]))


if __name__ == "__main__":
    main()
