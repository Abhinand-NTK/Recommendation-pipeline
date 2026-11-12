"""Build reports/Final_Report_Abhinand_NTK_2025ag05637.docx (convert to PDF with Word / LibreOffice)."""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports"
res = pd.read_csv(R / "model_comparison.csv", index_col=0)
val = json.loads((R / "validation_report.json").read_text())
split = json.loads((ROOT / "data/processed/split_metadata.json").read_text())
fmeta = json.loads((ROOT / "data/features/feature_metadata.json").read_text())
best = json.loads((ROOT / "models/training_summary.json").read_text())["best_hybrid_weights"]

doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2)
sec.top_margin = sec.bottom_margin = Cm(1.6)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10)
st.paragraph_format.space_after = Pt(3)


def h(text, lvl=1):
    p = doc.add_heading(text, level=lvl)
    for r in p.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)


def clean(text):
    import re
    return re.sub(r"(?<!\*)\*(?!\*)", "", text.replace("`", ""))


def para(text):
    text = clean(text)
    p = doc.add_paragraph()
    parts = text.split("**")
    for i, part in enumerate(parts):
        r = p.add_run(part)
        r.bold = i % 2 == 1
    return p


def bullet(text):
    text = clean(text)
    p = doc.add_paragraph(style="List Bullet")
    parts = text.split("**")
    for i, part in enumerate(parts):
        p.add_run(part).bold = i % 2 == 1
    p.paragraph_format.space_after = Pt(1)


def table(rows, header, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for c, txt in zip(t.rows[0].cells, header):
        c.text = ""
        c.paragraphs[0].add_run(txt).bold = True
    for row in rows:
        cells = t.add_row().cells
        for c, txt in zip(cells, row):
            c.text = str(txt)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(8.5)
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("End-to-End Data Management Pipeline for a Recommendation System")
r.bold, r.font.size = True, Pt(15)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run("DM4ML – Assignment 2 | Name: Abhinand NTK | ID: 2025ag05637 | "
            "Repo: https://github.com/Abhinand-NTK/Recommendation-pipeline").italic = True
sub.runs[0].font.size = Pt(8.5)

h("1. Problem and dataset")
s = val["summary"]
para(f"The goal is to recommend movies a user is likely to enjoy, using past ratings and item/user metadata. "
     f"The pipeline uses the public **MovieLens-1M** dataset (6,040 users, 3,883 movies, 1,000,209 ratings on "
     f"a 1–5 scale, 2000–2003), which provides all three required entities: user details (gender, age band, "
     f"occupation, zip code), item details (title, year, genres) and timestamped user–item interactions. "
     f"Because the real data is already clean, ingestion appends a documented, seeded set of "
     f"{sum(val['injected_faults'].values()):,} corrupt records (missing/invalid ratings, bad timestamps, "
     f"duplicates, orphan IDs) so the validation layer is exercised on realistic failures; this can be "
     f"switched off in the config.")

h("2. Architecture and design decisions")
doc.add_picture(str(R / "figures/architecture_compact.png"), width=Cm(17))
para("Each stage is an independent module with a `run(cfg)` entry point that reads its inputs from disk and "
     "writes versioned outputs, so stages can be re-run in isolation and orchestrated as a DAG. All "
     "parameters live in one YAML config. Key decisions:")
bullet("**Layered storage** – raw (CSV, kept as text so type errors survive) → validated → processed "
       "(Parquet) → features → models, plus a separate error layer. Raw files are SHA-256 hashed.")
bullet(f"**Validation** – 29 rules across three tables. ERROR-severity rows are moved to `data/errors/` with "
       f"a machine-readable reason, WARNING rows (e.g. missing gender/genre) are kept and imputed later. "
       f"Referential integrity is checked against the *validated* user/item tables, and a Pandera schema "
       f"contract guards the clean output. Result: users {s['users']['pass_rate']:.2%}, items "
       f"{s['items']['pass_rate']:.2%}, interactions {s['interactions']['pass_rate']:.2%} pass; "
       f"{s['interactions']['rejected_rows']:,} interactions rejected.")
bullet(f"**Time-based split** – a global cut-off at the 80th time percentile "
       f"({split['cutoff_utc'][:10]}): {split['train_rows']:,} train vs {split['test_rows']:,} test rows. "
       f"This mimics production (train on the past, predict the future) and prevents leakage; a per-user "
       f"chronological split is available as an option.")
bullet(f"**Feature store** – a custom, file-based layer: {fmeta['n_user_features']} user, "
       f"{fmeta['n_item_features']} item and {fmeta['n_interaction_features']} interaction features computed "
       f"as-of the cut-off from train data only. A Python registry is the single source of truth and generates "
       f"the feature catalog. The `FeatureStore` API (with defaults for cold users) is used by the hybrid "
       f"model during training, by tuning, and by `predict.py` at serving time.")
bullet(f"**Tuning without test leakage** – hybrid weights are grid-searched on the last 10% of the *train* "
       f"period with features rebuilt for that window; best: genre {best['w_genre']}, popularity "
       f"{best['w_popularity']}. Every grid point is a nested MLflow run.")
bullet("**Orchestration** – a Prefect 3 flow with explicit task dependencies, retries on ingestion and an "
       "optional daily schedule; plus a dependency-aware plain-Python runner (`--from-stage`, `--only`).")
EV = R / "screenshots"
doc.add_picture(str(EV / "prefect_run_graph.png"), width=Cm(13.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cr = cap.add_run("Successful Prefect run (steel-gaur): all 7 tasks Completed in dependency order, 30 m 28 s.")
cr.italic, cr.font.size = True, Pt(8.5)

h("3. Models")
table([["Popularity (baseline)", "Ranks items by number of ratings; Bayesian-average rating as rating prediction"],
       ["Matrix factorization", "Explicit biased MF, r = μ + b_u + b_i + p_u·q_i, 64 factors, ALS, 15 iterations"],
       ["Implicit ALS", "Weighted MF on implicit 'watched' feedback (Hu et al., 2008), 128 factors, λ=100"],
       ["Hybrid", "z-score blend of implicit-ALS score + user–item genre affinity (feature store) + popularity prior"]],
      ["Model", "Description"], [4, 13])

h("4. Results")
k = 10
rows = []
for m, rr in res.iterrows():
    f = lambda c: "n/a" if pd.isna(rr.get(c)) else f"{rr[c]:.4f}"
    rows.append([m, f(f"precision_at_{k}"), f(f"recall_at_{k}"), f(f"ndcg_at_{k}"), f(f"coverage_at_{k}"),
                 f(f"diversity_at_{k}"), f("rmse"), f("mae")])
para(f"Evaluated on {int(res['evaluated_users'].iloc[0]):,} warm-start test users (relevant = rating ≥ 4; "
     f"candidates exclude items already seen in training).")
table(rows, ["Model", "P@10", "R@10", "NDCG@10", "Coverage", "Diversity", "RMSE", "MAE"])
doc.add_picture(str(R / "figures/model_comparison.png"), width=Cm(17))
para("**Findings.** (1) Under a strict future split, popularity is a strong baseline: the test period is "
     "dominated by heavily-rated titles. (2) Explicit MF gives the best rating accuracy (RMSE 0.894 vs 0.961) "
     "but ranks poorly, because predicting a 5-star rating for a niche film is different from predicting what "
     "a user will watch next. (3) Implicit ALS, which models *what* users choose, beats popularity on every "
     "accuracy metric while covering ~3× more of the catalogue. (4) The hybrid gives the best precision, "
     "recall and NDCG, at the cost of coverage – a classic accuracy vs. coverage trade-off that the weights "
     "control. The genre signal adds little on top of collaborative filtering for warm users.")

h("5. Limitations and future work")
bullet("Only warm users are ranked; cold-start users (in test but not train) would need the content/"
       "demographic features as a fallback model.")
bullet("Hybrid tuning favours popularity, which lowers catalogue coverage; a diversity or coverage term in "
       "the objective (or re-ranking) would balance this.")
bullet("MovieLens-1M is dated and dense; fault injection simulates, but does not fully reproduce, real "
       "production data issues. The feature store is file-based (no online low-latency serving); Feast "
       "or a DB-backed store would be the next step.")
bullet("Hyper-parameters for the MF models were set by hand; a broader search (e.g. Optuna) on the "
       "validation window could improve results.")

h("6. Tools")
para("Python 3.13, Pandas, NumPy/SciPy (ALS implementations), Pandera (schema contracts), PyArrow/Parquet "
     "(storage), MLflow (tracking: params, metrics, artifacts, nested tuning runs), Prefect 3 "
     "(orchestration), Matplotlib & Graphviz (figures), pytest (unit tests).")

from docx.enum.text import WD_BREAK
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
h("Appendix – Evidence of execution")


def fig_cell(cell, img, width_cm, caption):
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(EV / img), width=Cm(width_cm))
    c = cell.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.italic, r.font.size = True, Pt(8)


grid = doc.add_table(rows=2, cols=2)
fig_cell(grid.cell(0, 0), "prefect_task_runs.png", 7.6, "A1 – Prefect task runs: all 7 tasks Completed")
fig_cell(grid.cell(0, 1), "mlflow_runs.png", 9.0, "A2 – MLflow: one tracked run per model + nested tuning runs")
fig_cell(grid.cell(1, 0), "validation_report.png", 9.4, "A3 – Validation report (reports/validation_report.md)")
fig_cell(grid.cell(1, 1), "error_dataset.png", 6.4, "A4 – Error dataset with error_reason (data/errors/)")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run().add_picture(str(EV / "feature_catalog.png"), width=Cm(17))
c = doc.add_paragraph()
c.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = c.add_run("A5 – Feature catalog generated from the registry (reports/feature_catalog.md)")
r.italic, r.font.size = True, Pt(8)

out = R / "Final_Report_Abhinand_NTK_2025ag05637.docx"
doc.save(out)
print("Wrote", out)
