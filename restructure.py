"""
Restructures Neuro_Project into a professional research/enterprise layout.
Run once from the project root: python restructure.py
"""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# ─── 1. Create new directory tree ─────────────────────────────────────────────
NEW_DIRS = [
    "data/raw",
    "data/processed",
    "models/artifacts",
    "models/reports/figures",
    "src/stroke_prediction/data",
    "src/stroke_prediction/models",
    "src/stroke_prediction/visualization",
    "tests",
]
for d in NEW_DIRS:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

print("✓ Directory tree created")

# ─── 2. Move data file ────────────────────────────────────────────────────────
old_csv = ROOT / "data/raw_data/healthcare-dataset-stroke-data.csv"
new_csv = ROOT / "data/raw/healthcare-dataset-stroke-data.csv"
if old_csv.exists() and not new_csv.exists():
    shutil.copy2(old_csv, new_csv)
    shutil.rmtree(ROOT / "data/raw_data", ignore_errors=True)
    print("✓ Dataset moved  →  data/raw/")

# ─── 3. Move model outputs ────────────────────────────────────────────────────
src_out = ROOT / "models/stroke_outputs"
if src_out.exists():
    for f in src_out.iterdir():
        if f.suffix == ".png":
            shutil.copy2(f, ROOT / "models/reports/figures" / f.name)
        elif f.suffix == ".csv":
            shutil.copy2(f, ROOT / "models/reports" / f.name)
    shutil.rmtree(src_out)
    print("✓ Benchmark CSVs  →  models/reports/")
    print("✓ Figures         →  models/reports/figures/")

# Move pkl + json artifacts
for artifact in ["best_stroke_model.pkl", "preprocessor.pkl", "model_metadata.json"]:
    src = ROOT / "models" / artifact
    dst = ROOT / "models/artifacts" / artifact
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
print("✓ Model artifacts  →  models/artifacts/")

# ─── 4. Split notebook into EDA + Modeling ───────────────────────────────────
nb_src = ROOT / "notebooks/01_eda_.ipynb"
with open(nb_src, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
SPLIT_AT = next(
    i for i, c in enumerate(cells)
    if "Preprocessing" in "".join(c["source"]) and c["cell_type"] == "markdown"
)

def make_nb(cell_list, title_md):
    return {
        "nbformat": nb["nbformat"],
        "nbformat_minor": nb["nbformat_minor"],
        "metadata": nb["metadata"],
        "cells": [{"cell_type":"markdown","id":"nb-title","metadata":{},
                   "source":[title_md]}] + cell_list,
    }

# Update data path inside the EDA notebook cells
eda_cells  = cells[:SPLIT_AT]
model_cells = cells[SPLIT_AT:]

for cell in eda_cells:
    src = "".join(cell["source"])
    if "raw_data" in src:
        fixed = src.replace(
            r"C:\Users\Beddad\Desktop\Neuro_Project\data\raw_data\healthcare-dataset-stroke-data.csv",
            "../data/raw/healthcare-dataset-stroke-data.csv"
        ).replace("raw_data", "raw")
        cell["source"] = [fixed]

for cell in model_cells:
    src = "".join(cell["source"])
    if "stroke_outputs" in src or "raw_data" in src:
        fixed = (src
            .replace("../models/stroke_outputs", "../models/reports")
            .replace("stroke_outputs", "reports")
            .replace(
                r"C:\Users\Beddad\Desktop\Neuro_Project\data\raw_data\healthcare-dataset-stroke-data.csv",
                "../data/raw/healthcare-dataset-stroke-data.csv"
            )
            .replace("raw_data", "raw")
        )
        cell["source"] = [fixed]
    # inject sys.path for src package (first code cell only)
    if cell["cell_type"] == "code" and "import" in "".join(cell["source"]):
        s = "".join(cell["source"])
        if "sys.path" not in s:
            cell["source"] = [
                "import sys, pathlib\n"
                "sys.path.insert(0, str(pathlib.Path('..') / 'src'))\n\n"
            ] + cell["source"]
        break  # only first code cell

nb1 = make_nb(eda_cells,  "# Stroke Risk Prediction — Exploratory Data Analysis")
nb2 = make_nb(model_cells, "# Stroke Risk Prediction — Modelling & Evaluation")

with open(ROOT / "notebooks/01_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, ensure_ascii=False, indent=1)
with open(ROOT / "notebooks/02_modeling_and_evaluation.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, ensure_ascii=False, indent=1)

nb_src.unlink()
print("✓ Notebooks split  →  01_exploratory_data_analysis.ipynb")
print("                       02_modeling_and_evaluation.ipynb")

print("✓ Old notebook removed")

# ─── 5. Update app.py data path ───────────────────────────────────────────────
app_path = ROOT / "app.py"
app_src  = app_path.read_text(encoding="utf-8")
app_src  = app_src.replace(
    'DATA_PATH          = Path("data/raw_data/healthcare-dataset-stroke-data.csv")',
    'DATA_PATH          = Path("data/raw/healthcare-dataset-stroke-data.csv")',
).replace(
    'DATA_PATH = Path("data/raw_data/healthcare-dataset-stroke-data.csv")',
    'DATA_PATH = Path("data/raw/healthcare-dataset-stroke-data.csv")',
)
app_path.write_text(app_src, encoding="utf-8")
print("✓ app.py data path updated")

print("\nAll done. Writing source package…")
