"""Evaluate PathQwen2.5 on the held-out test set across all 9 tasks.

Compares predicted JSON fields (in pathology_struct.parquet) against gold
labels in qa_test.jsonl. Handles label-space mismatches by normalizing free-text
model outputs into the canonical label space the gold labels use.

Reports accuracy + macro-F1 + per-class classification report + confusion
matrices, paper-comparable to Saluja et al. 2025.

Outputs:
  models/PathQwen2.5/eval_pathology.json          metric summary
  figures/pathology_confusion_<task>.png          confusion matrix per task

Run:
  python -m src.evaluation.pathology_eval
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import weave
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from src._weave_init import init_weave
from src.training.schema import TASKS

sns.set_theme(style="whitegrid", context="notebook")

# ---------------------------------------------------------------------------
# 1. Canonical label spaces (must match build_multitask_qa.py)
# ---------------------------------------------------------------------------
STAGE_LABELS = ["Stage I", "Stage II", "Stage III", "Stage IV"]
T_LABELS = ["T0", "T1", "T2", "T3", "T4", "Tis", "TX"]
N_LABELS = ["N0", "N1", "N2", "N3", "NX"]
M_LABELS = ["M0", "M1", "MX"]


# Map common free-text cancer descriptions to TCGA studyId labels.
# Built from your cohort's actual mappings (studyId × PRIMARY_DIAGNOSIS).
CANCER_TYPE_KEYWORDS = {
    "brca_tcga_gdc":         ["breast", "ductal", "lobular", "mammary"],
    "luad_tcga_gdc":         ["lung adenocarcinoma", "lung adeno"],
    "lusc_tcga_gdc":         ["lung squamous", "lung scc"],
    "hnsc_tcga_gdc":         ["head and neck", "head & neck", "tongue", "larynx", "oropharyn"],
    "coad_tcga_gdc":         ["colon"],
    "read_tcga_gdc":         ["rectum", "rectal"],
    "stad_tcga_gdc":         ["stomach", "gastric"],
    "esca_tcga_gdc":         ["esophag"],
    "prad_tcga_gdc":         ["prostat"],
    "blca_tcga_gdc":         ["bladder", "urothelial"],
    "ccrcc_tcga_gdc":        ["clear cell renal", "kirc"],
    "prcc_tcga_gdc":         ["papillary renal", "kirp"],
    "chrcc_tcga_gdc":        ["chromophobe"],
    "ucec_tcga_gdc":         ["endometri", "uterine corpus"],
    "ucs_tcga_gdc":          ["carcinosarcoma"],
    "cesc_tcga_gdc":         ["cervix", "cervical"],
    "hgsoc_tcga_gdc":        ["ovary", "ovarian", "serous"],
    "hcc_tcga_gdc":          ["hepatocellular", "liver"],
    "chol_tcga_gdc":         ["cholangio", "bile duct"],
    "paad_tcga_gdc":         ["pancrea"],
    "thpa_tcga_gdc":         ["thyroid"],
    "gbm_tcga_gdc":          ["glioblastoma"],
    "difg_tcga_gdc":         ["glioma", "astrocytoma", "oligodendro"],
    "skcm_tcga":             ["melanoma", "skin cutaneous"],
    "um_tcga_gdc":           ["uveal"],
    "acc_tcga_gdc":          ["adrenocortical"],
    "mnet_tcga_gdc":         ["pheochromocyt", "paragangli"],
    "thym_tcga_gdc":         ["thymoma", "thymic"],
    "plmeso_tcga_gdc":       ["mesotheli"],
    "nsgct_tcga_gdc":        ["testicular", "germ cell"],
    "dlbclnos_tcga_gdc":     ["dlbcl", "diffuse large b"],
    "soft_tissue_tcga_gdc":  ["sarcoma", "soft tissue", "liposarcoma", "leiomyosarcoma"],
}


# ---------------------------------------------------------------------------
# 2. Normalization helpers
# ---------------------------------------------------------------------------
def normalize_cancer_type(pred: str | None) -> str | None:
    """Map free-text model output to a TCGA studyId.

    Strategy: lowercase, scan keyword table for matches, pick the studyId with
    the most distinctive (longest-matching) keyword. Returns None if no match.
    """
    if pred is None or pd.isna(pred):
        return None
    s = str(pred).lower().strip()
    if not s:
        return None
    # Direct match: already a studyId
    for sid in CANCER_TYPE_KEYWORDS:
        if sid in s:
            return sid
    # Keyword scan — prefer longest match (most specific)
    best_sid, best_len = None, 0
    for sid, kws in CANCER_TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in s and len(kw) > best_len:
                best_sid, best_len = sid, len(kw)
    return best_sid


def normalize_stage(pred: str | None) -> str | None:
    if pred is None or pd.isna(pred):
        return None
    s = str(pred).strip().upper()
    # Standardize "STAGE IIA" -> "Stage II", "STAGE IIIB" -> "Stage III"
    m = re.search(r"STAGE\s*(IV|III|II|I)", s)
    if m:
        return f"Stage {m.group(1)}"
    # Sometimes the model emits just "I"/"II"/"III"/"IV"
    if s in {"I", "II", "III", "IV"}:
        return f"Stage {s}"
    return None


def normalize_t(pred: str | None) -> str | None:
    if pred is None or pd.isna(pred):
        return None
    s = str(pred).strip().upper()
    if s in {"TIS", "TX"}:
        return "Tis" if s == "TIS" else "TX"
    m = re.search(r"\bT(\d)", s)
    if m:
        return f"T{m.group(1)}"
    return None


def normalize_n(pred: str | None) -> str | None:
    if pred is None or pd.isna(pred):
        return None
    s = str(pred).strip().upper()
    if s == "NX":
        return "NX"
    m = re.search(r"\bN(\d)", s)
    if m:
        return f"N{m.group(1)}"
    return None


def normalize_m(pred: str | None) -> str | None:
    if pred is None or pd.isna(pred):
        return None
    s = str(pred).strip().upper()
    if s.startswith("M0"):  return "M0"
    if s.startswith("M1"):  return "M1"
    if s.startswith("MX"):  return "MX"
    return None


def normalize_bool(pred) -> str | None:
    """Boolean fields (prior_malignancy, prognosis_good) -> 'True'/'False'."""
    if pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return None
    if isinstance(pred, bool):
        return str(pred)
    if isinstance(pred, pd.api.extensions.ExtensionArray.__class__):
        return None
    s = str(pred).strip().lower()
    if s in {"true", "yes", "1", "good", "favorable", "positive"}:
        return "True"
    if s in {"false", "no", "0", "poor", "unfavorable", "negative"}:
        return "False"
    return None


def normalize_freetext(pred) -> str | None:
    """For primary_site / histology — keep as lowercase strings, drop NaN."""
    if pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return None
    return str(pred).strip().lower() or None


NORMALIZERS = {
    "cancer_type":      normalize_cancer_type,
    "primary_site":     normalize_freetext,
    "histology":        normalize_freetext,
    "ajcc_stage":       normalize_stage,
    "t_stage":          normalize_t,
    "n_stage":          normalize_n,
    "m_stage":          normalize_m,
    "prior_malignancy": normalize_bool,
    "prognosis_good":   normalize_bool,
}


def normalize_gold(task: str, val) -> str | None:
    """Apply the same normalizer to gold labels for fair comparison."""
    fn = NORMALIZERS.get(task, normalize_freetext)
    return fn(val)


# ---------------------------------------------------------------------------
# 3. Confusion-matrix plotting
# ---------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, task: str, out_dir: Path, max_classes: int = 12):
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return
    if len(classes) > max_classes:
        # Keep top-K most-frequent classes; everything else -> "Other"
        from collections import Counter
        top = {c for c, _ in Counter(y_true).most_common(max_classes)}
        y_true = [c if c in top else "Other" for c in y_true]
        y_pred = [c if c in top else "Other" for c in y_pred]
        classes = sorted(top | {"Other"})

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(min(14, 1 + 0.7 * len(classes)),
                                    min(12, 1 + 0.6 * len(classes))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(f"Confusion matrix — {task}  (n={len(y_true)})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out = out_dir / f"pathology_confusion_{task}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
@weave.op
def main(struct_parquet: str, qa_test: str, out_path: str,
         figures_dir: str = "figures") -> None:
    pred_df = pd.read_parquet(struct_parquet).set_index("TCGA_Barcode")
    print(f"loaded {len(pred_df)} predicted records, {len(pred_df.columns)} task columns")

    # Read qa_test, fold per-patient/task gold answers into a dict
    gold = defaultdict(dict)
    n_lines = 0
    for line in Path(qa_test).open():
        if not line.strip():
            continue
        n_lines += 1
        r = json.loads(line)
        ans = r["messages"][-1]["content"]
        # Some assistant turns may include CoT — take the LAST balanced JSON object
        try:
            # Find last '{...}' block in the assistant string
            m = list(re.finditer(r"\{[^{}]*\}", ans))
            if not m:
                continue
            obj = json.loads(m[-1].group(0))
        except Exception:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                gold[r["TCGA_Barcode"]][k] = v
    print(f"parsed {n_lines} qa_test rows -> {len(gold)} unique patients with gold labels")

    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    results = {}
    overall_summary = []
    print()
    print(f"{'task':22s} {'n':>6s}  {'acc':>7s}  {'F1':>7s}  {'parsed':>6s}")
    print("-" * 60)
    for task in TASKS:
        y_true, y_pred = [], []
        n_unparsed = 0
        for pid, golds in gold.items():
            if task not in golds or pid not in pred_df.index:
                continue
            g = normalize_gold(task, golds[task])
            p = NORMALIZERS[task](pred_df.loc[pid, task])
            if g is None:
                continue
            if p is None:
                # Model prediction couldn't be normalized — count as misprediction
                p = "<unparseable>"
                n_unparsed += 1
            y_true.append(str(g))
            y_pred.append(str(p))

        if not y_true:
            results[task] = {"n": 0}
            continue

        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        n = len(y_true)
        results[task] = {
            "n": n,
            "n_unparseable_predictions": n_unparsed,
            "accuracy": float(acc),
            "macro_f1": float(f1m),
            "per_class": classification_report(y_true, y_pred,
                                               output_dict=True, zero_division=0),
            "label_space_size": len(set(y_true)),
        }
        plot_confusion(y_true, y_pred, task, figures)
        print(f"{task:22s} {n:6d}  {acc:7.4f}  {f1m:7.4f}  {n_unparsed:6d}")
        overall_summary.append({"task": task, "n": n, "accuracy": acc,
                                "macro_f1": f1m})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_path}")

    # --- Bar chart of all 9 tasks ---
    df = pd.DataFrame(overall_summary)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        df_long = df.melt(id_vars=["task", "n"], value_vars=["accuracy", "macro_f1"],
                          var_name="Metric", value_name="Score")
        sns.barplot(df_long, y="task", x="Score", hue="Metric", ax=ax,
                    palette={"accuracy": "#5B7DB1", "macro_f1": "#C25450"})
        ax.set_xlim(0, 1.0)
        ax.set_title("PathQwen2.5 — Test set performance per task")
        ax.axvline(0.5, ls="--", color="grey", alpha=0.5, label="random baseline")
        for i, row in df.iterrows():
            ax.text(0.005, i, f"n={row['n']}", va="center", fontsize=9, color="black")
        fig.tight_layout()
        fig.savefig(figures / "pathology_tasks.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {figures/'pathology_tasks.png'}")

    # --- Comparison panel vs Saluja 2025 ---
    paper = {"cancer_type": 0.96, "ajcc_stage": 0.85, "prognosis_good": 0.48}
    cmp_rows = []
    for task, paper_val in paper.items():
        if task not in results or results[task].get("n", 0) == 0:
            continue
        ours = (results[task]["accuracy"] if task != "prognosis_good"
                else results[task]["macro_f1"])
        cmp_rows.append({"Task": task, "Source": "Saluja 2025", "Score": paper_val})
        cmp_rows.append({"Task": task, "Source": "PathQwen2.5 (ours)", "Score": ours})
    if cmp_rows:
        cmp = pd.DataFrame(cmp_rows)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(cmp, x="Task", y="Score", hue="Source", ax=ax,
                    palette={"Saluja 2025": "#999999", "PathQwen2.5 (ours)": "#440154"})
        ax.set_ylim(0, 1.0)
        ax.set_title("PathQwen2.5 vs Saluja 2025 — head-to-head on test")
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
        fig.tight_layout()
        fig.savefig(figures / "pathology_vs_saluja.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {figures/'pathology_vs_saluja.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", default="data/processed/features/pathology_struct.parquet")
    ap.add_argument("--qa-test", default="data/processed/pathology/qa_test.jsonl")
    ap.add_argument("--output", default="models/PathQwen2.5/eval_pathology.json")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()
    init_weave("pathology-eval")
    main(args.struct, args.qa_test, args.output, args.figures)
