"""Build ~70k QA pairs for 8-task multi-task instruction tuning.

Per-task masking: a row missing a label contributes to other tasks only.
We do NOT drop rows; we build one QA per (patient, task) when the label is
non-null.

Output: data/processed/pathology/qa_train.jsonl + qa_val.jsonl + qa_test.jsonl
        each row = {"messages": [...], "task": str, "TCGA_Barcode": str}

Splits respect data/processed/splits/splits.json (locked).

The Qwen2.5 chat template is applied at training time by the trainer; we just
write the message list + role pattern compatible with both Llama-3.1 and Qwen2.5.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import weave
import yaml
from tqdm import tqdm

from src._weave_init import init_weave
from src.training.schema import PathologyExtraction, TASKS


SYSTEM_PROMPT = (
    "You are an expert pathology AI assistant. "
    "Analyze the pathology report below and extract the requested field. "
    "Respond ONLY with a single-line JSON object matching the requested schema field. "
    "Do not include any explanations, headers, or prose."
)


def collapse_t(s):
    if pd.isna(s):
        return None
    m = re.match(r"(T\d|Tis|TX)", str(s).strip())
    return m.group(1) if m else None


def collapse_n(s):
    if pd.isna(s):
        return None
    m = re.match(r"(N\d|NX)", str(s).strip())
    return m.group(1) if m else None


def collapse_m(s):
    if pd.isna(s):
        return None
    s = str(s).strip().upper()
    if s.startswith("M0"):
        return "M0"
    if s.startswith("M1"):
        return "M1"
    if s.startswith("MX"):
        return "MX"
    return None


def collapse_stage(s):
    if pd.isna(s):
        return None
    m = re.match(r"Stage\s*(IV|III|II|I)", str(s).strip())
    if not m:
        return None
    return f"Stage {m.group(1)}"


def compute_prognosis_label(df: pd.DataFrame) -> pd.Series:
    """True iff patient survived past mean OS_MONTHS of their studyId cohort."""
    df = df.copy()
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    means = df.groupby("studyId")["OS_MONTHS"].transform("mean")
    label = pd.Series([None] * len(df), index=df.index, dtype=object)
    has_data = df["OS_MONTHS"].notna() & means.notna()
    label.loc[has_data] = (df.loc[has_data, "OS_MONTHS"] >= means.loc[has_data]).astype(bool).tolist()
    return label


TASK_PROMPTS = {
    "cancer_type": "What is the TCGA study cancer type? Output: {\"cancer_type\": \"<label>\"}",
    "primary_site": "What is the anatomical primary site? Output: {\"primary_site\": \"<text>\"}",
    "histology": "What is the histological diagnosis (ICD-O-3 morphology)? Output: {\"histology\": \"<text>\"}",
    "ajcc_stage": "What is the AJCC overall pathological stage (Stage I/II/III/IV)? Output: {\"ajcc_stage\": \"<label>\"}",
    "t_stage": "What is the pathological T stage (T0–T4, Tis, TX)? Output: {\"t_stage\": \"<label>\"}",
    "n_stage": "What is the pathological N stage (N0–N3, NX)? Output: {\"n_stage\": \"<label>\"}",
    "m_stage": "What is the pathological M stage (M0, M1, MX)? Output: {\"m_stage\": \"<label>\"}",
    "prior_malignancy": "Did this patient have a prior malignancy? Output: {\"prior_malignancy\": <true|false>}",
    "prognosis_good": "Will this patient likely survive past the mean disease-specific survival time for their cancer type? Output: {\"prognosis_good\": <true|false>}",
}


def build_qa(row, task: str, label) -> dict:
    user_prompt = (
        f"## Pathology Report:\n{row['text']}\n\n"
        f"## Question:\n{TASK_PROMPTS[task]}"
    )
    answer = {task: label}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": json.dumps(answer)},
    ]
    return {
        "messages": messages,
        "task": task,
        "TCGA_Barcode": row["TCGA_Barcode"],
    }


@weave.op
def main(cfg: dict) -> None:
    root = Path(cfg["project_root"])
    df = pd.read_csv(root / cfg["paths"]["cohort_csv"], low_memory=False)
    splits = json.loads((root / cfg["paths"]["splits_dir"] / "splits.json").read_text())

    # Build label columns
    df["lab_cancer_type"] = df["studyId"]  # paper-comparable, 100% coverage
    df["lab_primary_site"] = df["PRIMARY_SITE_PATIENT"]
    df["lab_histology"] = df["MORPHOLOGY"]
    df["lab_ajcc_stage"] = df["PATH_STAGE"].apply(collapse_stage)
    df["lab_t_stage"] = df["PATH_T_STAGE"].apply(collapse_t)
    df["lab_n_stage"] = df["PATH_N_STAGE"].apply(collapse_n)
    df["lab_m_stage"] = df["PATH_M_STAGE"].apply(collapse_m)
    def _to_bool(x):
        s = str(x).lower()
        if s in ("yes", "true", "1"):
            return True
        if s in ("no", "false", "0"):
            return False
        return None
    df["lab_prior_malignancy"] = df["PRIOR_MALIGNANCY"].apply(_to_bool)
    df["lab_prognosis_good"] = compute_prognosis_label(df)

    label_cols = {
        "cancer_type": "lab_cancer_type",
        "primary_site": "lab_primary_site",
        "histology": "lab_histology",
        "ajcc_stage": "lab_ajcc_stage",
        "t_stage": "lab_t_stage",
        "n_stage": "lab_n_stage",
        "m_stage": "lab_m_stage",
        "prior_malignancy": "lab_prior_malignancy",
        "prognosis_good": "lab_prognosis_good",
    }

    # Coverage report
    print("=== per-task label coverage ===")
    for task, col in label_cols.items():
        n = df[col].notna().sum()
        print(f"  {task:20s}: {n} / {len(df)} ({100*n/len(df):.1f}%)")

    # Build QA rows
    out_dir = root / cfg["paths"]["pathology_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    files = {s: open(out_dir / f"qa_{s}.jsonl", "w") for s in counts}

    split_lookup = {}
    for s, ids in splits.items():
        for i in ids:
            split_lookup[i] = s

    for _, row in tqdm(df.iterrows(), total=len(df), desc="building QA"):
        s = split_lookup.get(row["TCGA_Barcode"])
        if s is None:
            continue  # patient not in any split (rare-stratum dropped)
        for task, col in label_cols.items():
            lab = row[col]
            if lab is None or (isinstance(lab, float) and np.isnan(lab)):
                continue
            qa = build_qa(row, task, lab if not isinstance(lab, np.bool_) else bool(lab))
            files[s].write(json.dumps(qa, default=str) + "\n")
            counts[s] += 1

    for f in files.values():
        f.close()

    print("\n=== QA pair counts per split ===")
    for s, n in counts.items():
        print(f"  {s}: {n}")
    print(f"total: {sum(counts.values())}")

    (out_dir / "qa_manifest.json").write_text(json.dumps({
        "tasks": TASKS,
        "counts": counts,
        "system_prompt": SYSTEM_PROMPT,
        "task_prompts": TASK_PROMPTS,
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    init_weave("pathology-qa")
    main(cfg)
