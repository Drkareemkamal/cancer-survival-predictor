"""Clinical feature engineering aligned to the harmonized 8459-patient cohort.

Features built (per RESEARCH.pdf section 2.2):
- AGE (numeric) + AGE_SQ (non-linear age effect)
- SEX (one-hot), RACE (one-hot), ETHNICITY (one-hot)
- TNM stage one-hots (PATH_T_STAGE, PATH_N_STAGE, PATH_M_STAGE) -> simplified
- AJCC PATH_STAGE one-hot (Stage I/II/III/IV with substages collapsed)
- TNM_COMPOSITE = numeric encoding of T+N+M for ordinal signal
- PRIOR_MALIGNANCY, PRIOR_TREATMENT (binary)
- studyId one-hot (cancer cohort indicator - very strong baseline signal)
- Missing indicators for stage variables

All features are standardized within the training split, applied to val/test.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collapse_stage(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"Stage\s*(I{1,3}V?|IV|X)", str(s).strip())
    if not m:
        return np.nan
    base = m.group(1)
    if base == "X":
        return np.nan
    if "IV" in base:
        return "IV"
    if "III" in base:
        return "III"
    if "II" in base:
        return "II"
    if "I" in base:
        return "I"
    return np.nan


def encode_t(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"T(\d)", str(s))
    return int(m.group(1)) if m else np.nan


def encode_n(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r"N(\d)", str(s))
    return int(m.group(1)) if m else np.nan


def encode_m(s):
    if pd.isna(s):
        return np.nan
    s = str(s).upper()
    if s.startswith("M0"):
        return 0
    if s.startswith("M1"):
        return 1
    return np.nan


def build_features(cohort: pd.DataFrame, splits: dict) -> tuple[pd.DataFrame, dict]:
    df = cohort.copy()
    pid = "TCGA_Barcode"

    # Numeric
    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
    df["AGE_SQ"] = df["AGE"] ** 2

    # Stage encodings
    df["AJCC_STAGE"] = df["PATH_STAGE"].apply(collapse_stage)
    df["T_NUM"] = df["PATH_T_STAGE"].apply(encode_t)
    df["N_NUM"] = df["PATH_N_STAGE"].apply(encode_n)
    df["M_NUM"] = df["PATH_M_STAGE"].apply(encode_m)
    df["TNM_COMPOSITE"] = df[["T_NUM", "N_NUM", "M_NUM"]].sum(axis=1, min_count=1)

    # Missing indicators (informative missingness)
    for c in ["AJCC_STAGE", "T_NUM", "N_NUM", "M_NUM"]:
        df[f"{c}_MISSING"] = df[c].isna().astype(int)

    # Binary — handle multiple input encodings: True/False, "Yes"/"No", 1/0
    def _to_bin(x):
        s = str(x).strip().lower()
        if s in {"true", "yes", "y", "1", "positive", "present"}:
            return 1
        if s in {"false", "no", "n", "0", "negative", "absent", "not present"}:
            return 0
        return 0  # treat unknown/NaN as 0 (no prior malignancy/treatment)
    df["PRIOR_MALIGNANCY_BIN"] = df["PRIOR_MALIGNANCY"].apply(_to_bin)
    df["PRIOR_TREATMENT_BIN"] = df["PRIOR_TREATMENT"].apply(_to_bin)

    # One-hot
    cats = ["SEX", "RACE", "ETHNICITY", "AJCC_STAGE", "studyId"]
    for c in cats:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
    one_hot = pd.get_dummies(df[cats], prefix=cats, dummy_na=False)

    numeric_cols = ["AGE", "AGE_SQ", "T_NUM", "N_NUM", "M_NUM", "TNM_COMPOSITE",
                    "AJCC_STAGE_MISSING", "T_NUM_MISSING", "N_NUM_MISSING", "M_NUM_MISSING",
                    "PRIOR_MALIGNANCY_BIN", "PRIOR_TREATMENT_BIN"]

    feats = pd.concat([df[[pid] + numeric_cols].reset_index(drop=True),
                       one_hot.reset_index(drop=True)], axis=1)

    # Impute numerics with train-set median, then standardize
    train_ids = set(splits["train"])
    train_mask = feats[pid].isin(train_ids)
    medians = feats.loc[train_mask, numeric_cols].median()
    feats[numeric_cols] = feats[numeric_cols].fillna(medians)

    scaler = StandardScaler()
    scaler.fit(feats.loc[train_mask, numeric_cols].values)
    feats[numeric_cols] = scaler.transform(feats[numeric_cols].values)

    meta = {
        "n_features": feats.shape[1] - 1,
        "numeric_cols": numeric_cols,
        "one_hot_cols": list(one_hot.columns),
        "train_medians": medians.to_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return feats, meta


def main(cfg: dict) -> None:
    root = Path(cfg["project_root"])
    cohort = pd.read_csv(root / cfg["paths"]["cohort_csv"], low_memory=False)
    splits = json.loads((root / cfg["paths"]["splits_dir"] / "splits.json").read_text())

    feats, meta = build_features(cohort, splits)

    out_dir = root / cfg["paths"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_dir / "clinical.parquet", index=False)
    (out_dir / "clinical_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"clinical features: {feats.shape}  patients: {feats['TCGA_Barcode'].nunique()}")
    print(f"wrote {out_dir/'clinical.parquet'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    main(load_cfg(args.config))
