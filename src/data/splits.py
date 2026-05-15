"""Locked stratified train/val/test splits over the harmonized cohort.

Stratifies by studyId × event-status so every split sees every cancer type
and a balanced event rate. Writes patient-id lists to data/processed/splits/
along with a manifest hashing the input cohort.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def file_hash(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def make_splits(cfg: dict) -> None:
    root = Path(cfg["project_root"])
    cohort = pd.read_csv(root / cfg["paths"]["cohort_csv"], low_memory=False)

    pid = cfg["cohort"]["patient_id_col"]
    study = cfg["cohort"]["study_col"]
    status = cfg["cohort"]["os_status_col"]

    df = cohort[[pid, study, status]].dropna(subset=[pid, study, status]).copy()
    df["event"] = df[status].astype(str).str.startswith("1").astype(int)
    df["stratum"] = df[study].astype(str) + "_" + df["event"].astype(str)

    s = cfg["splits"]
    holdout_frac = s["val_frac"] + s["test_frac"]

    # First split: need >=2 in holdout per stratum -> ceil(2 / holdout_frac)
    min_first = int(-(-2 // holdout_frac))  # 7 for 30% holdout
    counts = df["stratum"].value_counts()
    safe1 = counts[counts >= min_first].index
    rare = df[~df["stratum"].isin(safe1)]
    safe_df = df[df["stratum"].isin(safe1)]

    train, holdout = train_test_split(
        safe_df, test_size=holdout_frac,
        random_state=s["seed"], stratify=safe_df["stratum"],
    )

    # Second split: need >=2 in each side of the holdout per stratum
    val_share = s["val_frac"] / holdout_frac
    h_counts = holdout["stratum"].value_counts()
    safe2 = h_counts[h_counts >= 2].index
    h_safe = holdout[holdout["stratum"].isin(safe2)]
    h_solo = holdout[~holdout["stratum"].isin(safe2)]

    val, test = train_test_split(
        h_safe, test_size=1 - val_share,
        random_state=s["seed"], stratify=h_safe["stratum"],
    )
    # Solo holdout patients go to test (more conservative for evaluation diversity)
    test = pd.concat([test, h_solo], ignore_index=True)

    # Distribute rare-stratum patients deterministically into train (signal preservation)
    train = pd.concat([train, rare], ignore_index=True)

    out_dir = root / cfg["paths"]["splits_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train[pid].tolist(), "val": val[pid].tolist(), "test": test[pid].tolist()}
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))

    manifest = {
        "cohort_csv": str(cfg["paths"]["cohort_csv"]),
        "cohort_sha256": file_hash(root / cfg["paths"]["cohort_csv"]),
        "n_total": int(len(df) + len(rare)),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_rare_into_train": int(len(rare)),
        "seed": s["seed"],
        "stratify_by": s["stratify_by"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"train={manifest['n_train']}  val={manifest['n_val']}  test={manifest['n_test']}")
    print(f"rare-stratum patients routed to train: {manifest['n_rare_into_train']}")
    print(f"wrote {out_dir/'splits.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    make_splits(load_cfg(args.config))
