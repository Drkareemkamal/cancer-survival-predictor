"""Resolve MAF paths for every cohort patient.

Bug it fixes: cohort CSV's `mutation_entity_id` column does NOT match folder
names on disk (intersection = 0). The correct file_id lives in
data/interim/maf_paths_from_new_json2.csv keyed by file_name.

Output: data/processed/features/mutation_paths.parquet
        columns: TCGA_Barcode, file_id, file_name, maf_path, exists, n_mutations
"""
import argparse
import gzip
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def count_mutations(p: Path) -> int:
    try:
        with gzip.open(p, "rt") as f:
            n = 0
            seen_header = False
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                if not seen_header:
                    seen_header = True
                    continue
                n += 1
        return n
    except Exception:
        return -1


def main(cfg: dict, validate: bool) -> None:
    root = Path(cfg["project_root"])
    cohort = pd.read_csv(root / cfg["paths"]["cohort_csv"], low_memory=False)
    maf_meta = pd.read_csv(root / cfg["paths"]["interim_maf_meta"])

    pid = cfg["cohort"]["patient_id_col"]

    df = cohort[[pid, "mutation_file_name"]].merge(
        maf_meta[["file_id", "file_name"]],
        left_on="mutation_file_name", right_on="file_name", how="left",
    )

    base = root / cfg["paths"]["raw_mutation_dir"]
    df["maf_path"] = df.apply(
        lambda r: str(base / r["file_id"] / r["mutation_file_name"])
        if pd.notna(r["file_id"]) else None,
        axis=1,
    )
    df["exists"] = df["maf_path"].apply(lambda p: bool(p) and Path(p).exists())

    print(f"cohort patients: {len(df)}")
    print(f"matched to maf_meta: {df['file_id'].notna().sum()}")
    print(f"MAF on disk: {df['exists'].sum()}")

    if validate:
        df["n_mutations"] = -1
        ok = df[df["exists"]].copy()
        for i, p in enumerate(tqdm(ok["maf_path"], desc="counting mutations")):
            df.loc[df["maf_path"] == p, "n_mutations"] = count_mutations(Path(p))
    else:
        df["n_mutations"] = -1  # populated lazily by features/mutation.py

    out_dir = root / cfg["paths"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mutation_paths.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--validate", action="store_true",
                    help="Read every MAF to count mutations (slow, ~5 min)")
    args = ap.parse_args()
    main(load_cfg(args.config), args.validate)
