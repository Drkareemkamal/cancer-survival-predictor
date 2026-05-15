"""Mutation feature engineering — binary (paper baseline) + impact-weighted (your contribution).

Per RESEARCH.pdf section 2.2:
  (a) Binary matrix: gene mutated (1) vs not (0) — paper's representation
  (c) Impact-weighted scalars per patient:
      - n_mutations_total
      - n_high_impact (Variant_Classification in HIGH set)
      - n_missense / n_nonsense / n_silent (granular)
  + variant-frequency filter: genes mutated in <1% of cohort dropped (paper's filter).

Variant impact tiers (GDC/MAF Variant_Classification standard):
  HIGH:     Frame_Shift_Del, Frame_Shift_Ins, Nonsense_Mutation, Splice_Site,
            Translation_Start_Site, Nonstop_Mutation, In_Frame_Del, In_Frame_Ins
  MED:      Missense_Mutation
  LOW:      Silent, RNA, Intron, IGR, 3'UTR, 5'UTR, 3'Flank, 5'Flank
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


HIGH_IMPACT = {
    "Frame_Shift_Del", "Frame_Shift_Ins", "Nonsense_Mutation",
    "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
    "In_Frame_Del", "In_Frame_Ins",
}
MISSENSE = {"Missense_Mutation"}


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_maf(p: Path) -> pd.DataFrame:
    """Read a GDC MAF .gz; return DataFrame[Hugo_Symbol, Variant_Classification]."""
    try:
        df = pd.read_csv(
            p, sep="\t", comment="#", low_memory=False,
            usecols=["Hugo_Symbol", "Variant_Classification"],
            compression="gzip",
        )
        return df
    except Exception as e:
        print(f"skip {p.name}: {e}")
        return pd.DataFrame(columns=["Hugo_Symbol", "Variant_Classification"])


def main(cfg: dict, freq_cutoff: float, max_genes: int) -> None:
    root = Path(cfg["project_root"])
    paths = pd.read_parquet(root / cfg["paths"]["features_dir"] / "mutation_paths.parquet")
    paths = paths[paths["exists"]].copy()
    print(f"loading {len(paths)} MAF files")

    pid_col = "TCGA_Barcode"
    rows = []  # list of (TCGA_Barcode, gene) pairs
    impact_rows = []  # per-patient summary

    for _, r in tqdm(paths.iterrows(), total=len(paths), desc="parsing MAFs"):
        df = parse_maf(Path(r["maf_path"]))
        if df.empty:
            impact_rows.append({pid_col: r[pid_col], "n_total": 0,
                                "n_high_impact": 0, "n_missense": 0,
                                "n_other": 0})
            continue

        df["Hugo_Symbol"] = df["Hugo_Symbol"].astype(str)
        df = df[df["Hugo_Symbol"].str.len() > 0]

        # Per-patient impact counts
        n_total = len(df)
        n_high = df["Variant_Classification"].isin(HIGH_IMPACT).sum()
        n_mis = df["Variant_Classification"].isin(MISSENSE).sum()
        impact_rows.append({pid_col: r[pid_col], "n_total": n_total,
                            "n_high_impact": int(n_high), "n_missense": int(n_mis),
                            "n_other": int(n_total - n_high - n_mis)})

        # Per-patient mutated genes (deduped)
        for g in df["Hugo_Symbol"].unique():
            rows.append((r[pid_col], g))

    long_df = pd.DataFrame(rows, columns=[pid_col, "gene"])
    print(f"unique gene-patient pairs: {len(long_df)}")

    # Frequency filter: keep genes mutated in >= freq_cutoff fraction of cohort
    n_patients = paths[pid_col].nunique()
    gene_freq = long_df.groupby("gene")[pid_col].nunique() / n_patients
    kept = gene_freq[gene_freq >= freq_cutoff].sort_values(ascending=False)
    if len(kept) > max_genes:
        kept = kept.head(max_genes)
    print(f"genes kept (freq>={freq_cutoff:.3f}, max={max_genes}): {len(kept)}")

    # Build wide binary matrix
    long_df = long_df[long_df["gene"].isin(kept.index)]
    long_df["v"] = 1
    binary = long_df.pivot_table(index=pid_col, columns="gene", values="v",
                                 aggfunc="max", fill_value=0).astype(np.int8)

    # Re-index to all 8459 patients (zero-fill missing)
    all_pids = paths[pid_col].tolist()
    binary = binary.reindex(all_pids, fill_value=0)
    binary.columns = [f"MUT_{c}" for c in binary.columns]
    binary = binary.reset_index()

    # Impact summary table (per patient scalars)
    impact_df = pd.DataFrame(impact_rows)
    # log1p the counts to tame heavy tails before downstream standardization
    for c in ["n_total", "n_high_impact", "n_missense", "n_other"]:
        impact_df[f"{c}_log1p"] = np.log1p(impact_df[c]).astype(np.float32)
    impact_df = impact_df[[pid_col,
                           "n_total_log1p", "n_high_impact_log1p",
                           "n_missense_log1p", "n_other_log1p"]]

    # Merge: binary + impact -> single feature table
    feats = binary.merge(impact_df, on=pid_col, how="left")

    out_dir = root / cfg["paths"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mutation.parquet"
    feats.to_parquet(out, index=False)

    meta = {
        "n_patients": len(feats),
        "n_binary_features": int(binary.shape[1] - 1),
        "n_impact_features": 4,
        "freq_cutoff": freq_cutoff,
        "max_genes": max_genes,
        "top_genes_freq": kept.head(50).to_dict(),
    }
    (out_dir / "mutation_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  shape={feats.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--freq-cutoff", type=float, default=0.01,
                    help="Drop genes mutated in <freq_cutoff fraction of cohort")
    ap.add_argument("--max-genes", type=int, default=1000)
    args = ap.parse_args()
    main(load_cfg(args.config), args.freq_cutoff, args.max_genes)
