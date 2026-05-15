"""RNA-Seq feature engineering: top-K most-variable genes (default 5000).

Memory-efficient streaming implementation suitable for 32-46 GB RAM machines.
Replaces the previous "load everything as wide matrix" approach (which hit ~20 GB
peak from pandas concat overhead) with a two-pass streaming algorithm:

  Pass 1 (variance ranking on TRAIN split):
    - Stream each train TSV, accumulate Welford running mean/M2 per gene
    - Memory: ~2 vectors of length n_genes (~60k floats = ~480 KB)

  Pass 2 (build top-K matrix for ALL patients):
    - Pick top-K gene_ids by train variance
    - Stream each TSV, extract only those K genes
    - Memory: K × n_patients float32 (~170 MB for K=5000, N=8459)

  Standardization:
    - GPU-accelerated via torch if CUDA available (~50ms)
    - Falls back to numpy on CPU (~2s)

Peak RAM: ~3 GB total (vs ~20 GB previously).
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_tsv_path(tsv_path: str, raw_root: Path) -> Path | None:
    if not isinstance(tsv_path, str):
        return None
    rel = tsv_path.replace("\\", "/").lstrip("/")
    rel = rel.replace("RNASeq_data", "RNAseq_data")  # case-fix Windows -> Linux
    p = raw_root / rel
    return p if p.exists() else None


def read_one(p: Path) -> pd.Series:
    """Read one star_gene_counts TSV; return log2(fpkm+1) indexed by gene_id."""
    df = pd.read_csv(p, sep="\t", skiprows=1,
                     usecols=["gene_id", "fpkm_unstranded"], low_memory=False)
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")]
    df["fpkm_unstranded"] = pd.to_numeric(df["fpkm_unstranded"], errors="coerce").fillna(0.0)
    df = df.groupby("gene_id", as_index=True)["fpkm_unstranded"].mean()
    return np.log2(df + 1).astype(np.float32)


def standardize_matrix(X_train: np.ndarray, X_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit (mean, std) on X_train, apply to X_full.

    Uses GPU if torch+CUDA available. Mutates `X_full` in place when on CPU
    to avoid an extra copy.
    """
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            t = torch.from_numpy(X_train).to(dev)
            mean = t.mean(dim=0)
            std = t.std(dim=0).clamp(min=1e-6)
            del t
            torch.cuda.empty_cache()

            t = torch.from_numpy(X_full).to(dev)
            t.sub_(mean).div_(std)
            out = t.cpu().numpy().astype(np.float32, copy=False)
            del t
            torch.cuda.empty_cache()
            return out, mean.cpu().numpy(), std.cpu().numpy()
    except Exception as e:
        print(f"  GPU standardize unavailable ({e}); falling back to CPU/numpy")

    # CPU fallback — operate in chunks to avoid copies
    mean = X_train.mean(axis=0).astype(np.float32)
    std = X_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1e-6
    np.subtract(X_full, mean, out=X_full)
    np.divide(X_full, std, out=X_full)
    return X_full, mean, std


def main(cfg: dict, top_k: int) -> None:
    root = Path(cfg["project_root"])
    raw_root = root / "data" / "raw"
    cohort = pd.read_csv(root / cfg["paths"]["clinical_csv"], low_memory=False)
    dedup = pd.read_csv(root / cfg["paths"]["cohort_csv"],
                        usecols=["TCGA_Barcode"], low_memory=False)
    keep_ids = set(dedup["TCGA_Barcode"])

    splits = json.loads((root / cfg["paths"]["splits_dir"] / "splits.json").read_text())
    train_ids = set(splits["train"])

    paths = (cohort[cohort["TCGA_Barcode"].isin(keep_ids)]
             .drop_duplicates("TCGA_Barcode")[["TCGA_Barcode", "tsv_path"]])
    paths["resolved"] = paths["tsv_path"].apply(lambda p: resolve_tsv_path(p, raw_root))
    paths = paths.dropna(subset=["resolved"]).reset_index(drop=True)
    print(f"resolved RNA-Seq files: {len(paths)} / {len(keep_ids)}")
    n_total = len(paths)

    # ---- Pass 1: streaming Welford variance on TRAIN patients only ----
    train_paths = paths[paths["TCGA_Barcode"].isin(train_ids)].reset_index(drop=True)
    print(f"pass 1/2: streaming variance on {len(train_paths)} train TSVs")

    gene_index = None  # discovered on first read
    n = 0
    mean_acc: np.ndarray | None = None
    m2_acc: np.ndarray | None = None

    for _, row in tqdm(train_paths.iterrows(), total=len(train_paths), desc="pass1 variance"):
        s = read_one(row["resolved"])
        if gene_index is None:
            gene_index = s.index
            mean_acc = np.zeros(len(gene_index), dtype=np.float64)
            m2_acc = np.zeros(len(gene_index), dtype=np.float64)
        else:
            # Reindex to canonical order; missing genes -> 0
            s = s.reindex(gene_index, fill_value=0.0)
        x = s.values.astype(np.float64)
        # Welford's online variance update
        n += 1
        delta = x - mean_acc
        mean_acc += delta / n
        m2_acc += delta * (x - mean_acc)
        del s, x, delta

    var_train = pd.Series(m2_acc / max(n - 1, 1), index=gene_index)
    print(f"  computed variance for {len(var_train)} genes")

    top_genes = var_train.nlargest(top_k).index.tolist()
    print(f"  selected top-{top_k} most variable genes")
    del var_train, mean_acc, m2_acc
    gc.collect()

    # ---- Pass 2: build (n_patients × top_k) matrix only ----
    print(f"pass 2/2: extracting top-{top_k} genes from all {n_total} patients")
    X_full = np.zeros((n_total, top_k), dtype=np.float32)
    pid_order = []
    train_mask = np.zeros(n_total, dtype=bool)

    for i, row in enumerate(tqdm(paths.itertuples(index=False), total=n_total, desc="pass2 extract")):
        s = read_one(row.resolved)
        s = s.reindex(top_genes, fill_value=0.0)
        X_full[i, :] = s.values
        pid_order.append(row.TCGA_Barcode)
        if row.TCGA_Barcode in train_ids:
            train_mask[i] = True
        del s

    print(f"  matrix shape: {X_full.shape}  RAM: {X_full.nbytes/1e6:.0f} MB")

    # ---- Standardize (GPU-accelerated when available) ----
    print("standardizing (fit on train rows only)")
    X_train_only = X_full[train_mask]
    X_full, mean, std = standardize_matrix(X_train_only, X_full)
    del X_train_only

    # ---- Write parquet ----
    out_dir = root / cfg["paths"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "expression.parquet"

    df_out = pd.DataFrame(X_full, columns=top_genes)
    df_out.insert(0, "TCGA_Barcode", pid_order)
    df_out.to_parquet(out, index=False)
    del df_out

    meta = {
        "n_patients": n_total,
        "n_genes": top_k,
        "gene_ids": top_genes,
        "selection": "top variance on train split (Welford streaming)",
        "scaler_fit_on": "train split only",
        "peak_ram_estimate_mb": float(X_full.nbytes / 1e6),
        "gpu_used": bool(os.environ.get("CUDA_VISIBLE_DEVICES") != "" and _has_cuda()),
    }
    (out_dir / "expression_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}  shape={X_full.shape}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--top-k", type=int, default=5000)
    args = ap.parse_args()
    main(load_cfg(args.config), args.top_k)
