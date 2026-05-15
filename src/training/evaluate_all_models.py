"""
Comprehensive Evaluation & Comparison of All Fine-Tuning Strategies
====================================================================
Compares all 6 fine-tuned models (ClinicalBERT × 3 strategies + Llama × 3 strategies)
using Concordance Index (C-index), risk stratification, and Kaplan-Meier analysis.

Run this AFTER training all models. It reads the embedding CSVs from data/processed/.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# CONFIGURATION
# =====================================================================
DATA_DIR = 'data/processed'
OUTPUT_DIR = 'data/processed/evaluation'
DATA_PATH = 'data/processed/merged_tcga_data_text_dedup.csv'

# Map: strategy_name -> (embedding_csv, risk_score_col, emb_prefix)
MODELS = {
    'ClinicalBERT Baseline': {
        'csv': 'finetuned_text_embeddings.csv',
        'emb_prefix': 'text_emb_',
        'family': 'ClinicalBERT',
    },
    'ClinicalBERT Conditioned': {
        'csv': 'finetuned_text_conditioned_embeddings.csv',
        'emb_prefix': 'text_emb_',
        'family': 'ClinicalBERT',
    },
    'ClinicalBERT Hierarchical': {
        'csv': 'finetuned_text_hierarchical_embeddings.csv',
        'emb_prefix': 'text_emb_',
        'family': 'ClinicalBERT',
    },
    'Llama-3 Baseline': {
        'csv': 'finetuned_llama_embeddings.csv',
        'emb_prefix': 'llama_emb_',
        'family': 'OpenBioLLM-8B',
    },
    'Llama-3 Conditioned': {
        'csv': 'finetuned_llama_conditioned_embeddings.csv',
        'emb_prefix': 'llama_emb_',
        'family': 'OpenBioLLM-8B',
    },
    'Llama-3 Hierarchical': {
        'csv': 'finetuned_llama_hierarchical_embeddings.csv',
        'emb_prefix': 'llama_emb_',
        'family': 'OpenBioLLM-8B',
    },
}


def load_survival_data():
    """Load the original survival data (OS_MONTHS, OS_STATUS, DISEASE_TYPE)."""
    df = pd.read_csv(DATA_PATH, low_memory=False,
                     usecols=['TCGA_Barcode', 'OS_MONTHS', 'OS_STATUS', 'DISEASE_TYPE'])
    df = df.dropna(subset=['OS_MONTHS', 'OS_STATUS']).reset_index(drop=True)
    df['event'] = df['OS_STATUS'].astype(str).str.contains('DECEASED').astype(int)
    df['duration'] = df['OS_MONTHS'].astype(float)
    return df


def compute_cindex(risk_scores, durations, events):
    """Compute Harrell's Concordance Index."""
    try:
        # Higher risk score = higher hazard, so negate for concordance_index
        # (lifelines expects: higher value = longer survival)
        ci = concordance_index(durations, -np.array(risk_scores), events)
        return ci
    except Exception:
        return np.nan


def evaluate_model(model_name, model_cfg, survival_df):
    """Evaluate a single model: compute C-index overall and per cancer type."""
    csv_path = os.path.join(DATA_DIR, model_cfg['csv'])
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {model_name}: {csv_path} not found")
        return None

    emb_df = pd.read_csv(csv_path)
    
    # Merge with survival data
    if 'cancer_type' not in emb_df.columns:
        # Baseline models don't have cancer_type in their CSVs
        merged = emb_df.merge(survival_df[['TCGA_Barcode', 'duration', 'event', 'DISEASE_TYPE']],
                              on='TCGA_Barcode', how='inner')
        merged.rename(columns={'DISEASE_TYPE': 'cancer_type'}, inplace=True)
    else:
        merged = emb_df.merge(survival_df[['TCGA_Barcode', 'duration', 'event']],
                              on='TCGA_Barcode', how='inner')

    if 'risk_score' not in merged.columns:
        print(f"  [SKIP] {model_name}: no risk_score column")
        return None

    # Overall C-index
    overall_ci = compute_cindex(merged['risk_score'], merged['duration'], merged['event'])

    # Per cancer type C-index
    per_type = {}
    for ct in merged['cancer_type'].unique():
        ct_df = merged[merged['cancer_type'] == ct]
        if ct_df['event'].sum() >= 5 and len(ct_df) >= 20:  # Need enough events
            ci = compute_cindex(ct_df['risk_score'], ct_df['duration'], ct_df['event'])
            per_type[ct] = {'c_index': ci, 'n_samples': len(ct_df),
                            'n_events': int(ct_df['event'].sum())}

    return {
        'model_name': model_name,
        'family': model_cfg['family'],
        'overall_c_index': overall_ci,
        'n_samples': len(merged),
        'n_events': int(merged['event'].sum()),
        'per_type': per_type,
        'merged_df': merged,
        'emb_prefix': model_cfg['emb_prefix'],
    }


def plot_cindex_comparison(results, output_dir):
    """Bar chart comparing overall C-index across all models."""
    names = [r['model_name'] for r in results]
    cindexes = [r['overall_c_index'] for r in results]
    families = [r['family'] for r in results]

    colors = {'ClinicalBERT': ['#2196F3', '#1565C0', '#0D47A1'],
              'OpenBioLLM-8B': ['#FF9800', '#E65100', '#BF360C']}
    
    family_counts = {}
    bar_colors = []
    for f in families:
        idx = family_counts.get(f, 0)
        bar_colors.append(colors.get(f, ['#999'])[min(idx, 2)])
        family_counts[f] = idx + 1

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(range(len(names)), cindexes, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (C-index=0.5)')

    for bar, ci in zip(bars, cindexes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{ci:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Concordance Index (C-index)', fontsize=12)
    ax.set_title('Model Comparison: Overall Concordance Index', fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, max(cindexes) + 0.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'overall_cindex_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_cancer_heatmap(results, output_dir):
    """Heatmap of C-index per cancer type × model."""
    all_types = set()
    for r in results:
        all_types.update(r['per_type'].keys())
    all_types = sorted(all_types)

    if not all_types:
        print("  [SKIP] Not enough data for per-cancer heatmap")
        return

    matrix = []
    model_names = []
    for r in results:
        row = [r['per_type'].get(ct, {}).get('c_index', np.nan) for ct in all_types]
        matrix.append(row)
        model_names.append(r['model_name'])

    matrix = np.array(matrix)
    
    # Shorten cancer type names for display
    short_types = [t[:30] for t in all_types]

    fig, ax = plt.subplots(figsize=(max(16, len(all_types) * 1.2), max(6, len(model_names) * 1.2)))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=0.8)
    
    ax.set_xticks(range(len(short_types)))
    ax.set_xticklabels(short_types, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(len(all_types)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='C-index')
    ax.set_title('C-index by Cancer Type × Model', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(output_dir, 'per_cancer_cindex_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_kaplan_meier(results, output_dir):
    """Kaplan-Meier curves for High vs Low risk groups for each model."""
    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    kmf = KaplanMeierFitter()

    for i, r in enumerate(results):
        ax = axes[i]
        df = r['merged_df']

        # Split into High/Low risk at median
        median_risk = df['risk_score'].median()
        high_risk = df[df['risk_score'] >= median_risk]
        low_risk = df[df['risk_score'] < median_risk]

        kmf.fit(low_risk['duration'], low_risk['event'], label='Low Risk')
        kmf.plot_survival_function(ax=ax, ci_show=True, color='#2196F3')

        kmf.fit(high_risk['duration'], high_risk['event'], label='High Risk')
        kmf.plot_survival_function(ax=ax, ci_show=True, color='#F44336')

        ax.set_title(f"{r['model_name']}\nC-index={r['overall_c_index']:.4f}", fontsize=11)
        ax.set_xlabel('Time (Months)')
        ax.set_ylabel('Survival Probability')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Kaplan-Meier Survival Curves: High vs Low Risk Groups', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, 'kaplan_meier_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_risk_distributions(results, output_dir):
    """Violin/distribution plot of risk scores by event status for each model."""
    n_models = len(results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        df = r['merged_df']

        alive = df[df['event'] == 0]['risk_score']
        deceased = df[df['event'] == 1]['risk_score']

        ax.hist(alive, bins=50, alpha=0.6, label='Alive', color='#4CAF50', density=True)
        ax.hist(deceased, bins=50, alpha=0.6, label='Deceased', color='#F44336', density=True)
        ax.set_title(f"{r['model_name']}", fontsize=11)
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Risk Score Distributions: Alive vs Deceased', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, 'risk_distributions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_embedding_tsne(results, output_dir, max_samples=3000):
    """t-SNE visualization of embeddings colored by cancer type for each model."""
    for r in results:
        df = r['merged_df']
        prefix = r['emb_prefix']
        emb_cols = [c for c in df.columns if c.startswith(prefix)]

        if not emb_cols or len(df) < 50:
            continue

        # Subsample for speed
        if len(df) > max_samples:
            sample_df = df.sample(max_samples, random_state=42)
        else:
            sample_df = df

        X = sample_df[emb_cols].values
        # PCA to 50 dims first for speed, then t-SNE to 2
        pca = PCA(n_components=min(50, X.shape[1]), random_state=42)
        X_pca = pca.fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        X_2d = tsne.fit_transform(X_pca)

        # Color by top cancer types
        top_types = sample_df['cancer_type'].value_counts().head(10).index.tolist()
        colors_map = plt.cm.tab10(np.linspace(0, 1, 10))

        fig, ax = plt.subplots(figsize=(12, 10))
        for idx, ct in enumerate(top_types):
            mask = sample_df['cancer_type'] == ct
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, alpha=0.5,
                      color=colors_map[idx], label=ct[:25])

        other_mask = ~sample_df['cancer_type'].isin(top_types)
        if other_mask.sum() > 0:
            ax.scatter(X_2d[other_mask, 0], X_2d[other_mask, 1], s=5, alpha=0.2,
                      color='gray', label='Other')

        ax.set_title(f"t-SNE: {r['model_name']} Embeddings", fontsize=14)
        ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.2)

        safe_name = r['model_name'].replace(' ', '_').replace('-', '_').lower()
        path = os.path.join(output_dir, f'tsne_{safe_name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


def generate_comparison_matrix(results, output_dir):
    """Generate and save a comprehensive comparison CSV."""
    rows = []
    for r in results:
        row = {
            'Model': r['model_name'],
            'Architecture': r['family'],
            'Overall_C_Index': r['overall_c_index'],
            'N_Samples': r['n_samples'],
            'N_Events': r['n_events'],
        }
        # Add per-type C-index
        for ct, info in r['per_type'].items():
            safe_ct = ct[:30].replace(' ', '_')
            row[f'CI_{safe_ct}'] = info['c_index']
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'full_comparison_matrix.csv')
    comp_df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Also print a nice summary table
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON TABLE")
    print(f"{'='*80}")
    summary = comp_df[['Model', 'Architecture', 'Overall_C_Index', 'N_Samples', 'N_Events']].copy()
    summary = summary.sort_values('Overall_C_Index', ascending=False)
    print(summary.to_string(index=False))
    print(f"{'='*80}")

    return comp_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading survival data...")
    survival_df = load_survival_data()
    print(f"  {len(survival_df)} patients loaded")

    # Evaluate all available models
    print("\n--- Evaluating Models ---")
    results = []
    for name, cfg in MODELS.items():
        print(f"\nEvaluating: {name}")
        result = evaluate_model(name, cfg, survival_df)
        if result is not None:
            results.append(result)
            print(f"  Overall C-index: {result['overall_c_index']:.4f}")
            print(f"  Samples: {result['n_samples']}, Events: {result['n_events']}")
        else:
            print(f"  [SKIPPED]")

    if not results:
        print("\nNo models found! Run the training scripts first.")
        print("Expected CSV files in data/processed/:")
        for name, cfg in MODELS.items():
            print(f"  - {cfg['csv']}")
        sys.exit(1)

    # Generate all plots and comparison metrics
    print(f"\n--- Generating Evaluation Plots ({len(results)} models) ---")

    print("\n1. Overall C-index Comparison:")
    plot_cindex_comparison(results, OUTPUT_DIR)

    print("\n2. Per-Cancer-Type C-index Heatmap:")
    plot_per_cancer_heatmap(results, OUTPUT_DIR)

    print("\n3. Kaplan-Meier Survival Curves:")
    plot_kaplan_meier(results, OUTPUT_DIR)

    print("\n4. Risk Score Distributions:")
    plot_risk_distributions(results, OUTPUT_DIR)

    print("\n5. t-SNE Embedding Visualizations:")
    plot_embedding_tsne(results, OUTPUT_DIR)

    print("\n6. Comparison Matrix CSV:")
    comp_df = generate_comparison_matrix(results, OUTPUT_DIR)

    # Final summary
    best = max(results, key=lambda x: x['overall_c_index'])
    worst = min(results, key=lambda x: x['overall_c_index'])

    print(f"\n{'='*80}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Models evaluated:    {len(results)}")
    print(f"  Best model:          {best['model_name']} (C-index={best['overall_c_index']:.4f})")
    print(f"  Worst model:         {worst['model_name']} (C-index={worst['overall_c_index']:.4f})")
    print(f"  Improvement:         {best['overall_c_index'] - worst['overall_c_index']:.4f}")
    print(f"")
    print(f"  All outputs in:      {OUTPUT_DIR}/")
    print(f"  Key files:")
    print(f"    - overall_cindex_comparison.png")
    print(f"    - per_cancer_cindex_heatmap.png")
    print(f"    - kaplan_meier_comparison.png")
    print(f"    - risk_distributions.png")
    print(f"    - tsne_*.png")
    print(f"    - full_comparison_matrix.csv")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
