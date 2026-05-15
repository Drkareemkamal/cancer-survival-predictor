"""Comprehensive evaluation of all trained survival models on the held-out test set.

Loads:
  - Cox PH       (models/baselines/cox.pkl)
  - RSF          (models/baselines/rsf.pkl)
  - Autoencoder  (models/multimodal/autoencoder/best.pt)
  - Transformer  (models/multimodal/transformer/best.pt)
  - Ensemble     (models/multimodal/ensemble/best.pt)

Produces (in figures/):
  Seaborn static plots
    - cindex_comparison.png        bar chart of val/test C-index per model
    - km_by_risk_<model>.png       KM curves stratified by risk tertile
    - time_dependent_auc.png       AUC at 6/12/24/36/60 months
    - risk_score_distribution.png  density plot per event status
    - ablation_table.png           full multimodal ablation table

  Plotly interactive (HTML)
    - dashboard.html               full interactive dashboard with 6 panels
    - km_interactive.html          KM curves with hover tooltips
    - cindex_interactive.html      ablation bar chart

  evaluation_summary.json          all numbers + bootstrapped 95 % CIs
  evaluation_report.md             human-readable summary

Run:
    python -m src.evaluation.evaluate_all
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from plotly.subplots import make_subplots
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

from src.models.autoencoder import MissingAwareMultimodalAutoencoder
from src.models.data_loaders import load_all, split_indices
from src.models.ensemble import AdaptiveEnsembleSurvival
from src.models.transformer import RobustTransformerSurvival
from src.models.train_multimodal import _features_for, _struct_to_numeric

sns.set_theme(style="whitegrid", context="talk", palette="viridis")

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMES_OF_INTEREST = np.array([6, 12, 24, 36, 60], dtype=float)


# ---------------------------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------------------------
def load_aligned_test_data() -> tuple[dict, np.ndarray, dict, dict]:
    """Return (test_features_per_modality, availability, surv_test, dims)."""
    data = load_all()
    surv = data.survival.copy()
    surv["time"] = surv["time"].clip(lower=0.1)

    modalities = ["clinical", "expression", "mutation", "pathology_struct", "pathology_embed"]
    mods, avails, dims = {}, [], {}
    for m in modalities:
        if m == "pathology_struct" and data.pathology_struct is not None:
            data.pathology_struct = _struct_to_numeric(data.pathology_struct)
        X, mask = _features_for(data, m, surv)
        if X is None:
            continue
        mods[m] = X
        avails.append(mask)
        dims[m] = X.shape[1]
    avail = np.stack(avails, axis=1).astype(np.float32)

    idx = split_indices(data, surv)
    test_idx = idx["test"]

    test_mods = {m: arr[test_idx] for m, arr in mods.items()}
    test_avail = avail[test_idx]
    test_surv = surv.iloc[test_idx].reset_index(drop=True)

    # Train surv is needed by sksurv time-dependent AUC for the censoring distribution
    train_idx = idx["train"]
    train_surv = surv.iloc[train_idx].reset_index(drop=True)

    return {
        "test_mods": test_mods,
        "test_avail": test_avail,
        "test_surv": test_surv,
        "train_surv": train_surv,
        "dims": dims,
        "modalities": list(mods.keys()),
        "train_X": {m: arr[train_idx] for m, arr in mods.items()},
    }


# ---------------------------------------------------------------------------
# 2. Load each trained model + predict risk scores on test set
# ---------------------------------------------------------------------------
def predict_cox(data: dict) -> np.ndarray | None:
    p = ROOT / "models/baselines/cox.pkl"
    if not p.exists():
        return None
    bundle = pickle.load(p.open("rb"))
    cph, feat_cols = bundle["model"], bundle["feat_cols"]
    # Cox baseline was trained on the modality subset declared in the pkl
    X = pd.concat([pd.DataFrame(data["test_mods"][m]) for m in bundle["modalities"]
                   if m in data["test_mods"]], axis=1)
    # Try to match column count (Cox baseline subset is usually clinical only)
    if X.shape[1] < len(feat_cols):
        # Pad missing with zeros (means standardized features at the mean)
        pad = np.zeros((len(X), len(feat_cols) - X.shape[1]))
        X = pd.concat([X.reset_index(drop=True), pd.DataFrame(pad)], axis=1)
    elif X.shape[1] > len(feat_cols):
        X = X.iloc[:, : len(feat_cols)]
    X.columns = feat_cols
    return cph.predict_partial_hazard(X).values


def predict_rsf(data: dict) -> np.ndarray | None:
    p = ROOT / "models/baselines/rsf.pkl"
    if not p.exists():
        return None
    bundle = pickle.load(p.open("rb"))
    rsf, feat_cols = bundle["model"], bundle["feat_cols"]
    X = pd.concat([pd.DataFrame(data["test_mods"][m]) for m in bundle["modalities"]
                   if m in data["test_mods"]], axis=1)
    if X.shape[1] < len(feat_cols):
        pad = np.zeros((len(X), len(feat_cols) - X.shape[1]))
        X = pd.concat([X.reset_index(drop=True), pd.DataFrame(pad)], axis=1)
    elif X.shape[1] > len(feat_cols):
        X = X.iloc[:, : len(feat_cols)]
    return rsf.predict(X.values)


def _build_deep(kind: str, dims: dict):
    if kind == "autoencoder":
        return MissingAwareMultimodalAutoencoder(dims, latent_dim=256, n_heads=8, dropout=0.1)
    if kind == "transformer":
        return RobustTransformerSurvival(dims, d_model=512, n_heads=8, n_layers=4, dim_ff=2048, dropout=0.1)
    if kind == "ensemble":
        ae = MissingAwareMultimodalAutoencoder(dims, latent_dim=256, n_heads=8, dropout=0.1)
        tr = RobustTransformerSurvival(dims, d_model=512, n_heads=8, n_layers=4, dim_ff=2048, dropout=0.1)
        return AdaptiveEnsembleSurvival(ae, tr)
    raise ValueError(kind)


def predict_deep(kind: str, data: dict) -> np.ndarray | None:
    p = ROOT / f"models/multimodal/{kind}/best.pt"
    if not p.exists():
        return None
    net = _build_deep(kind, data["dims"]).to(DEVICE)
    state = torch.load(p, map_location=DEVICE)
    net.load_state_dict(state)
    net.eval()

    inputs = {m: torch.from_numpy(arr).to(DEVICE) for m, arr in data["test_mods"].items()}
    avail = torch.from_numpy(data["test_avail"]).to(DEVICE)
    with torch.no_grad():
        risk = net(inputs, avail).cpu().numpy()
    return risk


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def bootstrap_cindex(risk, time, event, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(risk)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.integers(0, n, size=n)
        vals[i] = concordance_index(time[b], -risk[b], event[b])
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def time_dependent_auc(train_surv, test_surv, risk, times):
    y_train = Surv.from_arrays(train_surv["event"].astype(bool).values,
                                train_surv["time"].clip(lower=0.1).values)
    y_test = Surv.from_arrays(test_surv["event"].astype(bool).values,
                               test_surv["time"].clip(lower=0.1).values)
    # Restrict to times within test follow-up
    t_max = test_surv["time"].max()
    valid_times = times[times < t_max - 0.5]
    if len(valid_times) == 0:
        return {}, np.nan
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times=valid_times)
    return dict(zip([float(t) for t in valid_times], [float(a) for a in auc])), float(mean_auc)


def stratify_risk(risk, n_groups=3):
    quantiles = np.quantile(risk, np.linspace(0, 1, n_groups + 1)[1:-1])
    return np.digitize(risk, quantiles)


# ---------------------------------------------------------------------------
# 4. Plotting — Seaborn (static)
# ---------------------------------------------------------------------------
def plot_cindex_bar(scores: dict[str, dict]) -> Path:
    rows = []
    for name, s in scores.items():
        rows.append({"Model": name, "Split": "Validation", "C-index": s["val_cindex"]})
        rows.append({"Model": name, "Split": "Test", "C-index": s["test_cindex"],
                     "CI_low": s.get("test_ci_low", np.nan),
                     "CI_high": s.get("test_ci_high", np.nan)})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(df, x="Model", y="C-index", hue="Split", ax=ax,
                palette={"Validation": "#5B7DB1", "Test": "#C25450"})
    ax.set_ylim(0.5, 0.9)
    ax.set_title("Concordance Index — All Models (5 modalities)")
    ax.axhline(0.5, ls="--", color="grey", alpha=0.5, label="Random")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGURES_DIR / "cindex_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_km_by_risk(risk, test_surv, model_name, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    groups = stratify_risk(risk, n_groups=3)
    labels = ["Low risk", "Medium risk", "High risk"]
    palette = sns.color_palette("viridis", n_colors=3)
    kmf = KaplanMeierFitter()
    for g, (label, color) in enumerate(zip(labels, palette)):
        sel = groups == g
        if sel.sum() == 0:
            continue
        kmf.fit(test_surv.loc[sel, "time"], test_surv.loc[sel, "event"],
                label=f"{label} (n={int(sel.sum())})")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    lr = multivariate_logrank_test(test_surv["time"], groups, test_surv["event"])
    ax.set_title(f"Kaplan–Meier — {model_name}\n"
                 f"log-rank p = {lr.p_value:.3e}, χ² = {lr.test_statistic:.1f}")
    ax.set_xlabel("Months from diagnosis")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"logrank_p": float(lr.p_value), "logrank_chi2": float(lr.test_statistic)}


def plot_time_dependent_auc(auc_data: dict[str, dict], out_path: Path):
    rows = []
    for model, auc_dict in auc_data.items():
        for t, a in auc_dict.items():
            rows.append({"Model": model, "Time (months)": t, "AUC": a})
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(df, x="Time (months)", y="AUC", hue="Model", marker="o", ax=ax, lw=2.5)
    ax.axhline(0.5, ls="--", color="grey", alpha=0.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Time-Dependent ROC AUC")
    ax.legend(loc="lower right", title=None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_risk_distribution(risks: dict[str, np.ndarray], test_surv, out_path: Path):
    rows = []
    for model, r in risks.items():
        if r is None:
            continue
        # Standardize risk for cross-model comparability
        z = (r - r.mean()) / (r.std() + 1e-9)
        for risk_z, e in zip(z, test_surv["event"]):
            rows.append({"Model": model, "Risk (z-scored)": risk_z,
                         "Outcome": "Event (death)" if e else "Censored"})
    df = pd.DataFrame(rows)

    g = sns.FacetGrid(df, col="Model", col_wrap=3, hue="Outcome",
                      height=3.5, aspect=1.3, palette={"Event (death)": "#C25450", "Censored": "#5B7DB1"})
    g.map_dataframe(sns.kdeplot, x="Risk (z-scored)", fill=True, alpha=0.55)
    g.set_titles("{col_name}")
    g.add_legend()
    g.figure.suptitle("Risk Score Distribution by Outcome", y=1.02)
    g.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def plot_ablation_table(ablation: dict, out_path: Path):
    """Build the modality-ablation panel from saved baseline results_*.json."""
    rows = []
    for key, vals in ablation.items():
        for est in ("cox", "rsf"):
            if vals.get(est):
                rows.append({"Modalities": key,
                             "Estimator": est.upper(),
                             "Test C-index": vals[est]["test_cindex"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(df, y="Modalities", x="Test C-index", hue="Estimator", ax=ax, palette="viridis")
    ax.axvline(0.5, ls="--", color="grey", alpha=0.5)
    ax.set_xlim(0.5, 0.9)
    ax.set_title("Modality Ablation — Test C-index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Plotting — Plotly (interactive)
# ---------------------------------------------------------------------------
def plotly_cindex_bar(scores: dict[str, dict]) -> Path:
    rows = []
    for name, s in scores.items():
        rows.append({"Model": name, "Split": "Validation", "C-index": s["val_cindex"]})
        rows.append({"Model": name, "Split": "Test", "C-index": s["test_cindex"]})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Model", y="C-index", color="Split", barmode="group",
                 color_discrete_map={"Validation": "#5B7DB1", "Test": "#C25450"},
                 title="Concordance Index — Validation vs Test",
                 hover_data={"C-index": ":.4f"})
    fig.update_layout(yaxis_range=[0.5, 0.9], template="plotly_white",
                      legend=dict(orientation="h", y=1.05))
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="Random", annotation_position="bottom right")
    out = FIGURES_DIR / "cindex_interactive.html"
    fig.write_html(out)
    return out


def plotly_km_curves(risks: dict, test_surv) -> Path:
    fig = go.Figure()
    palette = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]
    for (model, risk), color in zip(risks.items(), palette):
        if risk is None:
            continue
        groups = stratify_risk(risk, n_groups=3)
        for g, label in enumerate(["Low", "Medium", "High"]):
            sel = groups == g
            if sel.sum() == 0:
                continue
            kmf = KaplanMeierFitter().fit(test_surv.loc[sel, "time"],
                                          test_surv.loc[sel, "event"])
            x = kmf.survival_function_.index.values
            y = kmf.survival_function_.iloc[:, 0].values
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                name=f"{model} — {label} (n={int(sel.sum())})",
                line=dict(color=color, dash={"Low": "solid", "Medium": "dash", "High": "dot"}[label]),
                visible=(model == "Ensemble") if "Ensemble" in risks else (model == list(risks.keys())[0]),
            ))
    # Buttons to switch model
    models = [m for m, r in risks.items() if r is not None]
    buttons = []
    for i, m in enumerate(models):
        visible = []
        for mm in models:
            if mm == m:
                visible.extend([True] * 3)
            else:
                visible.extend([False] * 3)
        buttons.append(dict(label=m, method="update",
                            args=[{"visible": visible},
                                  {"title": f"Kaplan–Meier — {m} (risk tertiles, test set)"}]))
    fig.update_layout(
        title=f"Kaplan–Meier — {models[0] if models else ''} (risk tertiles, test set)",
        xaxis_title="Months from diagnosis",
        yaxis_title="Survival probability",
        template="plotly_white",
        hovermode="x unified",
        updatemenus=[dict(buttons=buttons, x=0.0, y=1.15, xanchor="left")],
    )
    out = FIGURES_DIR / "km_interactive.html"
    fig.write_html(out)
    return out


def plotly_dashboard(scores, risks, auc_data, test_surv) -> Path:
    """One HTML with 6 panels: bar, density, KM, time-AUC, scatter, table."""
    titles = ("Test C-index", "Risk distribution (Ensemble)",
              "KM curves — Ensemble", "Time-dependent AUC",
              "Risk vs survival time (Ensemble)", "Summary table")
    fig = make_subplots(
        rows=3, cols=2, subplot_titles=titles,
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "table"}]],
        vertical_spacing=0.10, horizontal_spacing=0.10,
    )

    # 1. Test C-index bar
    df_bar = pd.DataFrame([{"Model": n, "C": s["test_cindex"]} for n, s in scores.items()])
    fig.add_trace(go.Bar(x=df_bar["Model"], y=df_bar["C"], marker_color="#C25450",
                         text=df_bar["C"].round(4), textposition="outside",
                         name="Test C", showlegend=False), row=1, col=1)
    fig.update_yaxes(range=[0.5, 0.9], row=1, col=1)

    # 2. Risk distribution histogram per outcome (Ensemble)
    ens_risk = risks.get("Ensemble")
    if ens_risk is not None:
        z = (ens_risk - ens_risk.mean()) / (ens_risk.std() + 1e-9)
        for evt, color, name in [(0, "#5B7DB1", "Censored"), (1, "#C25450", "Event")]:
            mask = test_surv["event"] == evt
            fig.add_trace(go.Histogram(x=z[mask], name=name, marker_color=color,
                                       nbinsx=40, opacity=0.6), row=1, col=2)

    # 3. KM curves for Ensemble
    if ens_risk is not None:
        groups = stratify_risk(ens_risk, n_groups=3)
        for g, (label, color) in enumerate(zip(["Low", "Medium", "High"],
                                                ["#5ec962", "#3b528b", "#440154"])):
            sel = groups == g
            kmf = KaplanMeierFitter().fit(test_surv.loc[sel, "time"], test_surv.loc[sel, "event"])
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=kmf.survival_function_.iloc[:, 0],
                mode="lines", name=f"{label} (n={int(sel.sum())})",
                line=dict(color=color, width=3),
            ), row=2, col=1)

    # 4. Time-dependent AUC
    for model, auc_dict in auc_data.items():
        if not auc_dict:
            continue
        ts = sorted(auc_dict)
        fig.add_trace(go.Scatter(x=ts, y=[auc_dict[t] for t in ts], mode="lines+markers",
                                 name=model), row=2, col=2)

    # 5. Risk vs survival scatter (Ensemble)
    if ens_risk is not None:
        z = (ens_risk - ens_risk.mean()) / (ens_risk.std() + 1e-9)
        fig.add_trace(go.Scatter(
            x=z, y=test_surv["time"],
            mode="markers",
            marker=dict(color=test_surv["event"], colorscale=[[0, "#5B7DB1"], [1, "#C25450"]],
                        size=6, opacity=0.6),
            text=[f"event={int(e)}" for e in test_surv["event"]],
            name="patients", showlegend=False,
        ), row=3, col=1)

    # 6. Summary table
    table_rows = [["Model", "Val C", "Test C", "95 % CI"]]
    for n, s in scores.items():
        ci = f"[{s.get('test_ci_low', float('nan')):.3f}, {s.get('test_ci_high', float('nan')):.3f}]"
        table_rows.append([n,
                           f"{s['val_cindex']:.4f}",
                           f"{s['test_cindex']:.4f}", ci])
    cols = list(zip(*table_rows))
    fig.add_trace(go.Table(
        header=dict(values=list(cols[0]) if isinstance(cols[0], tuple) else cols[0]),
        cells=dict(values=[list(c) if isinstance(c, tuple) else c for c in zip(*table_rows[1:])]),
    ), row=3, col=2)

    fig.update_layout(height=1300, template="plotly_white",
                      title="Multimodal Survival — Interactive Dashboard (Test Set)",
                      showlegend=True)
    out = FIGURES_DIR / "dashboard.html"
    fig.write_html(out)
    return out


# ---------------------------------------------------------------------------
# 6. Pathology evaluation panel
# ---------------------------------------------------------------------------
def plot_pathology_tasks(out_path: Path) -> dict | None:
    p = ROOT / "models/PathQwen2.5/eval_pathology.json"
    if not p.exists():
        print("[skip] pathology eval not found; run src.evaluation.pathology_eval first")
        return None
    data = json.loads(p.read_text())
    rows = []
    for task, m in data.items():
        if isinstance(m, dict) and m.get("n", 0) > 0:
            rows.append({"Task": task,
                         "Accuracy": m.get("accuracy"),
                         "Macro F1": m.get("macro_f1"),
                         "N": m.get("n")})
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("Task")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df_long = df.reset_index().melt(id_vars=["Task", "N"], value_vars=["Accuracy", "Macro F1"],
                                     var_name="Metric", value_name="Score")
    sns.barplot(df_long, y="Task", x="Score", hue="Metric", ax=axes[0],
                palette={"Accuracy": "#5B7DB1", "Macro F1": "#C25450"})
    axes[0].set_xlim(0, 1.0)
    axes[0].set_title("PathQwen2.5 — Per-task scores")
    axes[0].axvline(0.5, ls="--", color="grey", alpha=0.5)

    # Paper-comparison panel (Saluja 2025 published numbers)
    paper = {"cancer_type": 0.96, "ajcc_stage": 0.85, "prognosis_good": 0.48}
    cmp_rows = []
    for task, paper_val in paper.items():
        if task not in df.index:
            continue
        ours = df.loc[task, "Accuracy" if task != "prognosis_good" else "Macro F1"]
        cmp_rows.append({"Task": task, "Source": "Saluja 2025", "Score": paper_val})
        cmp_rows.append({"Task": task, "Source": "PathQwen2.5 (ours)", "Score": ours})
    if cmp_rows:
        cmp = pd.DataFrame(cmp_rows)
        sns.barplot(cmp, x="Task", y="Score", hue="Source", ax=axes[1],
                    palette={"Saluja 2025": "#999999", "PathQwen2.5 (ours)": "#440154"})
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title("Head-to-head vs Saluja 2025")
        for tick in axes[1].get_xticklabels():
            tick.set_rotation(15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return data


# ---------------------------------------------------------------------------
# 7. Orchestration
# ---------------------------------------------------------------------------
def main():
    print("loading test data …")
    data = load_aligned_test_data()
    test_surv = data["test_surv"]
    train_surv = data["train_surv"]
    print(f"  test: {len(test_surv)} patients, "
          f"{int(test_surv['event'].sum())} events ({100*test_surv['event'].mean():.1f} %)")

    # --- Predict risk per model ---
    print("\npredicting test risk scores …")
    risks: dict[str, np.ndarray | None] = {}
    risks["Cox PH"]      = predict_cox(data)
    risks["RSF"]         = predict_rsf(data)
    risks["Autoencoder"] = predict_deep("autoencoder", data)
    risks["Transformer"] = predict_deep("transformer", data)
    risks["Ensemble"]    = predict_deep("ensemble", data)

    # --- Compute test C-index (+ bootstrap CI) per model ---
    print("\ncomputing C-index + bootstrap CIs …")
    scores = {}
    for name, risk in risks.items():
        if risk is None:
            print(f"  {name:14s}  (model missing — skipping)")
            continue
        c_mean, c_lo, c_hi = bootstrap_cindex(risk, test_surv["time"].values,
                                              test_surv["event"].values)
        # Read val C from the saved JSON (already computed during training)
        val_c = _val_cindex_from_disk(name)
        scores[name] = {"val_cindex": val_c, "test_cindex": c_mean,
                        "test_ci_low": c_lo, "test_ci_high": c_hi}
        print(f"  {name:14s}  test C = {c_mean:.4f}  95 % CI [{c_lo:.4f}, {c_hi:.4f}]")

    # --- Time-dependent AUC ---
    print("\ncomputing time-dependent AUC …")
    auc_data = {}
    for name, risk in risks.items():
        if risk is None:
            continue
        per_t, mean = time_dependent_auc(train_surv, test_surv, risk, TIMES_OF_INTEREST)
        auc_data[name] = per_t
        scores[name]["mean_time_auc"] = mean
        print(f"  {name:14s}  mean AUC = {mean:.4f}  per-t = "
              f"{ {int(k): round(v, 3) for k, v in per_t.items()} }")

    # --- Generate static seaborn plots ---
    print("\ngenerating seaborn plots …")
    plot_cindex_bar(scores)
    km_stats = {}
    for name, risk in risks.items():
        if risk is None:
            continue
        out = FIGURES_DIR / f"km_by_risk_{name.replace(' ', '_')}.png"
        stats = plot_km_by_risk(risk, test_surv, name, out)
        km_stats[name] = stats
        scores[name].update(stats)
    plot_time_dependent_auc(auc_data, FIGURES_DIR / "time_dependent_auc.png")
    plot_risk_distribution(risks, test_surv, FIGURES_DIR / "risk_score_distribution.png")

    # --- Ablation panel ---
    print("\nbuilding ablation table …")
    ablation = {}
    for json_p in (ROOT / "models/baselines").glob("results_*.json"):
        d = json.loads(json_p.read_text())
        key = " + ".join(d["modalities"])
        ablation[key] = d
    plot_ablation_table(ablation, FIGURES_DIR / "ablation_table.png")

    # --- Pathology task panel ---
    print("\ngenerating pathology-task figure …")
    path_data = plot_pathology_tasks(FIGURES_DIR / "pathology_tasks.png")

    # --- Plotly interactive ---
    print("\ngenerating interactive plotly dashboards …")
    plotly_cindex_bar(scores)
    plotly_km_curves(risks, test_surv)
    plotly_dashboard(scores, risks, auc_data, test_surv)

    # --- Save summary ---
    summary = {
        "scores": scores,
        "auc_per_time": auc_data,
        "ablation": {k: {"cox": v.get("cox", {}), "rsf": v.get("rsf", {})}
                     for k, v in ablation.items()},
        "test_n": int(len(test_surv)),
        "test_event_rate": float(test_surv["event"].mean()),
    }
    (FIGURES_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))

    write_markdown_report(summary, path_data)

    print(f"\n✓ all artifacts written to {FIGURES_DIR}/")


def _val_cindex_from_disk(name: str) -> float:
    """Pull val C from the saved training JSONs."""
    # Cox/RSF: try the all-modality run first, fall back to clinical-only
    baseline_jsons = [
        ROOT / "models/baselines/results_all.json",
        ROOT / "models/baselines/results_clin_pstruct_mut.json",
        ROOT / "models/baselines/results_clin_pstruct.json",
        ROOT / "models/baselines/results_clin_mut.json",
        ROOT / "models/baselines/results_clinical.json",
    ]
    if name == "Cox PH":
        for p in baseline_jsons:
            if not p.exists(): continue
            d = json.loads(p.read_text())
            cox = d.get("cox")
            if isinstance(cox, dict):
                return cox.get("val_cindex", float("nan"))
        return float("nan")
    if name == "RSF":
        for p in baseline_jsons:
            if not p.exists(): continue
            d = json.loads(p.read_text())
            rsf = d.get("rsf")
            if isinstance(rsf, dict):
                return rsf.get("val_cindex", float("nan"))
        return float("nan")
    deep_paths = {
        "Autoencoder": ROOT / "models/multimodal/autoencoder/results.json",
        "Transformer": ROOT / "models/multimodal/transformer/results.json",
        "Ensemble":    ROOT / "models/multimodal/ensemble/results.json",
    }
    p = deep_paths.get(name)
    if not p or not p.exists():
        return float("nan")
    return json.loads(p.read_text()).get("val_cindex", float("nan"))


def write_markdown_report(summary: dict, path_data: dict | None) -> None:
    md = ["# Evaluation Report — Test Set Performance", ""]
    md.append(f"- Test patients: **{summary['test_n']}**")
    md.append(f"- Event rate: **{100*summary['test_event_rate']:.1f}%**")
    md.append("")
    md.append("## Survival Models — Test C-index")
    md.append("")
    md.append("| Model | Val C | Test C | 95 % CI | Mean time-AUC | KM log-rank p |")
    md.append("|---|---|---|---|---|---|")
    for name, s in summary["scores"].items():
        md.append(f"| **{name}** | {s.get('val_cindex', float('nan')):.4f} "
                  f"| {s.get('test_cindex', float('nan')):.4f} "
                  f"| [{s.get('test_ci_low', float('nan')):.4f}, "
                  f"{s.get('test_ci_high', float('nan')):.4f}] "
                  f"| {s.get('mean_time_auc', float('nan')):.4f} "
                  f"| {s.get('logrank_p', float('nan')):.2e} |")
    md.append("")
    md.append("## Files generated")
    md.append("")
    for f in sorted(FIGURES_DIR.glob("*")):
        md.append(f"- `figures/{f.name}`")
    if path_data:
        md.append("")
        md.append("## PathQwen2.5 — Per-task accuracy")
        md.append("")
        md.append("| Task | N | Accuracy | Macro F1 |")
        md.append("|---|---|---|---|")
        for task, m in path_data.items():
            if isinstance(m, dict) and m.get("n", 0) > 0:
                md.append(f"| {task} | {m['n']} | {m.get('accuracy', float('nan')):.4f} "
                          f"| {m.get('macro_f1', float('nan')):.4f} |")
    (FIGURES_DIR / "evaluation_report.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
