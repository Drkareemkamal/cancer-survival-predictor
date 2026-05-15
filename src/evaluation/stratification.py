"""Risk stratification: KM curves, log-rank tests, hazard ratios.

Stratifies a held-out set into Low/Med/High risk by predicted-risk tertiles
and reports significance + clinical interpretation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test


def stratify(risk: np.ndarray, n_groups: int = 3) -> np.ndarray:
    quantiles = np.quantile(risk, np.linspace(0, 1, n_groups + 1)[1:-1])
    return np.digitize(risk, quantiles)  # 0..n_groups-1


def km_plot(time: np.ndarray, event: np.ndarray, group: np.ndarray,
            labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    for g, label in enumerate(labels):
        sel = group == g
        kmf.fit(time[sel], event[sel], label=f"{label} (n={sel.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stratification_report(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                          out_dir: Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    g = stratify(risk, n_groups=3)
    labels = ["Low", "Medium", "High"]

    # Multivariate log-rank
    lr = multivariate_logrank_test(time, g, event)

    # Cox HR with risk-group as ordinal (Low=0, Med=1, High=2)
    df = pd.DataFrame({"time": time, "event": event.astype(int), "g": g})
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    hr = float(np.exp(cph.params_["g"]))

    km_plot(time, event, g, labels, name, out_dir / f"{name}_km.png")

    return {
        "n_low": int((g == 0).sum()),
        "n_med": int((g == 1).sum()),
        "n_high": int((g == 2).sum()),
        "logrank_p": float(lr.p_value),
        "logrank_chi2": float(lr.test_statistic),
        "hr_high_vs_low_ordinal": hr,
    }
