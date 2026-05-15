"""Survival evaluation metrics: C-index, time-dependent AUC, integrated Brier score.

All functions take risk scores (higher = higher hazard), event times, and event
indicators. Bootstrap helper for 95% CIs included.
"""
from __future__ import annotations

import numpy as np
from lifelines.utils import concordance_index
from sksurv.util import Surv
from sksurv.metrics import (
    cumulative_dynamic_auc,
    integrated_brier_score,
    brier_score,
)


def cindex(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    return float(concordance_index(time, -risk, event))


def time_dependent_auc(risk_train, time_train, event_train,
                      risk_test, time_test, event_test,
                      times: list[float]) -> dict:
    """Time-dependent ROC-AUC at each requested follow-up time (months)."""
    y_train = Surv.from_arrays(event_train.astype(bool), time_train)
    y_test = Surv.from_arrays(event_test.astype(bool), time_test)
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_test, times=times)
    return {"per_time": dict(zip([float(t) for t in times], auc.tolist())),
            "mean": float(mean_auc)}


def ibs(risk_train, time_train, event_train,
        risk_test, time_test, event_test,
        times: np.ndarray) -> float:
    """Integrated Brier Score over `times`. Lower is better.

    sksurv expects survival probability predictions S(t|x), but we have a single
    risk score per patient. We approximate S using exp(-risk * t / max_t) — this
    is a Cox-style proportional hazards approximation, fine for relative IBS.
    """
    y_train = Surv.from_arrays(event_train.astype(bool), time_train)
    y_test = Surv.from_arrays(event_test.astype(bool), time_test)
    # Normalize risk to [0, 1] then build naive S(t) = exp(-risk * t / t_max)
    r = (risk_test - risk_test.min()) / (risk_test.ptp() + 1e-9)
    t_max = times.max()
    surv_probs = np.exp(-np.outer(r, times) / t_max)
    return float(integrated_brier_score(y_train, y_test, surv_probs, times))


def bootstrap_ci(stat_fn, n_boot: int = 1000, seed: int = 42, *args) -> tuple[float, float]:
    """Bootstrap a 95% CI for any statistic returning a float."""
    rng = np.random.default_rng(seed)
    n = len(args[0])
    vals = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.integers(0, n, size=n)
        vals[i] = stat_fn(*[a[b] for a in args])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
