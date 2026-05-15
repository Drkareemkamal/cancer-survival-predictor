"""Shared weave initialization helper.

Each subprocess entry point calls `init_weave(stage_name)` once. The W&B
WANDB_API_KEY in .env authenticates weave; the project name comes from
WEAVE_PROJECT env (defaults to 'cancer-survival-predictor').
"""
import os

import weave
from dotenv import load_dotenv

load_dotenv()

_WEAVE_INITIALIZED = False


def init_weave(stage: str | None = None) -> None:
    """Initialize weave once per process. Safe to call multiple times."""
    global _WEAVE_INITIALIZED
    if _WEAVE_INITIALIZED:
        return
    project = os.getenv("WEAVE_PROJECT", "cancer-survival-predictor")
    try:
        weave.init(project)
        _WEAVE_INITIALIZED = True
        if stage:
            print(f"[weave] tracking '{stage}' under project '{project}'")
    except Exception as e:
        print(f"[weave] init failed: {e} — running without tracking")
