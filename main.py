"""Cancer-survival pipeline orchestrator.

Stages run via subprocess so each step is observable. Stage outputs are
checked before re-running unless --force is passed.

Examples:
  python main.py --stage data
  python main.py --stage features
  python main.py --stage pathology-qa
  python main.py --stage pathology-train
  python main.py --stage survival-baselines
  python main.py --stage survival-deep
  python main.py --stage all
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import weave
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

# Weave project name — override via WEAVE_PROJECT env or .env
WEAVE_PROJECT = os.getenv("WEAVE_PROJECT", "cancer-survival-predictor")

STAGES = {
    "splits":              ["python", "-m", "src.data.splits",                "--config", "configs/data.yaml"],
    "mutation-paths":      ["python", "-m", "src.data.ingest_mutation",       "--config", "configs/data.yaml"],
    "clinical":            ["python", "-m", "src.features.clinical",          "--config", "configs/data.yaml"],
    "expression":          ["python", "-m", "src.features.expression",        "--config", "configs/data.yaml", "--top-k", "5000"],
    "mutation":            ["python", "-m", "src.features.mutation",          "--config", "configs/data.yaml"],
    "pathology-qa":        ["python", "-m", "src.training.build_multitask_qa","--config", "configs/data.yaml"],
    "pathology-cot":       ["python", "-m", "src.training.distill_cot",       "--config", "configs/pathology_llm.yaml"],
    "pathology-train":     ["python", "-m", "src.training.unsloth_finetune",  "--config", "configs/pathology_llm.yaml"],
    "survival-baselines":  ["python", "-m", "src.models.baselines",           "--modalities", "clinical"],
    "survival-deep":       ["python", "-m", "src.models.train_multimodal",    "--config", "configs/multimodal.yaml", "--model", "autoencoder"],
}

OUTPUT_PROBES = {
    "splits":             "data/processed/splits/splits.json",
    "mutation-paths":     "data/processed/features/mutation_paths.parquet",
    "clinical":           "data/processed/features/clinical.parquet",
    "expression":         "data/processed/features/expression.parquet",
    "mutation":           "data/processed/features/mutation.parquet",
    "pathology-qa":       "data/processed/pathology/qa_train.jsonl",
    "pathology-cot":      "data/processed/pathology/qa_train_cot.jsonl",
    "pathology-train":    "models/PathQwen2.5/final/adapter_config.json",
}

GROUPS = {
    "data":     ["splits", "mutation-paths"],
    "features": ["clinical", "expression", "mutation"],
    "pathology": ["pathology-qa", "pathology-train"],
    "survival": ["survival-baselines", "survival-deep"],
    "all":      ["splits", "mutation-paths",
                 "clinical", "expression", "mutation",
                 "pathology-qa", "pathology-train",
                 "survival-baselines", "survival-deep"],
}


@weave.op
def run_stage(name: str, force: bool) -> int:
    """Run a single pipeline stage as a subprocess. Tracked by weave."""
    probe = OUTPUT_PROBES.get(name)
    if probe and (ROOT / probe).exists() and not force:
        print(f"[skip {name}] output exists at {probe} (use --force to rerun)")
        return 0
    cmd = STAGES[name]
    print(f"\n{'='*60}\n[run {name}] {' '.join(cmd)}\n{'='*60}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


@weave.op
def run_pipeline(stage: str, force: bool) -> int:
    """Run a stage or a group of stages."""
    if stage in GROUPS:
        for s in GROUPS[stage]:
            rc = run_stage(s, force)
            if rc != 0:
                print(f"[fail {s}] rc={rc}")
                return rc
        return 0
    if stage in STAGES:
        return run_stage(stage, force)
    print(f"unknown stage {stage}; valid: {sorted(STAGES)} or groups {sorted(GROUPS)}")
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    help=f"single stage name or group ({sorted(GROUPS)})")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    weave.init(WEAVE_PROJECT)
    sys.exit(run_pipeline(args.stage, args.force))


if __name__ == "__main__":
    main()
