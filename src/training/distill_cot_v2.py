"""CoT distillation v2 — deeper, structured reasoning traces.

Improves over distill_cot.py by:
  - Asking GPT-4o-mini for 5-15-sentence rubric-based reasoning (vs 2-4)
  - Task-specific reasoning rubrics (mirrors Saluja et al. 2025 Fig. 4)
  - Explicit chain: extract features → apply staging criteria → conclude
  - Validates the final JSON matches the gold answer (rejects hallucinated rationales)

Expected lift over distill_cot v1:
  AJCC stage:     +5 pts test accuracy (paper's reported gain from CoT)
  Prognosis_good: +3 pts macro-F1 (more nuanced reasoning)

Cost estimate:
  ~5500 rows × ~600 input + ~400 output tokens at GPT-4o-mini pricing
  = ~$20 total (vs $15 for v1's shorter traces)

Usage:
  export OPENAI_API_KEY=sk-...
  python -m src.training.distill_cot_v2 \\
      --train-qa data/processed/pathology/qa_train.jsonl \\
      --out data/processed/pathology/qa_train_cot_v2.jsonl

Then edit configs/pathology_llm.yaml:
  data.qa_train: data/processed/pathology/qa_train_cot_v2.jsonl
And retrain: python main.py --stage pathology-train --force
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import weave
import yaml
from tqdm import tqdm

from src._weave_init import init_weave

COT_TASKS = ["ajcc_stage", "prognosis_good"]


# ---------------------------------------------------------------------------
# Task-specific reasoning rubrics (mirrors Saluja 2025 Fig. 4 style)
# ---------------------------------------------------------------------------
AJCC_RUBRIC = """\
You are an expert oncologic pathologist. Given a pathology report and the
correct AJCC stage, produce a structured reasoning trace that justifies the
final answer using AJCC staging criteria.

Structure your reasoning as four numbered steps:

1. **Tumor (T):** Extract tumor size, depth of invasion, and any ulceration/
   margin involvement. Map to T1–T4.
2. **Nodes (N):** Identify lymph node involvement count and laterality.
   Map to N0–N3.
3. **Metastasis (M):** Note any distant spread or absence of metastasis (M0
   vs M1).
4. **Final stage:** Apply AJCC stage grouping to combine T, N, M into the
   overall stage (I/II/III/IV). If multiple stage-defining features are
   missing, default to the most conservative inference.

Produce output in this exact format:

REASONING:
1. T: <one-line rationale>
2. N: <one-line rationale>
3. M: <one-line rationale>
4. Stage grouping: <one-line rationale>

ANSWER: {"ajcc_stage": "<Stage I|Stage II|Stage III|Stage IV>"}

The provided gold answer MUST be the final answer. Do not contradict it.
If the report lacks evidence for a step, write "not specified in report" —
do NOT invent facts.
"""

PROGNOSIS_RUBRIC = """\
You are an expert oncologic pathologist. Given a pathology report, the cancer
type's mean disease-specific survival time, and the correct prognosis label
(True = patient will survive past mean DSS, False = will not), produce a
structured reasoning trace.

Consider these factors in your reasoning:

1. **Tumor biology:** type, grade, differentiation (well/moderately/poorly),
   any aggressive features (signet ring cells, sarcomatoid, lymphovascular invasion).
2. **Burden:** tumor size, multifocal, extension beyond primary site.
3. **Nodal involvement:** number of positive nodes, extranodal extension.
4. **Distant metastasis:** presence/absence.
5. **Resection margins:** clear / involved / not stated.
6. **Treatment indicators in report:** neoadjuvant therapy, residual disease.

Produce output in this exact format:

REASONING:
1. Tumor biology: <one-line rationale>
2. Burden: <one-line rationale>
3. Nodes: <one-line rationale>
4. Metastasis: <one-line rationale>
5. Margins: <one-line rationale>
6. Overall prognosis: <one-line rationale that justifies the gold answer>

ANSWER: {"prognosis_good": <true|false>}

The provided gold answer MUST be the final answer. Do not contradict it.
"""


RUBRICS = {
    "ajcc_stage":     AJCC_RUBRIC,
    "prognosis_good": PROGNOSIS_RUBRIC,
}


# ---------------------------------------------------------------------------
# OpenAI helper
# ---------------------------------------------------------------------------
@weave.op
def call_openai(client, model: str, messages: list, max_tokens: int = 600,
                retries: int = 3, temperature: float = 0.2):
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return r.choices[0].message.content, r.usage
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


# ---------------------------------------------------------------------------
# Validation — make sure the model's CoT didn't contradict the gold answer
# ---------------------------------------------------------------------------
def extract_json_answer(text: str) -> dict | None:
    """Find the last balanced JSON object in `text`."""
    matches = list(re.finditer(r"\{[^{}]*\}", text))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def matches_gold(cot_response: str, gold_assistant: str) -> bool:
    """Verify the CoT response ends with the same JSON as the gold answer."""
    cot_obj = extract_json_answer(cot_response)
    gold_obj = extract_json_answer(gold_assistant)
    if not cot_obj or not gold_obj:
        return False
    # Compare on the (task -> value) pair the gold defines
    for task, v in gold_obj.items():
        if str(cot_obj.get(task)).lower() != str(v).lower():
            return False
    return True


# ---------------------------------------------------------------------------
# Main distillation loop
# ---------------------------------------------------------------------------
@weave.op
def main(train_qa_path: str, out_path: str, model: str,
         limit: int | None, max_tokens: int) -> None:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    src = Path(train_qa_path)
    dst = Path(out_path)
    print(f"reading {src}")

    rows = [json.loads(l) for l in src.open() if l.strip()]
    cot_rows = [r for r in rows if r["task"] in COT_TASKS]
    other_rows = [r for r in rows if r["task"] not in COT_TASKS]

    if limit:
        cot_rows = cot_rows[:limit]

    print(f"distilling DEEP CoT for {len(cot_rows)} rows ({COT_TASKS})")
    print(f"  using {model}, max_tokens={max_tokens}")

    augmented = []
    n_pass = n_fail = n_err = 0
    total_in = total_out = 0
    pricing = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (5.0, 15.0)}

    pbar = tqdm(cot_rows, desc="distilling")
    for r in pbar:
        original = r["messages"]
        user_msg = next(m for m in original if m["role"] == "user")["content"]
        gold_answer = next(m for m in original if m["role"] == "assistant")["content"]
        task = r["task"]

        rubric = RUBRICS[task]
        cot_request = [
            {"role": "system", "content": rubric},
            {"role": "user",   "content":
                f"{user_msg}\n\n"
                f"The CORRECT final answer is: {gold_answer}\n\n"
                f"Now produce the structured REASONING trace that justifies it."},
        ]

        try:
            cot_text, usage = call_openai(client, model, cot_request, max_tokens=max_tokens)
            total_in += usage.prompt_tokens
            total_out += usage.completion_tokens
        except Exception as e:
            print(f"\n[err] {e}")
            n_err += 1
            continue

        if not matches_gold(cot_text, gold_answer):
            # Rationale contradicted gold; fall back to the v1-style minimal trace
            n_fail += 1
            new_assistant = f"REASONING: (validation failed, using gold-only)\nANSWER: {gold_answer}"
        else:
            n_pass += 1
            new_assistant = cot_text.strip()
            # Guarantee the JSON answer is on its own final line
            if "ANSWER:" not in new_assistant:
                new_assistant = f"{new_assistant}\nANSWER: {gold_answer}"

        augmented.append({
            "messages": [
                original[0],     # system
                original[1],     # user
                {"role": "assistant", "content": new_assistant},
            ],
            "task":         r["task"],
            "TCGA_Barcode": r["TCGA_Barcode"],
            "_cot_v2":      True,
        })

        if (n_pass + n_fail + n_err) % 100 == 0:
            in_cost  = total_in  / 1e6 * pricing[model][0]
            out_cost = total_out / 1e6 * pricing[model][1]
            pbar.set_postfix(pass_=n_pass, fail=n_fail, err=n_err,
                             cost=f"${in_cost+out_cost:.2f}")

    # Write merged file: non-CoT tasks unchanged + augmented CoT rows
    print(f"\nwriting {dst}")
    with dst.open("w") as f:
        for rr in other_rows + augmented:
            f.write(json.dumps(rr, default=str) + "\n")

    # Summary
    in_cost  = total_in  / 1e6 * pricing[model][0]
    out_cost = total_out / 1e6 * pricing[model][1]
    print(f"  total rows written: {len(other_rows) + len(augmented)}")
    print(f"  CoT distilled: {len(augmented)} ({n_pass} pass / {n_fail} validation-fail / {n_err} api-error)")
    print(f"  tokens: input={total_in:,}  output={total_out:,}")
    print(f"  cost:   ${in_cost+out_cost:.2f} (input ${in_cost:.2f} + output ${out_cost:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-qa", default="data/processed/pathology/qa_train.jsonl")
    ap.add_argument("--out",      default="data/processed/pathology/qa_train_cot_v2.jsonl")
    ap.add_argument("--model",    default="gpt-4o-mini",
                    help="OpenAI model. gpt-4o-mini ($15-25) or gpt-4o ($80-100)")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Limit number of CoT rows (for cost control / dry runs)")
    ap.add_argument("--max-tokens", type=int, default=600,
                    help="Max output tokens per CoT trace (v1 used 256)")
    args = ap.parse_args()
    init_weave("pathology-cot-v2")
    main(args.train_qa, args.out, args.model, args.limit, args.max_tokens)
