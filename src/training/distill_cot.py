"""Optional: distill chain-of-thought (CoT) traces from GPT-4o-mini for the
hardest tasks (AJCC stage + prognosis), then prepend them to assistant outputs
in qa_train.jsonl. Empirically yields ~+5pt on stage and prognosis F1.

Cost estimate: ~5500 train rows for stage + prognosis combined,
                ~$15 with gpt-4o-mini at $0.15/1M input + $0.60/1M output.

Requires OPENAI_API_KEY in env. Run AFTER build_multitask_qa.py.
"""
import argparse
import json
import os
import time
from pathlib import Path

import weave
import yaml
from tqdm import tqdm

from src._weave_init import init_weave


COT_TASKS = ["ajcc_stage", "prognosis_good"]

COT_SYSTEM = (
    "You are an expert pathologist. Given a pathology report and a question, "
    "produce a brief step-by-step rationale (2-4 short sentences) followed by "
    "a single-line JSON object with the answer. Use this exact format:\n\n"
    "REASONING: <brief rationale>\nANSWER: <json>"
)


@weave.op
def call_openai(client, model: str, messages: list, max_tokens: int = 256, retries: int = 3):
    """Single GPT-4o-mini call. Each invocation = one weave trace with cost/latency."""
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.2, max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)


@weave.op
def main(cfg_path: str, train_qa_path: str, out_path: str, model: str, limit: int | None) -> None:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["project_root"])

    src = root / train_qa_path
    dst = root / out_path
    print(f"reading {src}")

    rows = [json.loads(l) for l in src.open() if l.strip()]
    cot_rows = [r for r in rows if r["task"] in COT_TASKS]
    other_rows = [r for r in rows if r["task"] not in COT_TASKS]

    if limit:
        cot_rows = cot_rows[:limit]

    print(f"distilling CoT for {len(cot_rows)} rows ({COT_TASKS})")
    augmented = []
    for r in tqdm(cot_rows):
        original = r["messages"]
        # Replace assistant turn with CoT version
        user_msg = next(m for m in original if m["role"] == "user")["content"]
        gold_answer = next(m for m in original if m["role"] == "assistant")["content"]

        cot_request = [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user",   "content": user_msg + f"\n\nThe correct ANSWER is: {gold_answer}\n"
                                                     "Provide concise reasoning that justifies this answer."},
        ]
        try:
            cot = call_openai(client, model, cot_request)
        except Exception as e:
            print(f"skip ({e})")
            continue

        new_assistant = f"{cot.strip()}"
        if "ANSWER:" not in new_assistant:
            # Make sure we always end with the JSON answer the model must learn to emit
            new_assistant = f"REASONING: {cot.strip()}\nANSWER: {gold_answer}"

        augmented.append({
            "messages": [
                original[0],          # system
                original[1],          # user
                {"role": "assistant", "content": new_assistant},
            ],
            "task": r["task"],
            "TCGA_Barcode": r["TCGA_Barcode"],
            "_cot": True,
        })

    # Write merged file: non-CoT tasks unchanged + augmented CoT rows
    with dst.open("w") as f:
        for r in other_rows + augmented:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"wrote {dst}  total rows: {len(other_rows) + len(augmented)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pathology_llm.yaml")
    ap.add_argument("--train-qa", default="data/processed/pathology/qa_train.jsonl")
    ap.add_argument("--out", default="data/processed/pathology/qa_train_cot.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of CoT rows for cost control")
    args = ap.parse_args()
    init_weave("pathology-cot")
    main(args.config, args.train_qa, args.out, args.model, args.limit)
