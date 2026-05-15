"""Per-task feature extraction — matches the exact prompts used during training.

This is the alternative to extract_features.py (which uses a single joint prompt
for all 9 tasks). The joint prompt is ~9× faster but produces free-text answers
the model wasn't trained to emit; this per-task script asks the SAME questions
the model saw during fine-tuning, giving paper-grade classification accuracy at
the cost of 9× more forward passes.

Inputs:
  * Adapter dir from --adapter-dir (default: models/PathQwen2.5/final)
  * Test cohort from data/processed/merged_tcga_data_text_dedup.csv

Outputs:
  * data/processed/features/pathology_struct.parquet      (8459 × 10)
  * data/processed/features/pathology_embed.parquet       (8459 × 3585)

Performance budget on RTX 3090 with Qwen2.5-7B-bnb-4bit:
  batch=4   ~3 hours for full 8459 cohort
  batch=8   ~2 hours    (uses ~18 GB VRAM)
  batch=12  ~1.5 hours  (uses ~22 GB)

Same robustness features as extract_features.py:
  * Length-bucketed sort for less padding waste
  * Left-padding (required for batched generate)
  * Periodic *_partial.parquet checkpoints every 200 patients (resumable)
  * Adaptive embed sub-batch with OOM-shrink
  * Strict type coercion (bool fields kept as bool in Parquet)

Run:
  python -m src.training.extract_features_per_task \\
      --config configs/pathology_llm.yaml \\
      --adapter-dir models/PathQwen2.5/final \\
      --batch-size 8
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import weave
import yaml
from tqdm import tqdm

from src._weave_init import init_weave
from src.training.schema import TASKS
from src.training.extract_features import (
    _to_bool_or_none, _to_str_or_none, BOOL_FIELDS, COL_ORDER,
    _struct_df_with_clean_types, _write_struct_parquet,
    _embed_batch_chunked, _embed_rows_to_df,
)

CHECKPOINT_EVERY = 100


# ---------------------------------------------------------------------------
# Constrained-decoding label vocabularies (used only when --constrained-decoding)
# ---------------------------------------------------------------------------
# Maps each closed-set task -> list of valid label strings the model is allowed
# to emit. Restricting decoding to these strings eliminates unparseable outputs.
CLOSED_SET_LABELS: dict[str, list[str]] = {
    "ajcc_stage":       ["Stage I", "Stage II", "Stage III", "Stage IV"],
    "t_stage":          ["T0", "T1", "T2", "T3", "T4", "Tis", "TX"],
    "n_stage":          ["N0", "N1", "N2", "N3", "NX"],
    "m_stage":          ["M0", "M1", "MX"],
    "prior_malignancy": ["true", "false"],
    "prognosis_good":   ["true", "false"],
}


class _ConstrainedJsonProcessor:
    """A `LogitsProcessor` that forces generated text to match exactly one of
    the strings in `allowed_labels` (wrapped in {"<task>": "<label>"} JSON).

    Works by computing the union of token-ID prefixes for every allowed full
    completion, then at each step masking logits to keep only tokens that
    extend at least one allowed prefix.

    Per-row tracking via `self._row_states` lets us apply different label sets
    per batch row, but in practice we run one task at a time so all rows in a
    batch share the same constraint — much simpler and faster.
    """

    def __init__(self, tokenizer, allowed_labels: list[str], task: str):
        # The model has to emit literally:  {"<task>": "<label>"}
        # We pre-tokenize every full completion and store token sequences.
        completions = [f'{{"{task}": "{lab}"}}' for lab in allowed_labels]
        # For booleans the gold is unquoted true/false, so build BOTH variants
        if task in BOOL_FIELDS:
            completions = [f'{{"{task}": {lab}}}' for lab in allowed_labels]
        self.token_seqs: list[list[int]] = [
            tokenizer.encode(c, add_special_tokens=False) for c in completions
        ]
        self.tokenizer = tokenizer
        self._step = 0   # how many tokens we've emitted so far for this batch
        self.batch_size = None

    def _allowed_at_step(self, step: int) -> set[int]:
        """Set of token IDs that are valid as the `step`-th generated token."""
        allowed: set[int] = set()
        for seq in self.token_seqs:
            if step < len(seq):
                allowed.add(seq[step])
        return allowed

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # scores shape: (batch, vocab). We only allow tokens that extend an
        # allowed prefix at the current step.
        if self.batch_size is None:
            self.batch_size = scores.size(0)
        allowed = self._allowed_at_step(self._step)
        if not allowed:
            # We've consumed every step in every completion -> force EOS
            allowed = {self.tokenizer.eos_token_id}
        mask = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(sorted(allowed), device=scores.device, dtype=torch.long)
        mask[:, idx] = scores[:, idx]
        self._step += 1
        return mask


# --- Exact training prompts (copied verbatim from build_multitask_qa.py) ----
SYSTEM_PROMPT = (
    "You are an expert pathology AI assistant. "
    "Analyze the pathology report below and extract the requested field. "
    "Respond ONLY with a single-line JSON object matching the requested schema field. "
    "Do not include any explanations, headers, or prose."
)

TASK_PROMPTS = {
    "cancer_type":      "What is the TCGA study cancer type? Output: {\"cancer_type\": \"<label>\"}",
    "primary_site":     "What is the anatomical primary site? Output: {\"primary_site\": \"<text>\"}",
    "histology":        "What is the histological diagnosis (ICD-O-3 morphology)? Output: {\"histology\": \"<text>\"}",
    "ajcc_stage":       "What is the AJCC overall pathological stage (Stage I/II/III/IV)? Output: {\"ajcc_stage\": \"<label>\"}",
    "t_stage":          "What is the pathological T stage (T0–T4, Tis, TX)? Output: {\"t_stage\": \"<label>\"}",
    "n_stage":          "What is the pathological N stage (N0–N3, NX)? Output: {\"n_stage\": \"<label>\"}",
    "m_stage":          "What is the pathological M stage (M0, M1, MX)? Output: {\"m_stage\": \"<label>\"}",
    "prior_malignancy": "Did this patient have a prior malignancy? Output: {\"prior_malignancy\": <true|false>}",
    "prognosis_good":   "Will this patient likely survive past the mean disease-specific survival time for their cancer type? Output: {\"prognosis_good\": <true|false>}",
}


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_messages_for_task(text: str, task: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"## Pathology Report:\n{text}\n\n## Question:\n{TASK_PROMPTS[task]}"},
    ]


def _parse_task_answer(decoded: str, task: str):
    """Strip the model's JSON answer for one task. Returns the value or None."""
    import json_repair
    s = decoded.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    # Find the LAST JSON-looking block (tolerates accidental CoT prefix)
    matches = list(re.finditer(r"\{[^{}]*\}", s))
    obj = None
    for m in reversed(matches):
        try:
            obj = json_repair.loads(m.group(0))
            if isinstance(obj, dict):
                break
        except Exception:
            continue
    if not isinstance(obj, dict):
        return None
    val = obj.get(task)
    if val is None and len(obj) == 1:
        val = next(iter(obj.values()))
    if task in BOOL_FIELDS:
        return _to_bool_or_none(val)
    return _to_str_or_none(val)


@weave.op
def main(cfg: dict, adapter_dir: str, batch_size: int,
         constrained_decoding: bool = False) -> None:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from transformers import LogitsProcessorList

    root = Path(cfg["project_root"])
    out_dir = root / cfg.get("paths", {}).get("features_dir", "data/processed/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(
        root / "data/processed/merged_tcga_data_text_dedup.csv",
        usecols=["TCGA_Barcode", "text"], low_memory=False,
    )
    print(f"cohort: {len(cohort)} reports")

    # ---- Resume from partial checkpoint if exists ------------------------
    struct_partial = out_dir / "pathology_struct_partial.parquet"
    embed_partial  = out_dir / "pathology_embed_partial.parquet"
    done_ids: set[str] = set()
    if struct_partial.exists() and embed_partial.exists():
        prev_struct = pd.read_parquet(struct_partial)
        prev_embed  = pd.read_parquet(embed_partial)
        prev_struct = _struct_df_with_clean_types(prev_struct)
        done_ids = set(prev_struct["TCGA_Barcode"]) & set(prev_embed["TCGA_Barcode"])
        print(f"resuming: {len(done_ids)} patients done")
    else:
        prev_struct = pd.DataFrame()
        prev_embed  = pd.DataFrame()

    cohort = cohort[~cohort["TCGA_Barcode"].isin(done_ids)].reset_index(drop=True)
    print(f"remaining: {len(cohort)} reports  batch_size={batch_size}")

    # Length-bucket sort
    cohort["_len"] = cohort["text"].astype(str).str.len()
    cohort = cohort.sort_values("_len", kind="stable").reset_index(drop=True)

    # ---- Load model -----------------------------------------------------
    base = cfg["base_model"]
    template = "qwen-2.5" if "qwen" in base.lower() else "llama-3.1"
    max_seq = cfg["training"]["max_seq_length"]
    truncate_to = max_seq - 256

    print(f"loading {base} + adapter {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=template)
    FastLanguageModel.for_inference(model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    new_struct: list[dict] = []
    new_embed: list[dict] = []
    embed_sub_batch_state = {"sub": batch_size}

    def flush_partial() -> None:
        if not new_struct:
            return
        s_df = pd.concat([prev_struct, pd.DataFrame(new_struct)], ignore_index=True)
        e_df = pd.concat([prev_embed,  _embed_rows_to_df(new_embed)], ignore_index=True)
        _write_struct_parquet(s_df, struct_partial)
        e_df.to_parquet(embed_partial, index=False)

    n_total = len(cohort)
    pbar = tqdm(total=n_total, desc=f"per-task extract (batch={batch_size})")
    done_in_loop = 0

    for start in range(0, n_total, batch_size):
        chunk = cohort.iloc[start: start + batch_size]
        texts = chunk["text"].tolist()
        ids = chunk["TCGA_Barcode"].tolist()
        char_cap = truncate_to * 4
        texts = [t[:char_cap] if len(t) > char_cap else t for t in (str(x) for x in texts)]

        # ----- ONE forward pass for embeddings (reused across all 9 tasks) -----
        embed_prompt = "## Pathology Report:\n" + "\n---\n".join(texts)  # not used directly
        # We embed each report individually (same as joint script): build embed-only inputs
        embed_msgs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"## Pathology Report:\n{t}"}],
                tokenize=False, add_generation_prompt=False,
            )
            for t in texts
        ]
        embed_inputs = tokenizer(
            embed_msgs, return_tensors="pt", padding=True,
            truncation=True, max_length=truncate_to,
        ).to(model.device)
        target_sub = embed_sub_batch_state["sub"]
        while True:
            try:
                embeds = _embed_batch_chunked(model, embed_inputs, sub_batch=target_sub)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if target_sub <= 1:
                    raise
                target_sub = max(1, target_sub // 2)
                embed_sub_batch_state["sub"] = target_sub
                print(f"OOM in embed pass — shrinking sub_batch to {target_sub}")

        # ----- NINE generate() passes, one per task ---------------------------
        # Initialize per-patient records
        records = [{"TCGA_Barcode": pid} for pid in ids]

        for task in TASKS:
            prompts = []
            for t in texts:
                msgs = build_messages_for_task(t, task)
                prompts.append(tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=truncate_to,
            ).to(model.device)

            # Constrained-decoding path for closed-set tasks
            logits_processor = None
            if constrained_decoding and task in CLOSED_SET_LABELS:
                proc = _ConstrainedJsonProcessor(
                    tokenizer, CLOSED_SET_LABELS[task], task,
                )
                logits_processor = LogitsProcessorList([proc])

            with torch.no_grad():
                gen = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=48,                 # one short JSON answer is enough
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    logits_processor=logits_processor,
                )
            T_in = inputs["input_ids"].shape[1]
            new_tokens = gen[:, T_in:]
            decoded_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for j, decoded in enumerate(decoded_batch):
                records[j][task] = _parse_task_answer(decoded, task)

        # Stash
        for j, rec in enumerate(records):
            new_struct.append(rec)
            new_embed.append({"TCGA_Barcode": ids[j],
                              "embed": embeds[j].astype(np.float32)})

        done_in_loop += len(ids)
        pbar.update(len(ids))

        if done_in_loop // CHECKPOINT_EVERY > (done_in_loop - len(ids)) // CHECKPOINT_EVERY:
            flush_partial()
            try:
                vram = torch.cuda.memory_allocated() / 1e9
                pbar.set_postfix(saved=len(new_struct) + len(prev_struct),
                                 vram=f"{vram:.1f}GB")
            except Exception:
                pass

    pbar.close()
    flush_partial()

    # ---- Write final files ---------------------------------------------
    struct_df = (pd.concat([prev_struct, pd.DataFrame(new_struct)], ignore_index=True)
                 if new_struct else prev_struct)
    embed_df  = (pd.concat([prev_embed, _embed_rows_to_df(new_embed)], ignore_index=True)
                 if new_embed else prev_embed)
    struct_out = out_dir / "pathology_struct.parquet"
    embed_out  = out_dir / "pathology_embed.parquet"
    _write_struct_parquet(struct_df, struct_out)
    embed_df.to_parquet(embed_out, index=False)
    print(f"wrote {struct_out}  shape={struct_df.shape}")
    print(f"wrote {embed_out}  shape={embed_df.shape}")

    for p in (struct_partial, embed_partial):
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pathology_llm.yaml")
    ap.add_argument("--adapter-dir", required=True,
                    help="Path to LoRA adapter dir (e.g. models/PathQwen2.5/final)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Patients per forward pass. RTX 3090: 8 = ~18 GB.")
    ap.add_argument("--constrained-decoding", action="store_true",
                    help="Force closed-set tasks (AJCC stage, T/N/M, booleans) "
                         "to emit only valid labels via logits masking. "
                         "Eliminates unparseable predictions, expected +5-10 pts "
                         "on AJCC stage / prognosis. Does not affect open-text "
                         "tasks (cancer_type, primary_site, histology).")
    args = ap.parse_args()
    init_weave("pathology-extract-per-task")
    main(load_cfg(args.config), args.adapter_dir, args.batch_size,
         constrained_decoding=args.constrained_decoding)
