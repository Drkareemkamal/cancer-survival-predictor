"""Run the fine-tuned pathology LLM on every report to produce two feature sets:

  1. STRUCTURED JSON  -> data/processed/features/pathology_struct.parquet
       9 fields per patient, extracted in one joint generate() call.

  2. EMBEDDINGS       -> data/processed/features/pathology_embed.parquet
       hidden_size-d mean-pooled hidden state from a quick forward pass.

Performance:
  * BATCHED generate() over multiple patients per step (default --batch-size 8)
  * Length-bucketed sampling so each batch is uniform length (less padding)
  * Left-padding for generate() so EOS detection works per row
  * Periodic checkpoint to *_partial.parquet every CHECKPOINT_EVERY patients
  * Resumable: re-run picks up where it left off

Speed budget on RTX 3090 with Qwen2.5-7B-bnb-4bit:
  batch=1  -> ~6-10 s / patient (~14 h total)
  batch=8  -> ~1.0-1.4 s / patient (~2.5 h total)
  batch=16 -> ~0.6-0.9 s / patient (~1.8 h total) — uses ~22 GB VRAM
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import weave
import yaml
from tqdm import tqdm

from src._weave_init import init_weave
from src.training.schema import PathologyExtraction, TASKS

CHECKPOINT_EVERY = 200


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# Joint prompt: ask for all 9 fields in one shot
# ----------------------------------------------------------------------------
SYSTEM_PROMPT_JOINT = (
    "You are an expert pathology AI assistant. Extract structured fields from "
    "the pathology report and respond with ONE JSON object on a single line "
    "with exactly these keys: cancer_type, primary_site, histology, ajcc_stage, "
    "t_stage, n_stage, m_stage, prior_malignancy, prognosis_good. "
    "Use the literal value null if a field cannot be determined. "
    "Do not include any prose, headers, or explanations."
)

USER_PROMPT_JOINT = "## Pathology Report:\n{text}\n\n## Output JSON (single line, all 9 keys):"


def build_joint_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_JOINT},
        {"role": "user", "content": USER_PROMPT_JOINT.format(text=text)},
    ]


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
BOOL_FIELDS = {"prior_malignancy", "prognosis_good"}

# Stable column order for Parquet — guarantees same schema across checkpoint writes
COL_ORDER = ["TCGA_Barcode"] + list(TASKS)


def _to_bool_or_none(v):
    """Coerce arbitrary LLM output into strict {True, False, None}.

    Handles every paraphrase the fine-tuned model might emit:
      True / False                    -> True / False
      "yes" / "no"                    -> True / False
      "No prior malignancy"           -> False  ("no" comes first)
      "Prior malignancy: No"          -> False  ("no" comes after the colon)
      "Yes, prior malignancy"         -> True
      "Prior malignancy present"      -> True
      None / "" / "unknown" / "n/a"   -> None
    """
    import re

    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if v in (0, 1):
            return bool(v)
        return None

    s = str(v).strip().lower()
    if not s or s in {"unknown", "n/a", "na", "none", "null", "not specified",
                      "not reported", "not stated", "not available"}:
        return None

    # Exact match (fast path)
    if s in {"true", "yes", "y", "1", "positive", "present"}:
        return True
    if s in {"false", "no", "n", "0", "negative", "absent", "not present"}:
        return False

    # Tokenize on word boundaries so "no" matches "no" but not "node"
    tokens = set(re.findall(r"[a-z0-9]+", s))
    NEG_WORDS = {"no", "false", "negative", "absent", "denies", "without"}
    POS_WORDS = {"yes", "true", "positive", "present", "has", "had", "with"}

    has_neg = bool(tokens & NEG_WORDS)
    has_pos = bool(tokens & POS_WORDS)

    # Pure negation wins (e.g., "no prior malignancy", "prior malignancy: no")
    if has_neg and not has_pos:
        return False
    # Pure affirmation
    if has_pos and not has_neg:
        return True
    # Both — need to disambiguate. The first negation/affirmation token wins.
    if has_neg and has_pos:
        first_neg = min((s.find(w) for w in NEG_WORDS if w in tokens), default=10**9)
        first_pos = min((s.find(w) for w in POS_WORDS if w in tokens), default=10**9)
        return first_pos < first_neg  # whichever appears first

    return None


def _to_str_or_none(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    s = str(v).strip()
    return s if s else None


def _normalize_record(obj: dict | None) -> dict:
    """Coerce parsed dict into our 9 fields with the RIGHT TYPES so
    Parquet schema inference is stable across checkpoint writes."""
    if not isinstance(obj, dict):
        obj = {}
    rec = {}
    for t in TASKS:
        raw = obj.get(t)
        if t in BOOL_FIELDS:
            rec[t] = _to_bool_or_none(raw)
        else:
            rec[t] = _to_str_or_none(raw)
    return rec


def _parse_json_answer(decoded: str) -> dict:
    import json_repair

    decoded = decoded.strip()
    if decoded.startswith("```"):
        decoded = decoded.strip("`")
        if decoded.lower().startswith("json"):
            decoded = decoded[4:]
    try:
        obj = json_repair.loads(decoded)
        if isinstance(obj, list) and obj:
            obj = obj[0] if isinstance(obj[0], dict) else {}
        return _normalize_record(obj if isinstance(obj, dict) else None)
    except Exception:
        return _normalize_record(None)


# ----------------------------------------------------------------------------
# Batched inference helpers
# ----------------------------------------------------------------------------
def _build_prompts(texts: list[str], tokenizer, truncate_to: int) -> list[str]:
    prompts = []
    char_cap = truncate_to * 4
    for t in texts:
        t = str(t)
        if len(t) > char_cap:
            t = t[:char_cap]
        msgs = build_joint_messages(t)
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def _embed_batch_chunked(model, inputs, sub_batch: int) -> np.ndarray:
    """Memory-safe embedding pass that processes a large batch in sub-chunks.

    Each sub-chunk runs with output_hidden_states=True (needed to access the
    final layer's hidden state), but only one chunk is in VRAM at a time, and
    we immediately free hidden states after pooling.
    """
    B, T = inputs["input_ids"].shape
    out_rows = []
    for i in range(0, B, sub_batch):
        sub = {k: v[i : i + sub_batch] for k, v in inputs.items()}
        with torch.no_grad():
            out = model(
                input_ids=sub["input_ids"],
                attention_mask=sub["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            last = out.hidden_states[-1]                              # (b, T, H)
            mask = sub["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out_rows.append(pooled.float().cpu().numpy())
            # Free GPU tensors before next sub-batch
            del out, last, mask, pooled
        torch.cuda.empty_cache()
    return np.concatenate(out_rows, axis=0)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
@weave.op
def main(cfg: dict, adapter_dir: str, batch_size: int) -> None:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    root = Path(cfg["project_root"])
    out_dir = root / cfg.get("paths", {}).get("features_dir", "data/processed/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(
        root / "data/processed/merged_tcga_data_text_dedup.csv",
        usecols=["TCGA_Barcode", "text"],
        low_memory=False,
    )
    print(f"cohort: {len(cohort)} reports")

    # ---- Resume from partial checkpoint ---------------------------------
    struct_partial = out_dir / "pathology_struct_partial.parquet"
    embed_partial = out_dir / "pathology_embed_partial.parquet"
    done_ids: set[str] = set()
    if struct_partial.exists() and embed_partial.exists():
        prev_struct = pd.read_parquet(struct_partial)
        prev_embed = pd.read_parquet(embed_partial)
        # Re-normalize prior partial in case it has stale types from old runs
        prev_struct = _struct_df_with_clean_types(prev_struct)
        done_ids = set(prev_struct["TCGA_Barcode"]) & set(prev_embed["TCGA_Barcode"])
        print(f"resuming from checkpoint: {len(done_ids)} patients done")
    else:
        prev_struct = pd.DataFrame()
        prev_embed = pd.DataFrame()

    cohort = cohort[~cohort["TCGA_Barcode"].isin(done_ids)].reset_index(drop=True)
    print(f"remaining: {len(cohort)} reports  batch_size={batch_size}")

    # ---- Length-bucket sort: shortest first reduces padding waste -------
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

    # generate() needs left-padding so per-row EOS works inside a batch
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Inference loop -------------------------------------------------
    new_struct: list[dict] = []
    new_embed: list[dict] = []

    # Adaptive embed sub-batch — starts at the full batch size, shrinks (and
    # stays shrunk) when an OOM is observed. Avoids oscillation: once we know
    # 12 OOMs, we don't try 12 again on the next batch.
    embed_sub_batch_state = {"sub": batch_size}

    def flush_partial() -> None:
        if not new_struct:
            return
        s_df = pd.concat([prev_struct, pd.DataFrame(new_struct)], ignore_index=True)
        e_df = pd.concat([prev_embed, _embed_rows_to_df(new_embed)], ignore_index=True)
        _write_struct_parquet(s_df, struct_partial)
        e_df.to_parquet(embed_partial, index=False)

    n_total = len(cohort)
    pbar = tqdm(total=n_total, desc=f"extracting (batch={batch_size})")
    done_in_loop = 0

    for start in range(0, n_total, batch_size):
        chunk = cohort.iloc[start : start + batch_size]
        texts = chunk["text"].tolist()
        ids = chunk["TCGA_Barcode"].tolist()

        prompts = _build_prompts(texts, tokenizer, truncate_to)
        # Pad to the longest prompt in this batch only (not to max_seq)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=truncate_to,
        ).to(model.device)

        # Step A: embeddings via a CHUNKED forward pass.
        # The embed pass needs output_hidden_states=True which materializes all
        # 28 layer outputs in VRAM (~3x larger than a regular forward pass).
        # We try the largest sub-batch and shrink on OOM. State across calls
        # via embed_sub_batch_state so we don't oscillate.
        target_sub = embed_sub_batch_state["sub"]
        while True:
            try:
                embeds = _embed_batch_chunked(model, inputs, sub_batch=target_sub)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if target_sub <= 1:
                    print("OOM at sub_batch=1 — re-raising")
                    raise
                target_sub = max(1, target_sub // 2)
                embed_sub_batch_state["sub"] = target_sub
                print(f"OOM during embed pass — shrinking sub_batch to {target_sub}")

        # Step B: batched generation for the JSON answers
        try:
            with torch.no_grad():
                gen = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=180,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("OOM during generate — try a smaller --batch-size")
            raise

        # Slice off the prompt part for each row
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        # With left-padding, the prompt occupies the LAST prompt_len tokens of
        # the input region, but generate() returns input_ids[..., :T_in] +
        # new tokens after T_in. So the new tokens are gen[:, T_in:].
        T_in = inputs["input_ids"].shape[1]
        new_tokens = gen[:, T_in:]
        decoded_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for j, (pid, decoded) in enumerate(zip(ids, decoded_batch)):
            rec = _parse_json_answer(decoded)
            rec["TCGA_Barcode"] = pid
            new_struct.append(rec)
            new_embed.append({"TCGA_Barcode": pid, "embed": embeds[j].astype(np.float32)})

        done_in_loop += len(ids)
        pbar.update(len(ids))

        # Periodic checkpoint
        if done_in_loop // CHECKPOINT_EVERY > (done_in_loop - len(ids)) // CHECKPOINT_EVERY:
            flush_partial()
            try:
                vram_used = torch.cuda.memory_allocated() / 1e9
                pbar.set_postfix(saved=len(new_struct) + len(prev_struct),
                                 vram=f"{vram_used:.1f}GB")
            except Exception:
                pass

    pbar.close()
    flush_partial()

    # ---- Write final files ---------------------------------------------
    struct_df = (
        pd.concat([prev_struct, pd.DataFrame(new_struct)], ignore_index=True)
        if new_struct else prev_struct
    )
    embed_df = (
        pd.concat([prev_embed, _embed_rows_to_df(new_embed)], ignore_index=True)
        if new_embed else prev_embed
    )

    struct_out = out_dir / "pathology_struct.parquet"
    embed_out = out_dir / "pathology_embed.parquet"
    _write_struct_parquet(struct_df, struct_out)
    embed_df.to_parquet(embed_out, index=False)
    print(f"wrote {struct_out}  shape={struct_df.shape}")
    print(f"wrote {embed_out}  shape={embed_df.shape}")

    for p in (struct_partial, embed_partial):
        if p.exists():
            p.unlink()


def _struct_df_with_clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a structured-output DataFrame so each column has a stable type.

    Critical for resumable Parquet writes: pyarrow infers schema from the
    first non-null value, so mixed-type columns crash on the second write.
    """
    if df is None or df.empty:
        return df

    # Preserve TCGA_Barcode as string
    if "TCGA_Barcode" in df.columns:
        df["TCGA_Barcode"] = df["TCGA_Barcode"].astype(str)

    # Ensure all task columns exist in the canonical order
    for t in TASKS:
        if t not in df.columns:
            df[t] = None

    # Coerce booleans cleanly, strings cleanly
    for t in TASKS:
        if t in BOOL_FIELDS:
            df[t] = df[t].apply(_to_bool_or_none).astype("boolean")  # nullable bool
        else:
            df[t] = df[t].apply(_to_str_or_none).astype("string")    # nullable str
    return df[COL_ORDER]


def _write_struct_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write the structured-output DataFrame with explicit pandas dtypes
    so pyarrow infers a stable schema across writes."""
    df_clean = _struct_df_with_clean_types(df.copy())
    df_clean.to_parquet(path, index=False)


def _embed_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    arr = np.stack([r["embed"] for r in rows]).astype(np.float32)
    df = pd.DataFrame(arr, columns=[f"E{i}" for i in range(arr.shape[1])])
    df.insert(0, "TCGA_Barcode", [r["TCGA_Barcode"] for r in rows])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pathology_llm.yaml")
    ap.add_argument("--adapter-dir", required=True,
                    help="Path to LoRA adapter directory (e.g. models/PathQwen2.5/final)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Patients per forward pass. RTX 3090: 8 = ~16 GB, 16 = ~22 GB.")
    args = ap.parse_args()
    init_weave("pathology-extract")
    main(load_cfg(args.config), args.adapter_dir, args.batch_size)
