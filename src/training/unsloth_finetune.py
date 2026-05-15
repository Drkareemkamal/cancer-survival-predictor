"""Multi-task pathology fine-tune with Unsloth + LoRA.

Default base: unsloth/Qwen2.5-7B-Instruct-bnb-4bit  -> Hub repo: PathQwen2.5
Baseline:     unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit (--base-model override)

Per configs/pathology_llm.yaml:
  LoRA r=32, alpha=32, dropout=0
  target_modules = q,k,v,o,gate,up,down + embed_tokens, lm_head
  max_seq_length = 4096, FlashAttention 2 enabled
  batch=4, grad_accum=4 (eff 16)
  epochs=5 with early-stopping (patience=3 on val_loss) -> stops at plateau
  lr=2e-4, cosine, warmup=5%, optim=adamw_8bit, bf16
  4-bit nf4 + double quant (matches Saluja 2025)

Credentials (from .env via python-dotenv):
  HF_TOKEN       -> push_to_hub auth
  HF_REPO_ID     -> falls back to "PathQwen2.5" if absent
  WANDB_API_KEY  -> wandb logging
"""
import argparse
import json
import os
from pathlib import Path

# Fix CUDA memory fragmentation before importing torch/unsloth
# Enables expandable memory segments to avoid fragmentation with large 4-bit models
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import weave
import yaml
from datasets import load_dataset
from dotenv import load_dotenv

# Load .env from project root before any other env-sensitive imports
load_dotenv()

from src._weave_init import init_weave  # noqa: E402


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@weave.op
def main(cfg: dict, base_model: str | None, output_dir: str | None) -> None:
    # Late imports — Unsloth wants to patch transformers before transformers loads
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments, EarlyStoppingCallback
    from huggingface_hub import login as hf_login
    import evaluate
    import numpy as np
    import torch

    root = Path(cfg["project_root"])
    base = base_model or cfg["base_model"]
    out_dir = Path(output_dir or cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]
    out_cfg = cfg["output"]

    # ---- HF Hub auth ------------------------------------------------------
    push_to_hub = out_cfg.get("push_to_hub", False)
    hub_model_id = None
    if push_to_hub:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("WARN: HF_TOKEN not in .env; disabling push_to_hub")
            push_to_hub = False
        else:
            hf_login(token=hf_token, add_to_git_credential=False)
            hub_model_id = (out_cfg.get("hub_model_id")
                            or os.getenv("HF_REPO_ID")
                            or "PathQwen2.5")
            print(f"will push final adapter to HF Hub -> {hub_model_id}")

    # ---- W&B auth ---------------------------------------------------------
    use_wandb = cfg.get("wandb", {}).get("enabled", False) and os.getenv("WANDB_API_KEY")
    if use_wandb:
        import wandb
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]

    # Pick chat template by model family
    template = "qwen-2.5" if "qwen" in base.lower() else "llama-3.1"

    print(f"loading base model: {base}")
    print(f"  flash_attention_2: {train_cfg.get('use_flash_attention_2', True)}")
    print(f"  4-bit quant: nf4 + double quant")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=train_cfg["max_seq_length"],
        dtype=None,            # auto bf16 on Ampere (RTX 3090)
        load_in_4bit=True,
        # Unsloth wires up FA2 internally when flash-attn is installed
    )

    tokenizer = get_chat_template(tokenizer, chat_template=template)

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        preds = np.asarray(preds)
        labels = np.asarray(labels)
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)

        vocab_size = len(tokenizer)
        preds = np.where(
            (preds >= 0) & (preds < vocab_size),
            preds,
            pad_token_id,
        ).astype(np.int64)
        labels = np.where(labels != -100, labels, pad_token_id)
        labels = np.where(
            (labels >= 0) & (labels < vocab_size),
            labels,
            pad_token_id,
        ).astype(np.int64)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        rouge_results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        bleu_results = bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels],
        )

        return {
            "rougeL": rouge_results["rougeL"],
            "bleu": bleu_results["bleu"],
            "gen_len": np.mean([len(pred.split()) for pred in decoded_preds]),
        }

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=train_cfg["seed"],
        use_rslora=False,
        loftq_config=None,
    )

    print("loading datasets")
    ds = load_dataset(
        "json",
        data_files={
            "train": str(root / cfg["data"]["qa_train"]),
            "val":   str(root / cfg["data"]["qa_val"]),
        },
    )

    def format_one(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(format_one, num_proc=4, remove_columns=ds["train"].column_names)
    print(f"  train: {len(ds['train'])}  val: {len(ds['val'])}")

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        fp16=False,
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        prediction_loss_only=False,
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=False,
        seed=train_cfg["seed"],
        report_to="wandb" if use_wandb else "none",
        run_name=out_cfg["run_name"],
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy=out_cfg.get("hub_strategy", "end"),
        hub_private_repo=out_cfg.get("hub_private_repo", True),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        packing=train_cfg["packing"],
        args=args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=train_cfg["early_stopping_patience"],
        )],
    )

    print("=== starting training ===")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  vram total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    trainer.train()

    final_dir = out_dir / "final"
    print(f"saving final adapter -> {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if push_to_hub:
        print(f"pushing to hub -> {hub_model_id}")
        trainer.push_to_hub(commit_message="Final adapter (early-stopped on val_loss)")

    (out_dir / "train_manifest.json").write_text(json.dumps({
        "base_model": base,
        "chat_template": template,
        "hub_model_id": hub_model_id,
        "lora": lora_cfg,
        "training": train_cfg,
        "n_train": len(ds["train"]),
        "n_val": len(ds["val"]),
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pathology_llm.yaml")
    ap.add_argument("--base-model", default=None,
                    help="Override the base model (e.g. for Llama baseline)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    init_weave("pathology-train")
    main(load_cfg(args.config), args.base_model, args.output_dir)
