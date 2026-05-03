"""
Strategy 3: Hierarchical Two-Stage Fine-Tuning (OpenBioLLM-8B, 4-bit)
===================================================================
Stage 1: Pan-cancer fine-tuning on ALL samples.
Stage 2: Per-cancer-type fine-tuning from the Stage 1 LoRA checkpoint,
         only for cancer types with enough samples (500+ and 5%+ event rate).

Note: Because OpenBioLLM-8B uses bitsandbytes 4-bit quantization, we can only
save/reload LoRA adapter weights (not the full model state_dict). The base
quantized model is reloaded fresh each time and the adapters are applied on top.
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

MIN_SAMPLES = 500
MIN_EVENT_RATE = 0.05


class LlamaSurvivalDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.dropna(subset=['text', 'OS_MONTHS', 'OS_STATUS']).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, 'text'])
        duration = float(self.df.loc[idx, 'OS_MONTHS'])
        status_str = str(self.df.loc[idx, 'OS_STATUS'])
        event = 1.0 if 'DECEASED' in status_str else 0.0
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'duration': torch.tensor(duration, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32),
            'patient_id': str(self.df.loc[idx, 'TCGA_Barcode']),
            'cancer_type': str(self.df.loc[idx, 'DISEASE_TYPE'])
        }


def create_llama_model(model_name="aaditya/Llama3-OpenBioLLM-8B"):
    """Create a fresh quantized Llama model with LoRA adapters and a risk head."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    base_model = AutoModel.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)
    peft_config = LoraConfig(
        task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
    )
    peft_model = get_peft_model(base_model, peft_config)
    return peft_model


class SurvivalLlamaModel(torch.nn.Module):
    def __init__(self, peft_model):
        super().__init__()
        self.base_model = peft_model
        self.risk_head = torch.nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        seq_len = attention_mask.sum(dim=1) - 1
        bs = input_ids.shape[0]
        seq_len = torch.clamp(seq_len, min=0, max=input_ids.shape[1] - 1)
        cls_emb = outputs.last_hidden_state[torch.arange(bs, device=input_ids.device), seq_len]
        cls_emb = cls_emb.to(self.risk_head.weight.dtype)
        return self.risk_head(cls_emb).squeeze(-1), cls_emb


def cox_ph_loss(log_h, events, durations):
    idx = torch.argsort(durations, descending=True)
    events = events[idx]; log_h = log_h[idx]
    log_h_max = torch.max(log_h)
    risk_set_sums = torch.cumsum(torch.exp(log_h - log_h_max), dim=0)
    log_risk_set_sums = torch.log(risk_set_sums) + log_h_max
    loss = -(log_h - log_risk_set_sums) * events
    return loss.sum() / (events.sum() + 1e-8)


def train_loop(model, train_loader, val_loader, device, optimizer,
               max_epochs, patience, save_dir, stage_name,
               use_wandb=False, wandb_prefix=""):
    """Training loop. Saves LoRA adapters + risk_head separately."""
    best_val = float('inf'); best_epoch = 0; no_improve = 0
    train_losses, val_losses = [], []
    os.makedirs(save_dir, exist_ok=True)
    risk_head_path = os.path.join(save_dir, 'risk_head.pt')

    for epoch in range(max_epochs):
        model.train()
        t_loss, t_n = 0, 0
        pbar = tqdm(train_loader, desc=f"{stage_name} Ep {epoch+1}/{max_epochs} [Train]")
        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            dur = batch['duration'].to(device)
            ev = batch['event'].to(device)
            optimizer.zero_grad()
            rs, _ = model(ids, mask)
            loss = cox_ph_loss(rs, ev, dur)
            if torch.isnan(loss): continue
            loss.backward(); optimizer.step()
            t_loss += loss.item(); t_n += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            if use_wandb:
                import wandb; wandb.log({f"{wandb_prefix}train/batch_loss": loss.item()})

        avg_t = t_loss / max(t_n, 1)
        train_losses.append(avg_t)

        model.eval()
        v_loss, v_n = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"{stage_name} Ep {epoch+1} [Val]", leave=False):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                dur = batch['duration'].to(device)
                ev = batch['event'].to(device)
                rs, _ = model(ids, mask)
                loss = cox_ph_loss(rs, ev, dur)
                if not torch.isnan(loss): v_loss += loss.item(); v_n += 1

        avg_v = v_loss / max(v_n, 1)
        val_losses.append(avg_v)
        print(f"  {stage_name} Ep {epoch+1}: train={avg_t:.4f} val={avg_v:.4f}")

        if use_wandb:
            import wandb
            wandb.log({f"{wandb_prefix}train/epoch_loss": avg_t,
                       f"{wandb_prefix}val/epoch_loss": avg_v, f"{wandb_prefix}epoch": epoch+1})

        if avg_v < best_val:
            best_val = avg_v; best_epoch = epoch + 1; no_improve = 0
            # Save LoRA adapters
            model.base_model.save_pretrained(save_dir)
            # Save risk head separately
            torch.save(model.risk_head.state_dict(), risk_head_path)
            print(f"    >>> Best saved (val={best_val:.4f})")
        else:
            no_improve += 1
            print(f"    --- No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"    *** Early stopping! Best was epoch {best_epoch} ***")
                break

    return train_losses, val_losses, best_epoch


def load_best_model(save_dir, model_name, device):
    """Reload a fresh quantized model and apply saved LoRA adapters + risk head."""
    peft_model = create_llama_model(model_name)
    # Load saved LoRA weights
    from peft import set_peft_model_state_dict
    import safetensors.torch
    adapter_path = os.path.join(save_dir, 'adapter_model.safetensors')
    if os.path.exists(adapter_path):
        state_dict = safetensors.torch.load_file(adapter_path)
        set_peft_model_state_dict(peft_model, state_dict)

    model = SurvivalLlamaModel(peft_model)
    # Load risk head
    risk_head_path = os.path.join(save_dir, 'risk_head.pt')
    if os.path.exists(risk_head_path):
        model.risk_head.load_state_dict(torch.load(risk_head_path, map_location=device))
    model.risk_head = model.risk_head.to(device)
    return model


def run_hierarchical(data_path, output_dir,
                     stage1_epochs=20, stage2_epochs=10,
                     batch_size=4, lr=2e-5, stage2_lr=1e-5,
                     patience=3, val_split=0.15,
                     hf_token=None, hf_repo_id=None,
                     wandb_api_key=None, wandb_project=None):

    use_wandb = wandb_project and wandb_api_key
    model_name = "aaditya/Llama3-OpenBioLLM-8B"

    if hf_token:
        from huggingface_hub import login; login(token=hf_token)
    if use_wandb:
        import wandb; wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, name="Llama3-Hierarchical", config={
            "strategy": "hierarchical_two_stage", "model": model_name,
            "stage1_epochs": stage1_epochs, "stage2_epochs": stage2_epochs,
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    full_dataset = LlamaSurvivalDataset(df, tokenizer)

    # ===================================================================
    # STAGE 1: Pan-Cancer
    # ===================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: Pan-Cancer Fine-Tuning (OpenBioLLM-8B)")
    print(f"{'='*60}")

    total = len(full_dataset)
    val_n = int(total * val_split); train_n = total - val_n
    train_ds, val_ds = random_split(full_dataset, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    print(f"Split: {train_n} train / {val_n} val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    peft_model = create_llama_model(model_name)
    model = SurvivalLlamaModel(peft_model).to(device)
    model.risk_head = model.risk_head.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    stage1_dir = os.path.join(output_dir, 'llama_stage1_pancancer')
    s1_train, s1_val, s1_best = train_loop(
        model, train_loader, val_loader, device, optimizer,
        stage1_epochs, patience, stage1_dir,
        stage_name="[Stage1]", use_wandb=use_wandb, wandb_prefix="stage1/"
    )

    # Plot Stage 1
    fig, ax = plt.subplots(figsize=(12, 7))
    ep = range(1, len(s1_train) + 1)
    ax.plot(ep, s1_train, marker='o', label='Train'); ax.plot(ep, s1_val, marker='s', label='Val')
    ax.axvline(x=s1_best, color='green', linestyle='--', label=f'Best ({s1_best})')
    ax.set_title('Stage 1: Pan-Cancer OpenBioLLM-8B', fontsize=14)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'llama_hierarchical_stage1.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Push Stage 1 to HF Hub
    if hf_token and hf_repo_id:
        s1_model = load_best_model(stage1_dir, model_name, device)
        s1_repo = hf_repo_id + "-llama-hierarchical-stage1"
        print(f"Pushing Stage 1 to HF Hub: {s1_repo}...")
        s1_model.base_model.push_to_hub(s1_repo, token=hf_token)
        tokenizer.push_to_hub(s1_repo, token=hf_token)
        del s1_model

    # Free GPU memory before Stage 2
    del model, peft_model, optimizer
    torch.cuda.empty_cache()

    # ===================================================================
    # STAGE 2: Per-Cancer-Type
    # ===================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: Per-Cancer-Type Fine-Tuning (OpenBioLLM-8B)")
    print(f"{'='*60}")

    clean_df = full_dataset.df
    clean_df['_event'] = clean_df['OS_STATUS'].astype(str).str.contains('DECEASED').astype(int)
    type_stats = clean_df.groupby('DISEASE_TYPE')['_event'].agg(['count', 'sum', 'mean'])
    type_stats.columns = ['total', 'deaths', 'event_rate']
    viable = type_stats[(type_stats['total'] >= MIN_SAMPLES) & (type_stats['event_rate'] >= MIN_EVENT_RATE)]

    print(f"\nViable cancer types:")
    for ct, row in viable.iterrows():
        print(f"  {ct}: {int(row['total'])} samples, {row['event_rate']:.1%} event rate")

    stage2_results = {}

    for cancer_type in viable.index:
        print(f"\n--- Stage 2: {cancer_type} ---")

        ct_indices = [i for i in range(len(full_dataset)) if full_dataset.df.loc[i, 'DISEASE_TYPE'] == cancer_type]
        ct_dataset = Subset(full_dataset, ct_indices)
        ct_val_n = max(int(len(ct_indices) * val_split), 1)
        ct_train_n = len(ct_indices) - ct_val_n
        ct_train, ct_val = random_split(ct_dataset, [ct_train_n, ct_val_n],
                                        generator=torch.Generator().manual_seed(42))

        ct_train_loader = DataLoader(ct_train, batch_size=batch_size, shuffle=True)
        ct_val_loader = DataLoader(ct_val, batch_size=batch_size, shuffle=False)

        # Load Stage 1 model as starting point
        ct_model = load_best_model(stage1_dir, model_name, device)
        ct_optimizer = torch.optim.AdamW(ct_model.parameters(), lr=stage2_lr)

        safe_name = cancer_type.replace(' ', '_').replace(',', '').lower()[:40]
        ct_save_dir = os.path.join(output_dir, f'llama_stage2_{safe_name}')

        s2_train, s2_val, s2_best = train_loop(
            ct_model, ct_train_loader, ct_val_loader, device, ct_optimizer,
            stage2_epochs, patience, ct_save_dir,
            stage_name=f"[{cancer_type[:20]}]",
            use_wandb=use_wandb, wandb_prefix=f"stage2_{safe_name}/"
        )

        stage2_results[cancer_type] = {
            'train_losses': s2_train, 'val_losses': s2_val,
            'best_epoch': s2_best, 'best_val_loss': min(s2_val),
            'save_dir': ct_save_dir, 'n_samples': len(ct_indices)
        }

        # Push Stage 2 to HF Hub
        if hf_token and hf_repo_id:
            s2_repo = hf_repo_id + f"-llama-hierarchical-{safe_name}"
            print(f"Pushing Stage 2 ({cancer_type[:25]}) to HF Hub: {s2_repo}...")
            ct_model.base_model.push_to_hub(s2_repo, token=hf_token)
            tokenizer.push_to_hub(s2_repo, token=hf_token)

        # Free GPU memory between cancer types
        del ct_model, ct_optimizer
        torch.cuda.empty_cache()

    # ===================================================================
    # EXTRACT EMBEDDINGS
    # ===================================================================
    print(f"\n{'='*60}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'='*60}")

    # Stage 1 model for non-viable types
    model = load_best_model(stage1_dir, model_name, device)
    model.eval()

    extract_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    embs, risks, pids, cts = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(extract_loader, desc="Extracting (Stage 1)"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            rs, emb = model(ids, mask)
            embs.append(emb.to(torch.float32).cpu())
            risks.extend(rs.cpu().numpy()); pids.extend(batch['patient_id'])
            cts.extend(batch['cancer_type'])

    all_embs = torch.cat(embs, dim=0).numpy()
    base_df = pd.DataFrame(all_embs, columns=[f'llama_emb_{i}' for i in range(all_embs.shape[1])])
    base_df['risk_score'] = risks; base_df['TCGA_Barcode'] = pids
    base_df['cancer_type'] = cts; base_df['model_used'] = 'stage1_pancancer'

    del model; torch.cuda.empty_cache()

    # Override with Stage 2 models for viable types
    for cancer_type, info in stage2_results.items():
        ct_model = load_best_model(info['save_dir'], model_name, device)
        ct_model.eval()

        ct_mask = base_df['cancer_type'] == cancer_type
        ct_indices = [i for i in range(len(full_dataset)) if full_dataset.df.loc[i, 'DISEASE_TYPE'] == cancer_type]
        ct_loader = DataLoader(Subset(full_dataset, ct_indices), batch_size=batch_size, shuffle=False)

        ct_embs, ct_risks = [], []
        with torch.no_grad():
            for batch in tqdm(ct_loader, desc=f"Extracting ({cancer_type[:25]})"):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                rs, emb = ct_model(ids, mask)
                ct_embs.append(emb.to(torch.float32).cpu())
                ct_risks.extend(rs.cpu().numpy())

        ct_arr = torch.cat(ct_embs, dim=0).numpy()
        emb_cols = [f'llama_emb_{i}' for i in range(ct_arr.shape[1])]
        base_df.loc[ct_mask, emb_cols] = ct_arr
        base_df.loc[ct_mask, 'risk_score'] = ct_risks
        safe = cancer_type.replace(' ', '_').replace(',', '').lower()[:40]
        base_df.loc[ct_mask, 'model_used'] = f'stage2_{safe}'

        del ct_model; torch.cuda.empty_cache()

    out_path = os.path.join(output_dir, 'finetuned_llama_hierarchical_embeddings.csv')
    base_df.to_csv(out_path, index=False)

    # --- Stage 2 comparison plot ---
    if stage2_results:
        n_plots = len(stage2_results)
        cols = min(3, n_plots); rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_plots == 1: axes = [axes]
        else: axes = axes.flatten()
        for i, (ct, info) in enumerate(stage2_results.items()):
            ax = axes[i]
            ep = range(1, len(info['train_losses']) + 1)
            ax.plot(ep, info['train_losses'], marker='o', label='Train', linewidth=1.5)
            ax.plot(ep, info['val_losses'], marker='s', label='Val', linewidth=1.5)
            ax.axvline(x=info['best_epoch'], color='green', linestyle='--', alpha=0.7)
            ax.set_title(f"{ct[:30]}\n(n={info['n_samples']})", fontsize=10)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle('Stage 2: Per-Cancer OpenBioLLM-8B', fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'llama_hierarchical_stage2.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Comparison CSV
    comp_rows = [{'cancer_type': 'ALL (Stage 1)', 'best_epoch': s1_best,
                  'best_val_loss': min(s1_val), 'n_samples': total, 'stage': 'stage1'}]
    for ct, info in stage2_results.items():
        comp_rows.append({
            'cancer_type': ct, 'best_epoch': info['best_epoch'],
            'best_val_loss': info['best_val_loss'], 'n_samples': info['n_samples'], 'stage': 'stage2'
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_path = os.path.join(output_dir, 'llama_hierarchical_comparison.csv')
    comp_df.to_csv(comp_path, index=False)

    if use_wandb:
        import wandb; wandb.finish()

    print(f"\n{'='*60}")
    print("SUMMARY — Strategy 3: Hierarchical Two-Stage (OpenBioLLM-8B)")
    print(f"{'='*60}")
    print(f"  Stage 1 best epoch: {s1_best}")
    print(f"  Stage 1 best val:   {min(s1_val):.4f}")
    print(f"  Stage 2 types:      {len(stage2_results)}")
    for ct, info in stage2_results.items():
        print(f"    {ct[:35]:35s} ep={info['best_epoch']}  val={info['best_val_loss']:.4f}")
    print(f"  Embeddings: {out_path}")
    print(f"  Comparison: {comp_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    load_dotenv(override=True)
    run_hierarchical(
        data_path='data/processed/merged_tcga_data_final.csv',
        output_dir='data/processed',
        stage1_epochs=20, stage2_epochs=10,
        batch_size=4, lr=2e-5, stage2_lr=1e-5,
        patience=3, val_split=0.15,
        hf_token=os.environ.get("HF_TOKEN"),
        hf_repo_id=os.environ.get("HF_REPO_ID"),
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        wandb_project=os.environ.get("WANDB_PROJECT"),
    )
