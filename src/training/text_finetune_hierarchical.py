"""
Strategy 3: Hierarchical Two-Stage Fine-Tuning (ClinicalBERT)
=============================================================
Stage 1: Pan-cancer fine-tuning on ALL samples.
Stage 2: Per-cancer-type fine-tuning from the Stage 1 checkpoint,
         only for cancer types with enough samples (500+ and 5%+ event rate).
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import transformers
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np

# Bypass the strict torch >= 2.6 check
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Minimum requirements for a cancer type to get its own Stage 2 fine-tune
MIN_SAMPLES = 500
MIN_EVENT_RATE = 0.05


class PathologySurvivalDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.dropna(subset=['text', 'OS_MONTHS', 'OS_STATUS']).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

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


class SurvivalTextModel(torch.nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", use_lora=True):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        if use_lora:
            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION", r=8, lora_alpha=32,
                target_modules=["query", "value"], lora_dropout=0.1,
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
        self.risk_head = torch.nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        risk_score = self.risk_head(cls_embedding)
        return risk_score.squeeze(-1), cls_embedding


def cox_ph_loss(log_h, events, durations):
    idx = torch.argsort(durations, descending=True)
    events = events[idx]; log_h = log_h[idx]
    log_h_max = torch.max(log_h)
    risk_set_sums = torch.cumsum(torch.exp(log_h - log_h_max), dim=0)
    log_risk_set_sums = torch.log(risk_set_sums) + log_h_max
    loss = -(log_h - log_risk_set_sums) * events
    return loss.sum() / (events.sum() + 1e-8)


def train_loop(model, train_loader, val_loader, device, optimizer,
               max_epochs, patience, best_model_path, stage_name,
               use_wandb=False, wandb_prefix=""):
    """Reusable training loop with early stopping. Returns train/val losses and best epoch."""
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        total_train, n_train = 0, 0
        pbar = tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}/{max_epochs} [Train]")
        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            dur = batch['duration'].to(device)
            ev = batch['event'].to(device)
            optimizer.zero_grad()
            rs, _ = model(ids, mask)
            loss = cox_ph_loss(rs, ev, dur)
            if torch.isnan(loss):
                continue
            loss.backward(); optimizer.step()
            total_train += loss.item(); n_train += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            if use_wandb:
                import wandb
                wandb.log({f"{wandb_prefix}train/batch_loss": loss.item()})

        avg_train = total_train / max(n_train, 1)
        train_losses.append(avg_train)

        # --- Validate ---
        model.eval()
        total_val, n_val = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"{stage_name} Epoch {epoch+1}/{max_epochs} [Val]", leave=False):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                dur = batch['duration'].to(device)
                ev = batch['event'].to(device)
                rs, _ = model(ids, mask)
                loss = cox_ph_loss(rs, ev, dur)
                if not torch.isnan(loss):
                    total_val += loss.item(); n_val += 1

        avg_val = total_val / max(n_val, 1)
        val_losses.append(avg_val)
        print(f"  {stage_name} Epoch {epoch+1}: train={avg_train:.4f} val={avg_val:.4f}")

        if use_wandb:
            import wandb
            wandb.log({f"{wandb_prefix}train/epoch_loss": avg_train,
                       f"{wandb_prefix}val/epoch_loss": avg_val,
                       f"{wandb_prefix}epoch": epoch + 1})

        if avg_val < best_val_loss:
            best_val_loss = avg_val; best_epoch = epoch + 1; epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    >>> Best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    --- No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"    *** Early stopping! Best epoch was {best_epoch} ***")
                break

    return train_losses, val_losses, best_epoch


def run_hierarchical(data_path, output_dir,
                     stage1_epochs=20, stage2_epochs=10,
                     batch_size=8, lr=1e-4, stage2_lr=5e-5,
                     patience=3, val_split=0.15,
                     hf_token=None, hf_repo_id=None,
                     wandb_api_key=None, wandb_project=None):
    
    use_wandb = wandb_project and wandb_api_key
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    if use_wandb:
        import wandb
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, name="ClinicalBERT-Hierarchical", config={
            "strategy": "hierarchical_two_stage",
            "stage1_epochs": stage1_epochs, "stage2_epochs": stage2_epochs,
            "batch_size": batch_size, "lr": lr, "stage2_lr": stage2_lr,
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    full_dataset = PathologySurvivalDataset(df, tokenizer)

    # ===================================================================
    # STAGE 1: Pan-Cancer Fine-Tuning
    # ===================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: Pan-Cancer Fine-Tuning")
    print(f"{'='*60}")

    total = len(full_dataset)
    val_n = int(total * val_split)
    train_n = total - val_n
    train_ds, val_ds = random_split(full_dataset, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    print(f"Pan-cancer split: {train_n} train / {val_n} val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SurvivalTextModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    stage1_model_path = os.path.join(output_dir, 'clinicalbert_stage1_pancancer.pt')
    s1_train, s1_val, s1_best = train_loop(
        model, train_loader, val_loader, device, optimizer,
        stage1_epochs, patience, stage1_model_path,
        stage_name="[Stage1]", use_wandb=use_wandb, wandb_prefix="stage1/"
    )

    # Plot Stage 1
    fig, ax = plt.subplots(figsize=(12, 7))
    ep = range(1, len(s1_train) + 1)
    ax.plot(ep, s1_train, marker='o', label='Train'); ax.plot(ep, s1_val, marker='s', label='Val')
    ax.axvline(x=s1_best, color='green', linestyle='--', label=f'Best ({s1_best})')
    ax.set_title('Stage 1: Pan-Cancer ClinicalBERT', fontsize=14)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cox PH Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'hierarchical_stage1_loss.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Push Stage 1 model to HF Hub
    if hf_token and hf_repo_id:
        s1_repo = hf_repo_id + "-hierarchical-stage1"
        print(f"Pushing Stage 1 model to HF Hub: {s1_repo}...")
        model.load_state_dict(torch.load(stage1_model_path, map_location=device))
        model.base_model.push_to_hub(s1_repo, token=hf_token)
        tokenizer.push_to_hub(s1_repo, token=hf_token)

    # ===================================================================
    # STAGE 2: Per-Cancer-Type Fine-Tuning
    # ===================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: Per-Cancer-Type Fine-Tuning")
    print(f"{'='*60}")

    # Identify viable cancer types
    clean_df = full_dataset.df
    clean_df['_event'] = clean_df['OS_STATUS'].astype(str).str.contains('DECEASED').astype(int)
    type_stats = clean_df.groupby('DISEASE_TYPE')['_event'].agg(['count', 'sum', 'mean'])
    type_stats.columns = ['total', 'deaths', 'event_rate']
    viable = type_stats[(type_stats['total'] >= MIN_SAMPLES) & (type_stats['event_rate'] >= MIN_EVENT_RATE)]

    print(f"\nViable cancer types for Stage 2 (n>={MIN_SAMPLES}, event_rate>={MIN_EVENT_RATE}):")
    for ct, row in viable.iterrows():
        print(f"  {ct}: {int(row['total'])} samples, {int(row['deaths'])} deaths ({row['event_rate']:.1%})")

    stage2_results = {}

    for cancer_type in viable.index:
        print(f"\n--- Stage 2: {cancer_type} ---")

        # Get indices for this cancer type
        ct_indices = [i for i in range(len(full_dataset)) if full_dataset.df.loc[i, 'DISEASE_TYPE'] == cancer_type]
        ct_dataset = Subset(full_dataset, ct_indices)

        ct_val_n = max(int(len(ct_indices) * val_split), 1)
        ct_train_n = len(ct_indices) - ct_val_n

        ct_train, ct_val = random_split(ct_dataset, [ct_train_n, ct_val_n],
                                        generator=torch.Generator().manual_seed(42))

        ct_train_loader = DataLoader(ct_train, batch_size=batch_size, shuffle=True)
        ct_val_loader = DataLoader(ct_val, batch_size=batch_size, shuffle=False)

        # Load Stage 1 checkpoint as starting point
        ct_model = SurvivalTextModel().to(device)
        ct_model.load_state_dict(torch.load(stage1_model_path, map_location=device))
        ct_optimizer = torch.optim.AdamW(ct_model.parameters(), lr=stage2_lr)

        safe_name = cancer_type.replace(' ', '_').replace(',', '').lower()[:40]
        ct_model_path = os.path.join(output_dir, f'clinicalbert_stage2_{safe_name}.pt')

        s2_train, s2_val, s2_best = train_loop(
            ct_model, ct_train_loader, ct_val_loader, device, ct_optimizer,
            stage2_epochs, patience, ct_model_path,
            stage_name=f"[{cancer_type[:20]}]",
            use_wandb=use_wandb, wandb_prefix=f"stage2_{safe_name}/"
        )

        stage2_results[cancer_type] = {
            'train_losses': s2_train, 'val_losses': s2_val,
            'best_epoch': s2_best, 'best_val_loss': min(s2_val),
            'model_path': ct_model_path, 'n_samples': len(ct_indices)
        }

        # Push Stage 2 model to HF Hub
        if hf_token and hf_repo_id:
            s2_repo = hf_repo_id + f"-hierarchical-{safe_name}"
            print(f"Pushing Stage 2 ({cancer_type[:25]}) to HF Hub: {s2_repo}...")
            ct_model.load_state_dict(torch.load(ct_model_path, map_location=device))
            ct_model.base_model.push_to_hub(s2_repo, token=hf_token)
            tokenizer.push_to_hub(s2_repo, token=hf_token)

    # ===================================================================
    # EXTRACT EMBEDDINGS (using best model per cancer type)
    # ===================================================================
    print(f"\n{'='*60}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'='*60}")

    # Load Stage 1 model as the default
    model.load_state_dict(torch.load(stage1_model_path, map_location=device))
    model.eval()

    all_emb_dfs = []
    extract_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # Collect all in one pass with stage1 model (for non-viable types)
    embeddings_list, risk_list, pid_list, ct_list = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(extract_loader, desc="Extracting (Stage 1 / pan-cancer)"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            rs, emb = model(ids, mask)
            embeddings_list.append(emb.cpu())
            risk_list.extend(rs.cpu().numpy())
            pid_list.extend(batch['patient_id'])
            ct_list.extend(batch['cancer_type'])

    stage1_embs = torch.cat(embeddings_list, dim=0).numpy()
    base_df = pd.DataFrame(stage1_embs, columns=[f'text_emb_{i}' for i in range(stage1_embs.shape[1])])
    base_df['risk_score'] = risk_list
    base_df['TCGA_Barcode'] = pid_list
    base_df['cancer_type'] = ct_list
    base_df['model_used'] = 'stage1_pancancer'

    # Now override embeddings for viable cancer types with their Stage 2 models
    for cancer_type, info in stage2_results.items():
        ct_model = SurvivalTextModel().to(device)
        ct_model.load_state_dict(torch.load(info['model_path'], map_location=device))
        ct_model.eval()

        ct_mask = base_df['cancer_type'] == cancer_type
        ct_indices = [i for i in range(len(full_dataset)) if full_dataset.df.loc[i, 'DISEASE_TYPE'] == cancer_type]
        ct_loader = DataLoader(Subset(full_dataset, ct_indices), batch_size=batch_size, shuffle=False)

        embs, risks = [], []
        with torch.no_grad():
            for batch in tqdm(ct_loader, desc=f"Extracting ({cancer_type[:25]})"):
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                rs, emb = ct_model(ids, mask)
                embs.append(emb.cpu()); risks.extend(rs.cpu().numpy())

        ct_embs = torch.cat(embs, dim=0).numpy()
        emb_cols = [f'text_emb_{i}' for i in range(ct_embs.shape[1])]
        base_df.loc[ct_mask, emb_cols] = ct_embs
        base_df.loc[ct_mask, 'risk_score'] = risks
        safe = cancer_type.replace(' ', '_').replace(',', '').lower()[:40]
        base_df.loc[ct_mask, 'model_used'] = f'stage2_{safe}'

    out_path = os.path.join(output_dir, 'finetuned_text_hierarchical_embeddings.csv')
    base_df.to_csv(out_path, index=False)

    # --- Stage 2 comparison plot ---
    if stage2_results:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, (ct, info) in enumerate(stage2_results.items()):
            if i >= len(axes):
                break
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
        fig.suptitle('Stage 2: Per-Cancer-Type Fine-Tuning Results', fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'hierarchical_stage2_losses.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- Save comparison CSV ---
    comparison_rows = [{'cancer_type': 'ALL (Stage 1)', 'best_epoch': s1_best,
                        'best_val_loss': min(s1_val), 'n_samples': total, 'stage': 'stage1'}]
    for ct, info in stage2_results.items():
        comparison_rows.append({
            'cancer_type': ct, 'best_epoch': info['best_epoch'],
            'best_val_loss': info['best_val_loss'], 'n_samples': info['n_samples'], 'stage': 'stage2'
        })
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(output_dir, 'hierarchical_comparison.csv')
    comp_df.to_csv(comp_path, index=False)

    if use_wandb:
        import wandb; wandb.finish()

    print(f"\n{'='*60}")
    print("SUMMARY — Strategy 3: Hierarchical Two-Stage (ClinicalBERT)")
    print(f"{'='*60}")
    print(f"  Stage 1 best epoch:  {s1_best}")
    print(f"  Stage 1 best val:    {min(s1_val):.4f}")
    print(f"  Stage 2 cancer types: {len(stage2_results)}")
    for ct, info in stage2_results.items():
        print(f"    {ct[:35]:35s} epoch={info['best_epoch']}  val={info['best_val_loss']:.4f}")
    print(f"  Embeddings CSV:  {out_path}")
    print(f"  Comparison CSV:  {comp_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    run_hierarchical(
        data_path='data/processed/merged_tcga_data_final.csv',
        output_dir='data/processed',
        stage1_epochs=20, stage2_epochs=10,
        batch_size=8, lr=1e-4, stage2_lr=5e-5,
        patience=3, val_split=0.15,
        hf_token=os.environ.get("HF_TOKEN"),
        hf_repo_id=os.environ.get("HF_REPO_ID"),
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        wandb_project=os.environ.get("WANDB_PROJECT"),
    )
