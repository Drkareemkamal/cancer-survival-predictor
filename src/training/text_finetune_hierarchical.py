"""
Strategy 3: Hierarchical Two-Stage Fine-Tuning (ClinicalBERT) — Enhanced
=========================================================================
Stage 1: Pan-cancer fine-tuning on ALL samples.
Stage 2: Per-cancer-type fine-tuning from the Stage 1 checkpoint,
         only for cancer types with enough samples (500+ and 5%+ event rate).
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import transformers
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines.utils import concordance_index

if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from peft import LoraConfig, get_peft_model
from tqdm import tqdm

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
                task_type="FEATURE_EXTRACTION", r=16, lora_alpha=32,
                target_modules=["query", "key", "value", "dense"], lora_dropout=0.2,
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
        hidden_size = self.base_model.config.hidden_size
        self.risk_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        risk_score = self.risk_head(pooled)
        return risk_score.squeeze(-1), pooled


def cox_ph_loss(log_h, events, durations):
    idx = torch.argsort(durations, descending=True)
    events = events[idx]; log_h = log_h[idx]
    log_h_max = torch.max(log_h)
    risk_set_sums = torch.cumsum(torch.exp(log_h - log_h_max), dim=0)
    log_risk_set_sums = torch.log(risk_set_sums) + log_h_max
    loss = -(log_h - log_risk_set_sums) * events
    return loss.sum() / (events.sum() + 1e-8)


def compute_cindex(risk_scores, durations, events):
    try:
        return concordance_index(durations, -np.array(risk_scores), events)
    except Exception:
        return 0.5


def stratified_split_indices(df, val_split=0.15, seed=42):
    event_col = df['OS_STATUS'].astype(str).str.contains('DECEASED').astype(int)
    if 'DISEASE_TYPE' in df.columns:
        strat_labels = df['DISEASE_TYPE'].astype(str) + '_' + event_col.astype(str)
    else:
        strat_labels = event_col
    min_class_count = strat_labels.value_counts().min()
    if min_class_count < 2:
        strat_labels = event_col

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(df)), strat_labels))
    return train_idx.tolist(), val_idx.tolist()


def train_loop(model, train_loader, val_loader, device, optimizer, scheduler,
               max_epochs, patience, best_model_path, stage_name,
               gradient_accumulation_steps=4,
               use_wandb=False, wandb_prefix=""):
    best_val_cindex = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    train_losses, val_losses, val_cindexes = [], [], []

    for epoch in range(max_epochs):
        model.train()
        total_train, n_train = 0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}/{max_epochs} [Train]")
        for step, batch in enumerate(pbar):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            dur = batch['duration'].to(device)
            ev = batch['event'].to(device)
            rs, _ = model(ids, mask)
            loss = cox_ph_loss(rs, ev, dur)
            if torch.isnan(loss):
                continue
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train += loss.item() * gradient_accumulation_steps
            n_train += 1
            pbar.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"})
            if use_wandb:
                import wandb
                wandb.log({f"{wandb_prefix}train/batch_loss": loss.item() * gradient_accumulation_steps})

        avg_train = total_train / max(n_train, 1)
        train_losses.append(avg_train)

        model.eval()
        total_val, n_val = 0, 0
        all_rs, all_dur, all_ev = [], [], []
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
                all_rs.extend(rs.cpu().numpy())
                all_dur.extend(dur.cpu().numpy())
                all_ev.extend(ev.cpu().numpy())

        avg_val = total_val / max(n_val, 1)
        val_losses.append(avg_val)

        val_cindex = compute_cindex(all_rs, all_dur, all_ev)
        val_cindexes.append(val_cindex)

        print(f"  {stage_name} Epoch {epoch+1}: train={avg_train:.4f} val={avg_val:.4f} cindex={val_cindex:.4f}")

        if use_wandb:
            import wandb
            wandb.log({f"{wandb_prefix}train/epoch_loss": avg_train,
                       f"{wandb_prefix}val/epoch_loss": avg_val,
                       f"{wandb_prefix}val/c_index": val_cindex,
                       f"{wandb_prefix}epoch": epoch + 1})

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex; best_epoch = epoch + 1; epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    >>> Best model saved (val_cindex={best_val_cindex:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    --- No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"    *** Early stopping! Best epoch was {best_epoch} ***")
                break

    return train_losses, val_losses, val_cindexes, best_epoch


def run_hierarchical(data_path, output_dir,
                     stage1_epochs=30, stage2_epochs=15,
                     batch_size=8, lr=2e-5, stage2_lr=1e-5,
                     patience=5, val_split=0.15,
                     gradient_accumulation_steps=4,
                     hf_token=None, hf_repo_id=None,
                     wandb_api_key=None, wandb_project=None):

    use_wandb = wandb_project and wandb_api_key
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    if use_wandb:
        import wandb
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, name="ClinicalBERT-Hierarchical-Enhanced", config={
            "strategy": "hierarchical_two_stage",
            "stage1_epochs": stage1_epochs, "stage2_epochs": stage2_epochs,
            "batch_size": batch_size, "effective_batch_size": batch_size * gradient_accumulation_steps,
            "lr": lr, "stage2_lr": stage2_lr,
            "lora_r": 16, "lora_targets": "query,key,value,dense",
            "pooling": "mean", "risk_head": "MLP(768->256->64->1)",
            "scheduler": "cosine_with_warmup", "gradient_clip": 1.0,
            "early_stop_metric": "c-index",
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

    train_idx, val_idx = stratified_split_indices(full_dataset.df, val_split=val_split)
    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    print(f"Pan-cancer split: {len(train_ds)} train / {len(val_ds)} val (stratified)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SurvivalTextModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = (len(train_loader) // gradient_accumulation_steps) * stage1_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)

    stage1_model_path = os.path.join(output_dir, 'clinicalbert_stage1_pancancer.pt')
    s1_train, s1_val, s1_ci, s1_best = train_loop(
        model, train_loader, val_loader, device, optimizer, scheduler,
        stage1_epochs, patience, stage1_model_path,
        stage_name="[Stage1]", gradient_accumulation_steps=gradient_accumulation_steps,
        use_wandb=use_wandb, wandb_prefix="stage1/"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ep = range(1, len(s1_train) + 1)
    ax1.plot(ep, s1_train, marker='o', label='Train'); ax1.plot(ep, s1_val, marker='s', label='Val')
    ax1.axvline(x=s1_best, color='green', linestyle='--', label=f'Best ({s1_best})')
    ax1.set_title('Stage 1: Pan-Cancer Loss', fontsize=14)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cox PH Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(ep, s1_ci, marker='D', label='Val C-index', color='purple')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax2.axvline(x=s1_best, color='green', linestyle='--', label=f'Best ({s1_best})')
    ax2.set_title('Stage 1: Val C-index', fontsize=14)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('C-index')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'hierarchical_stage1_loss.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

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

        ct_indices = [i for i in range(len(full_dataset)) if full_dataset.df.loc[i, 'DISEASE_TYPE'] == cancer_type]
        ct_dataset = Subset(full_dataset, ct_indices)

        ct_val_n = max(int(len(ct_indices) * val_split), 1)
        ct_train_n = len(ct_indices) - ct_val_n

        from torch.utils.data import random_split
        ct_train, ct_val = random_split(ct_dataset, [ct_train_n, ct_val_n],
                                        generator=torch.Generator().manual_seed(42))

        ct_train_loader = DataLoader(ct_train, batch_size=batch_size, shuffle=True)
        ct_val_loader = DataLoader(ct_val, batch_size=batch_size, shuffle=False)

        ct_model = SurvivalTextModel().to(device)
        ct_model.load_state_dict(torch.load(stage1_model_path, map_location=device))
        ct_optimizer = torch.optim.AdamW(ct_model.parameters(), lr=stage2_lr, weight_decay=0.01)

        ct_total_steps = (len(ct_train_loader) // gradient_accumulation_steps) * stage2_epochs
        ct_warmup = int(ct_total_steps * 0.1)
        ct_scheduler = get_cosine_schedule_with_warmup(ct_optimizer, num_warmup_steps=ct_warmup,
                                                        num_training_steps=ct_total_steps)

        safe_name = cancer_type.replace(' ', '_').replace(',', '').lower()[:40]
        ct_model_path = os.path.join(output_dir, f'clinicalbert_stage2_{safe_name}.pt')

        s2_train, s2_val, s2_ci, s2_best = train_loop(
            ct_model, ct_train_loader, ct_val_loader, device, ct_optimizer, ct_scheduler,
            stage2_epochs, patience, ct_model_path,
            stage_name=f"[{cancer_type[:20]}]",
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_wandb=use_wandb, wandb_prefix=f"stage2_{safe_name}/"
        )

        stage2_results[cancer_type] = {
            'train_losses': s2_train, 'val_losses': s2_val, 'val_cindexes': s2_ci,
            'best_epoch': s2_best, 'best_val_cindex': max(s2_ci),
            'model_path': ct_model_path, 'n_samples': len(ct_indices)
        }

        if hf_token and hf_repo_id:
            s2_repo = hf_repo_id + f"-hierarchical-{safe_name}"
            print(f"Pushing Stage 2 ({cancer_type[:25]}) to HF Hub: {s2_repo}...")
            ct_model.load_state_dict(torch.load(ct_model_path, map_location=device))
            ct_model.base_model.push_to_hub(s2_repo, token=hf_token)
            tokenizer.push_to_hub(s2_repo, token=hf_token)

    # ===================================================================
    # EXTRACT EMBEDDINGS
    # ===================================================================
    print(f"\n{'='*60}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(stage1_model_path, map_location=device))
    model.eval()

    extract_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

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
            ax.set_title(f"{ct[:30]}\n(n={info['n_samples']}, best_ci={info['best_val_cindex']:.3f})", fontsize=10)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle('Stage 2: Per-Cancer-Type Fine-Tuning Results', fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'hierarchical_stage2_losses.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    comparison_rows = [{'cancer_type': 'ALL (Stage 1)', 'best_epoch': s1_best,
                        'best_val_cindex': max(s1_ci), 'n_samples': len(full_dataset), 'stage': 'stage1'}]
    for ct, info in stage2_results.items():
        comparison_rows.append({
            'cancer_type': ct, 'best_epoch': info['best_epoch'],
            'best_val_cindex': info['best_val_cindex'], 'n_samples': info['n_samples'], 'stage': 'stage2'
        })
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(output_dir, 'hierarchical_comparison.csv')
    comp_df.to_csv(comp_path, index=False)

    if use_wandb:
        import wandb; wandb.finish()

    print(f"\n{'='*60}")
    print("SUMMARY — Strategy 3: Hierarchical Two-Stage (ClinicalBERT) Enhanced")
    print(f"{'='*60}")
    print(f"  Stage 1 best epoch:   {s1_best}")
    print(f"  Stage 1 best C-index: {max(s1_ci):.4f}")
    print(f"  Stage 2 cancer types: {len(stage2_results)}")
    for ct, info in stage2_results.items():
        print(f"    {ct[:35]:35s} epoch={info['best_epoch']}  cindex={info['best_val_cindex']:.4f}")
    print(f"  Embeddings CSV:  {out_path}")
    print(f"  Comparison CSV:  {comp_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    run_hierarchical(
        data_path='data/processed/merged_tcga_data_text_dedup.csv',
        output_dir='data/processed',
        stage1_epochs=30, stage2_epochs=15,
        batch_size=8, lr=2e-5, stage2_lr=1e-5,
        patience=5, val_split=0.15,
        gradient_accumulation_steps=4,
        hf_token=os.environ.get("HF_TOKEN"),
        hf_repo_id=os.environ.get("HF_REPO_ID"),
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        wandb_project=os.environ.get("WANDB_PROJECT"),
    )
