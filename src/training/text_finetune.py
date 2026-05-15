import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import transformers
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines.utils import concordance_index

if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from peft import LoraConfig, get_peft_model
from tqdm import tqdm

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
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'duration': torch.tensor(duration, dtype=torch.float32),
            'event': torch.tensor(event, dtype=torch.float32),
            'patient_id': str(self.df.loc[idx, 'TCGA_Barcode'])
        }

class SurvivalTextModel(torch.nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", use_lora=True):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

        if use_lora:
            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=16,
                lora_alpha=32,
                target_modules=["query", "key", "value", "dense"],
                lora_dropout=0.2,
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
    events = events[idx]
    log_h = log_h[idx]

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

def stratified_split(dataset, val_split=0.15, seed=42):
    df = dataset.df
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
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())

def train_and_extract(data_path, output_dir, max_epochs=40, batch_size=8, lr=1e-4,
                       patience=7, val_split=0.15,
                       gradient_accumulation_steps=4,
                       hf_token=None, hf_repo_id=None,
                       wandb_api_key=None, wandb_project=None):
    if hf_token:
        from huggingface_hub import login
        print("Logging into Hugging Face Hub...")
        login(token=hf_token)

    if wandb_project and wandb_api_key:
        import wandb
        print("Logging into Weights & Biases...")
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, name="ClinicalBERT-Finetune-CosineRestart", config={
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": lr,
            "patience": patience,
            "val_split": val_split,
            "model_name": "emilyalsentzer/Bio_ClinicalBERT",
            "lora_r": 16,
            "lora_targets": "query,key,value,dense",
            "pooling": "mean",
            "risk_head": "MLP(768->256->64->1)",
            "scheduler": "CosineAnnealingWarmRestarts(T_0=5,T_mult=2)",
            "gradient_clip": 1.0,
            "early_stop_metric": "c-index",
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    full_dataset = PathologySurvivalDataset(df, tokenizer)

    train_dataset, val_dataset = stratified_split(full_dataset, val_split=val_split)
    print(f"Dataset split: {len(train_dataset)} train / {len(val_dataset)} validation samples (stratified)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SurvivalTextModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5 * steps_per_epoch, T_mult=2, eta_min=1e-6
    )

    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  LR schedule: CosineAnnealingWarmRestarts — restart every 5 epochs, then 10, 20...")
    print(f"  LR range: {lr} → 1e-6, restarts back to {lr}")

    best_val_cindex = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'clinicalbert_best_model.pt')

    train_losses = []
    val_losses = []
    val_cindexes = []

    print(f"Starting training (max {max_epochs} epochs, patience={patience}, metric=C-index)...")
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        train_batches = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)

            risk_scores, _ = model(input_ids, attention_mask)
            loss = cox_ph_loss(risk_scores, events, durations)

            if torch.isnan(loss):
                continue

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * gradient_accumulation_steps
            train_batches += 1
            pbar.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"})

            if wandb_project and wandb_api_key:
                wandb.log({"train/batch_loss": loss.item() * gradient_accumulation_steps,
                           "train/lr": scheduler.get_last_lr()[0]})

        avg_train_loss = total_train_loss / max(train_batches, 1)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        val_batches = 0
        all_risk_scores = []
        all_durations = []
        all_events = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                durations = batch['duration'].to(device)
                events = batch['event'].to(device)

                risk_scores, _ = model(input_ids, attention_mask)
                loss = cox_ph_loss(risk_scores, events, durations)

                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    val_batches += 1

                all_risk_scores.extend(risk_scores.cpu().numpy())
                all_durations.extend(durations.cpu().numpy())
                all_events.extend(events.cpu().numpy())

        avg_val_loss = total_val_loss / max(val_batches, 1)
        val_losses.append(avg_val_loss)

        val_cindex = compute_cindex(all_risk_scores, all_durations, all_events)
        val_cindexes.append(val_cindex)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  val_cindex={val_cindex:.4f}")

        if wandb_project and wandb_api_key:
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "val/epoch_loss": avg_val_loss,
                "val/c_index": val_cindex,
                "epoch": epoch + 1
            })

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New best model saved (val_cindex={best_val_cindex:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --- No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"\n*** Early stopping triggered at epoch {epoch+1}! ***")
                print(f"    Best epoch was {best_epoch} with val_cindex={best_val_cindex:.4f}")
                break

    actual_epochs = epoch + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    epochs_range = range(1, actual_epochs + 1)
    ax1.plot(epochs_range, train_losses, marker='o', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, marker='s', label='Validation Loss', linewidth=2)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_title('ClinicalBERT Enhanced: Train vs Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cox PH Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, val_cindexes, marker='D', label='Val C-index', linewidth=2, color='purple')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax2.set_title('Validation C-index per Epoch', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('C-index', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, 'clinicalbert_training_loss.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training loss plot saved to {plot_path}")

    results_df = pd.DataFrame({
        'epoch': list(epochs_range),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_cindex': val_cindexes,
        'is_best': [e == best_epoch for e in epochs_range]
    })
    results_path = os.path.join(output_dir, 'clinicalbert_training_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Training results CSV saved to {results_path}")

    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    if hf_token and hf_repo_id:
        print(f"Pushing fine-tuned model to Hugging Face Hub: {hf_repo_id}...")
        model.base_model.push_to_hub(hf_repo_id, token=hf_token)
        tokenizer.push_to_hub(hf_repo_id, token=hf_token)

        readme_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                   'notebooks', 'README.md')
        if os.path.exists(readme_path):
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=hf_repo_id,
                token=hf_token,
            )
            print(f"Model card (README.md) pushed to {hf_repo_id}")

        print("Model push complete!")

    if wandb_project and wandb_api_key:
        import wandb
        wandb.finish()

    print("Extracting fine-tuned embeddings using best model...")
    model.eval()

    extract_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    embeddings_list = []
    risk_scores_list = []
    patient_ids = []

    with torch.no_grad():
        for batch in tqdm(extract_loader, desc="Extracting embeddings and risk scores"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            risk_scores, cls_embeddings = model(input_ids, attention_mask)

            embeddings_list.append(cls_embeddings.cpu())
            risk_scores_list.extend(risk_scores.cpu().numpy())
            patient_ids.extend(batch['patient_id'])

    all_embeddings = torch.cat(embeddings_list, dim=0).numpy()

    emb_df = pd.DataFrame(all_embeddings)
    emb_df.columns = [f'text_emb_{i}' for i in range(all_embeddings.shape[1])]
    emb_df['risk_score'] = risk_scores_list
    emb_df['TCGA_Barcode'] = patient_ids

    out_path = os.path.join(output_dir, 'finetuned_text_embeddings.csv')
    emb_df.to_csv(out_path, index=False)
    print(f"Embeddings saved to {out_path}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Best epoch:        {best_epoch} / {actual_epochs}")
    print(f"  Best val C-index:  {best_val_cindex:.4f}")
    print(f"  Final train loss:  {train_losses[-1]:.4f}")
    print(f"  Training plot:     {plot_path}")
    print(f"  Results CSV:       {results_path}")
    print(f"  Embeddings CSV:    {out_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    train_and_extract(
        data_path='data/processed/merged_tcga_data_text_dedup.csv',
        output_dir='data/processed',
        max_epochs=40,
        batch_size=8,
        lr=1e-4,
        patience=7,
        val_split=0.15,
        gradient_accumulation_steps=4,
        hf_token=hf_token,
        hf_repo_id=hf_repo_id,
        wandb_api_key=wandb_api_key,
        wandb_project=wandb_project
    )
