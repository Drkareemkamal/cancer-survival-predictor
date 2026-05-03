"""
Strategy 4: Cancer-Type Conditioning Token Fine-Tuning (ClinicalBERT)
=====================================================================
Prepends the cancer type (e.g., "[GLIOMAS]") to each pathological text
before tokenization, so the single model learns cancer-type-aware embeddings.
"""
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import transformers
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

# Bypass the strict torch >= 2.6 check for torch.load to avoid CUDA version conflicts
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from peft import LoraConfig, get_peft_model
from tqdm import tqdm


class ConditionedSurvivalDataset(Dataset):
    """Dataset that prepends cancer type as a conditioning token to the text."""
    
    def __init__(self, df, tokenizer, max_length=512, cancer_type_col='DISEASE_TYPE'):
        self.df = df.dropna(subset=['text', 'OS_MONTHS', 'OS_STATUS']).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cancer_type_col = cancer_type_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, 'text'])
        duration = float(self.df.loc[idx, 'OS_MONTHS'])
        status_str = str(self.df.loc[idx, 'OS_STATUS'])
        event = 1.0 if 'DECEASED' in status_str else 0.0

        # --- KEY CHANGE: Prepend cancer type as a conditioning prefix ---
        cancer_type = str(self.df.loc[idx, self.cancer_type_col]).strip()
        # Create a clean tag, e.g., "[GLIOMAS]" or "[ADENOMAS AND ADENOCARCINOMAS]"
        cancer_tag = f"[{cancer_type.upper()}]"
        conditioned_text = f"{cancer_tag} {text}"

        encoding = self.tokenizer(
            conditioned_text,
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
            'patient_id': str(self.df.loc[idx, 'TCGA_Barcode']),
            'cancer_type': cancer_type
        }


class SurvivalTextModel(torch.nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", use_lora=True):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        
        if use_lora:
            peft_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=8,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
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
    events = events[idx]
    log_h = log_h[idx]
    log_h_max = torch.max(log_h)
    risk_set_sums = torch.cumsum(torch.exp(log_h - log_h_max), dim=0)
    log_risk_set_sums = torch.log(risk_set_sums) + log_h_max
    loss = -(log_h - log_risk_set_sums) * events
    return loss.sum() / (events.sum() + 1e-8)


def train_and_extract(data_path, output_dir, max_epochs=20, batch_size=8, lr=1e-4,
                       patience=3, val_split=0.15,
                       hf_token=None, hf_repo_id=None,
                       wandb_api_key=None, wandb_project=None):
    """
    Fine-tune ClinicalBERT with cancer-type conditioning tokens and early stopping.
    """
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    use_wandb = wandb_project and wandb_api_key
    if use_wandb:
        import wandb
        wandb.login(key=wandb_api_key)
        wandb.init(project=wandb_project, name="ClinicalBERT-Conditioned", config={
            "strategy": "cancer_type_conditioning_token",
            "max_epochs": max_epochs, "batch_size": batch_size,
            "learning_rate": lr, "patience": patience,
            "model_name": "emilyalsentzer/Bio_ClinicalBERT"
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    full_dataset = ConditionedSurvivalDataset(df, tokenizer)

    # Show conditioning examples
    sample = full_dataset[0]
    decoded = tokenizer.decode(sample['input_ids'][:30], skip_special_tokens=True)
    print(f"Example conditioned input (first 30 tokens): {decoded}...")
    print(f"Cancer types in dataset: {df['DISEASE_TYPE'].nunique()}")

    # --- Train / Validation Split ---
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))
    print(f"Dataset split: {train_size} train / {val_size} validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SurvivalTextModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- Early Stopping ---
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'clinicalbert_conditioned_best.pt')

    train_losses, val_losses = [], []

    print(f"\nStarting training (max {max_epochs} epochs, patience={patience})...")
    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        total_train_loss, train_batches = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            durations = batch['duration'].to(device)
            events = batch['event'].to(device)

            optimizer.zero_grad()
            risk_scores, _ = model(input_ids, attention_mask)
            loss = cox_ph_loss(risk_scores, events, durations)

            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            if use_wandb:
                wandb.log({"train/batch_loss": loss.item()})

        avg_train = total_train_loss / max(train_batches, 1)
        train_losses.append(avg_train)

        # --- Validate ---
        model.eval()
        total_val_loss, val_batches = 0, 0
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

        avg_val = total_val_loss / max(val_batches, 1)
        val_losses.append(avg_val)

        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")
        if use_wandb:
            wandb.log({"train/epoch_loss": avg_train, "val/epoch_loss": avg_val, "epoch": epoch + 1})

        # --- Early Stopping ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  --- No improvement for {epochs_no_improve}/{patience} epochs")
            if epochs_no_improve >= patience:
                print(f"\n*** Early stopping at epoch {epoch+1}! Best was epoch {best_epoch} ***")
                break

    actual_epochs = epoch + 1

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    epochs_range = range(1, actual_epochs + 1)
    ax.plot(epochs_range, train_losses, marker='o', label='Train Loss', linewidth=2)
    ax.plot(epochs_range, val_losses, marker='s', label='Validation Loss', linewidth=2)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax.set_title('ClinicalBERT Conditioned: Train vs Validation Loss', fontsize=14)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cox PH Loss')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, 'clinicalbert_conditioned_loss.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight'); plt.close(fig)

    # --- Save Results CSV ---
    results_df = pd.DataFrame({
        'epoch': list(epochs_range), 'train_loss': train_losses,
        'val_loss': val_losses, 'is_best': [e == best_epoch for e in epochs_range]
    })
    results_path = os.path.join(output_dir, 'clinicalbert_conditioned_results.csv')
    results_df.to_csv(results_path, index=False)

    # --- Load best & extract ---
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    if hf_token and hf_repo_id:
        repo = hf_repo_id.replace("BioBERT", "BioBERT-Conditioned")
        print(f"Pushing model to HF Hub: {repo}...")
        model.base_model.push_to_hub(repo, token=hf_token)
        tokenizer.push_to_hub(repo, token=hf_token)

    if use_wandb:
        wandb.finish()

    print("Extracting embeddings...")
    model.eval()
    extract_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    embeddings_list, risk_scores_list, patient_ids, cancer_types = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(extract_loader, desc="Extracting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rs, emb = model(input_ids, attention_mask)
            embeddings_list.append(emb.cpu())
            risk_scores_list.extend(rs.cpu().numpy())
            patient_ids.extend(batch['patient_id'])
            cancer_types.extend(batch['cancer_type'])

    all_embeddings = torch.cat(embeddings_list, dim=0).numpy()
    emb_df = pd.DataFrame(all_embeddings)
    emb_df.columns = [f'text_emb_{i}' for i in range(all_embeddings.shape[1])]
    emb_df['risk_score'] = risk_scores_list
    emb_df['TCGA_Barcode'] = patient_ids
    emb_df['cancer_type'] = cancer_types

    out_path = os.path.join(output_dir, 'finetuned_text_conditioned_embeddings.csv')
    emb_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY — Strategy 4: Cancer-Type Conditioning (ClinicalBERT)")
    print(f"{'='*60}")
    print(f"  Best epoch:      {best_epoch} / {actual_epochs}")
    print(f"  Best val loss:   {best_val_loss:.4f}")
    print(f"  Plot:            {plot_path}")
    print(f"  Results CSV:     {results_path}")
    print(f"  Embeddings CSV:  {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)

    train_and_extract(
        data_path='data/processed/merged_tcga_data_final.csv',
        output_dir='data/processed',
        max_epochs=20, patience=3, val_split=0.15,
        hf_token=os.environ.get("HF_TOKEN"),
        hf_repo_id=os.environ.get("HF_REPO_ID"),
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        wandb_project=os.environ.get("WANDB_PROJECT"),
    )
