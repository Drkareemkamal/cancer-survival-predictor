"""
Diagnose data quality issues in merged_tcga_data_text_dedup.csv
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
csv_path = "data/processed/merged_tcga_data_text_dedup.csv"
logger.info(f"Loading data from {csv_path}")

df = pd.read_csv(csv_path, low_memory=False)
logger.info(f"Total rows: {len(df)}")

# Check required columns
required_cols = ['text', 'DISEASE_TYPE', 'AJCC_PATHOLOGIC_TUMOR_STAGE',
                 'OS_MONTHS', 'OS_STATUS']

print("\n" + "="*80)
print("DATA QUALITY DIAGNOSIS")
print("="*80)

# Check if columns exist
print("\n1. Column Existence:")
for col in required_cols:
    exists = col in df.columns
    print(f"   {col}: {'✓' if exists else '✗'}")

# Check missing values
print("\n2. Missing Values (NaN counts):")
for col in required_cols:
    if col in df.columns:
        nan_count = df[col].isna().sum()
        pct = (nan_count / len(df)) * 100
        print(f"   {col}: {nan_count}/{len(df)} ({pct:.1f}%)")

# Check text column
print("\n3. Text Column Analysis:")
if 'text' in df.columns:
    text_not_null = df['text'].notna().sum()
    text_empty = (df['text'].str.len() < 50).sum()
    print(f"   Non-null: {text_not_null}/{len(df)}")
    print(f"   Short (<50 chars): {text_empty}/{len(df)}")
    print(f"   Usable: {text_not_null - text_empty}/{len(df)}")

# Sample checking dropna
print("\n4. After dropna() on required columns:")
df_dropped = df.dropna(subset=required_cols)
print(f"   Remaining samples: {len(df_dropped)}/{len(df)}")

# More lenient checking - only text + disease + survival
print("\n5. More Lenient (text + DISEASE_TYPE + OS_MONTHS + OS_STATUS):")
lenient_cols = ['text', 'DISEASE_TYPE', 'OS_MONTHS', 'OS_STATUS']
df_lenient = df.dropna(subset=lenient_cols)
df_lenient = df_lenient[df_lenient['text'].str.len() > 50]
print(f"   Remaining samples: {len(df_lenient)}/{len(df)}")

# Check disease type distribution
if len(df_lenient) > 0:
    print("\n6. Disease Type Distribution (in lenient set):")
    disease_counts = df_lenient['DISEASE_TYPE'].value_counts()
    for disease, count in disease_counts.head(10).items():
        print(f"   {disease}: {count}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)

if len(df_dropped) > 0:
    print(f"✓ Use strict mode: {len(df_dropped)} samples available")
elif len(df_lenient) > 0:
    print(f"✗ Strict mode fails, but lenient mode works: {len(df_lenient)} samples")
    print("→ Modify generate_instruction_tuning_data.py to be more lenient")
else:
    print("✗ Critical issue: Almost no usable data")
    print("→ Check data file integrity")

print("="*80 + "\n")
