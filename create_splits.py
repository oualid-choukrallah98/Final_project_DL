"""
Create train/val/test splits for Indiana X-Ray dataset
Following TODO.md Day 1-2 requirements
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("="*80)
print("CREATING TRAIN/VAL/TEST SPLITS")
print("="*80)

# Load data
reports_df = pd.read_csv('data/indiana_reports.csv')
projections_df = pd.read_csv('data/indiana_projections.csv')

print(f"\nLoaded {len(reports_df)} reports and {len(projections_df)} image records")

# Filter for reports with FINDINGS (as recommended)
valid_reports = reports_df[reports_df['findings'].notna()].copy()
print(f"Reports with FINDINGS: {len(valid_reports)} ({100*len(valid_reports)/len(reports_df):.1f}%)")

# Filter projections to only include frontal images (as recommended)
frontal_projections = projections_df[projections_df['projection'] == 'Frontal'].copy()
print(f"Frontal images: {len(frontal_projections)} ({100*len(frontal_projections)/len(projections_df):.1f}%)")

# Join to get valid image-caption pairs
# Only keep reports that have frontal images AND findings
valid_data = frontal_projections.merge(valid_reports, on='uid', how='inner')
print(f"\nValid image-caption pairs (frontal + findings): {len(valid_data)}")

# Get unique UIDs for splitting
unique_uids = valid_data['uid'].unique()
print(f"Unique reports: {len(unique_uids)}")

# Split: 70% train, 15% val, 15% test
train_uids, temp_uids = train_test_split(
    unique_uids,
    test_size=0.30,
    random_state=RANDOM_SEED
)

val_uids, test_uids = train_test_split(
    temp_uids,
    test_size=0.50,  # 50% of 30% = 15%
    random_state=RANDOM_SEED
)

print("\n" + "="*80)
print("SPLIT STATISTICS")
print("="*80)
print(f"Train: {len(train_uids)} reports ({100*len(train_uids)/len(unique_uids):.1f}%)")
print(f"Val:   {len(val_uids)} reports ({100*len(val_uids)/len(unique_uids):.1f}%)")
print(f"Test:  {len(test_uids)} reports ({100*len(test_uids)/len(unique_uids):.1f}%)")

# Count images per split
train_images = valid_data[valid_data['uid'].isin(train_uids)]
val_images = valid_data[valid_data['uid'].isin(val_uids)]
test_images = valid_data[valid_data['uid'].isin(test_uids)]

print(f"\nTrain images: {len(train_images)}")
print(f"Val images:   {len(val_images)}")
print(f"Test images:  {len(test_images)}")

# Analyze normal vs abnormal distribution
train_normal = train_images[train_images['Problems'] == 'normal']
val_normal = val_images[val_images['Problems'] == 'normal']
test_normal = test_images[test_images['Problems'] == 'normal']

print("\n" + "="*80)
print("NORMAL VS ABNORMAL DISTRIBUTION")
print("="*80)
print(f"Train: {len(train_normal)} normal ({100*len(train_normal)/len(train_images):.1f}%), "
      f"{len(train_images)-len(train_normal)} abnormal ({100*(len(train_images)-len(train_normal))/len(train_images):.1f}%)")
print(f"Val:   {len(val_normal)} normal ({100*len(val_normal)/len(val_images):.1f}%), "
      f"{len(val_images)-len(val_normal)} abnormal ({100*(len(val_images)-len(val_normal))/len(val_images):.1f}%)")
print(f"Test:  {len(test_normal)} normal ({100*len(test_normal)/len(test_images):.1f}%), "
      f"{len(test_images)-len(test_normal)} abnormal ({100*(len(test_images)-len(test_normal))/len(test_images):.1f}%)")

# Save splits to JSON
splits = {
    "train": train_uids.tolist(),
    "val": val_uids.tolist(),
    "test": test_uids.tolist(),
    "metadata": {
        "random_seed": RANDOM_SEED,
        "split_ratios": [0.70, 0.15, 0.15],
        "total_reports": len(unique_uids),
        "image_type": "frontal_only",
        "caption_field": "findings",
        "date_created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
}

with open('data/data_splits.json', 'w') as f:
    json.dump(splits, f, indent=2)

print("\n" + "="*80)
print("SAVED")
print("="*80)
print("✓ Splits saved to: data/data_splits.json")

# Also save a detailed CSV for each split
train_data = valid_data[valid_data['uid'].isin(train_uids)][['uid', 'filename', 'findings', 'impression', 'Problems']]
val_data = valid_data[valid_data['uid'].isin(val_uids)][['uid', 'filename', 'findings', 'impression', 'Problems']]
test_data = valid_data[valid_data['uid'].isin(test_uids)][['uid', 'filename', 'findings', 'impression', 'Problems']]

train_data.to_csv('data/train_data.csv', index=False)
val_data.to_csv('data/val_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

print("✓ Detailed data saved to: data/train_data.csv, data/val_data.csv, data/test_data.csv")

# Generate statistics
print("\n" + "="*80)
print("CAPTION STATISTICS BY SPLIT")
print("="*80)

for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
    word_counts = split_data['findings'].apply(lambda x: len(str(x).split()))
    print(f"\n{split_name}:")
    print(f"  Caption length (words): mean={word_counts.mean():.1f}, "
          f"median={word_counts.median():.0f}, min={word_counts.min()}, max={word_counts.max()}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Build vocabulary from training data")
print("2. Create preprocessing modules")
print("3. Create PyTorch Dataset class")
