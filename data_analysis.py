import pandas as pd
import numpy as np
import os
from collections import Counter
import re

# Optional imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Note: PIL not available, skipping detailed image analysis")

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# Load the data
print("="*80)
print("INDIANA X-RAY DATASET ANALYSIS")
print("="*80)

# Load reports CSV
reports_df = pd.read_csv('/Users/ochoukrallah/Documents/DL_GT/Final_project_DL/data/indiana_reports.csv')
print(f"\n✓ Loaded indiana_reports.csv: {len(reports_df)} reports")

# Load projections CSV
projections_df = pd.read_csv('/Users/ochoukrallah/Documents/DL_GT/Final_project_DL/data/indiana_projections.csv')
print(f"✓ Loaded indiana_projections.csv: {len(projections_df)} image records")

print("\n" + "="*80)
print("1. REPORTS CSV STRUCTURE")
print("="*80)
print(f"\nColumns: {list(reports_df.columns)}")
print(f"\nDataset shape: {reports_df.shape}")
print(f"\nFirst 3 rows:")
print(reports_df.head(3))

print("\n" + "="*80)
print("2. COLUMN-BY-COLUMN ANALYSIS")
print("="*80)

for col in reports_df.columns:
    print(f"\n--- {col.upper()} ---")
    non_null = reports_df[col].notna().sum()
    null_count = reports_df[col].isna().sum()
    print(f"  Non-null entries: {non_null}/{len(reports_df)} ({100*non_null/len(reports_df):.1f}%)")
    print(f"  Null entries: {null_count}/{len(reports_df)} ({100*null_count/len(reports_df):.1f}%)")

    if col in ['findings', 'impression', 'indication', 'comparison']:
        # Text columns - analyze length
        texts = reports_df[col].dropna()
        if len(texts) > 0:
            lengths = texts.apply(lambda x: len(str(x).split()))
            print(f"  Text length (words): mean={lengths.mean():.1f}, median={lengths.median():.1f}, min={lengths.min()}, max={lengths.max()}")
            print(f"  Sample: {texts.iloc[0][:150]}...")

print("\n" + "="*80)
print("3. PROJECTIONS CSV ANALYSIS")
print("="*80)
print(f"\nColumns: {list(projections_df.columns)}")
print(f"\nDataset shape: {projections_df.shape}")
print(f"\nFirst 5 rows:")
print(projections_df.head(5))

# Count images per report
images_per_report = projections_df.groupby('uid').size()
print(f"\n--- Images per report ---")
print(f"  Mean: {images_per_report.mean():.2f}")
print(f"  Median: {images_per_report.median():.0f}")
print(f"  Min: {images_per_report.min()}")
print(f"  Max: {images_per_report.max()}")
print(f"\n  Distribution:")
for count, freq in sorted(Counter(images_per_report).items()):
    print(f"    {count} image(s): {freq} reports ({100*freq/len(images_per_report):.1f}%)")

# Projection types
print(f"\n--- Projection types ---")
projection_counts = projections_df['projection'].value_counts()
for proj, count in projection_counts.items():
    print(f"  {proj}: {count} ({100*count/len(projections_df):.1f}%)")

print("\n" + "="*80)
print("4. TEXT FIELDS COMPARISON")
print("="*80)

text_fields = ['indication', 'comparison', 'findings', 'impression']
print(f"\n{'Field':<15} {'Filled %':<12} {'Avg Words':<12} {'Median Words':<12} {'Max Words':<12}")
print("-" * 80)

for field in text_fields:
    filled_pct = 100 * reports_df[field].notna().sum() / len(reports_df)
    texts = reports_df[field].dropna()
    if len(texts) > 0:
        word_counts = texts.apply(lambda x: len(str(x).split()))
        avg_words = word_counts.mean()
        median_words = word_counts.median()
        max_words = word_counts.max()
        print(f"{field:<15} {filled_pct:>10.1f}% {avg_words:>11.1f} {median_words:>14.0f} {max_words:>14}")
    else:
        print(f"{field:<15} {filled_pct:>10.1f}% {'N/A':<11} {'N/A':<14} {'N/A':<14}")

print("\n" + "="*80)
print("5. IMAGE FILES ANALYSIS")
print("="*80)

image_dir = '/Users/ochoukrallah/Documents/DL_GT/Final_project_DL/data/images/images_normalized'
if os.path.exists(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    print(f"\n✓ Found {len(image_files)} PNG images")

    # Sample a few images to check properties
    if PIL_AVAILABLE:
        sample_size = min(50, len(image_files))
        print(f"\nAnalyzing {sample_size} sample images...")

        widths, heights, modes = [], [], []
        for img_file in image_files[:sample_size]:
            try:
                img_path = os.path.join(image_dir, img_file)
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    modes.append(img.mode)
            except Exception as e:
                print(f"  Error reading {img_file}: {e}")

        if widths:
            print(f"\n--- Image properties (from {len(widths)} samples) ---")
            print(f"  Width: mean={np.mean(widths):.0f}, min={min(widths)}, max={max(widths)}")
            print(f"  Height: mean={np.mean(heights):.0f}, min={min(heights)}, max={max(heights)}")
            print(f"  Modes: {Counter(modes)}")
    else:
        print("\n  (Skipping detailed image analysis - PIL not available)")
else:
    print(f"\n✗ Image directory not found: {image_dir}")

print("\n" + "="*80)
print("6. RECOMMENDATIONS FOR CAPTIONING")
print("="*80)

print("""
Based on this analysis, here are the recommended text fields for image captioning:

1. FINDINGS (Primary recommendation):
   - Most comprehensive field (detailed radiological description)
   - High fill rate
   - Appropriate length for captioning task (avg ~60-100 words)
   - Contains the actual description of what's visible in the image

2. IMPRESSION (Secondary/Alternative):
   - More concise summary/conclusion
   - Shorter text (useful for initial experiments)
   - Clinical interpretation rather than pure description

3. COMBINED APPROACH (Recommended for best results):
   - Concatenate: FINDINGS + IMPRESSION
   - Provides both detailed description and clinical summary
   - More complete information about the image

4. NOT RECOMMENDED:
   - INDICATION: Clinical question, not image description
   - COMPARISON: References previous exams, not this image
   - MeSH/Problems: Labels, not natural language descriptions

SUGGESTED STRATEGY:
- Start with FINDINGS only (easier, more focused)
- Later experiment with FINDINGS + IMPRESSION concatenated
- Use IMPRESSION alone for short-caption experiments
""")

print("\n" + "="*80)
print("7. DATA QUALITY CHECKS")
print("="*80)

# Check for missing data
print("\n--- Missing data summary ---")
print(f"Reports with NO findings: {reports_df['findings'].isna().sum()}")
print(f"Reports with NO impression: {reports_df['impression'].isna().sum()}")
print(f"Reports with NEITHER findings nor impression: {(reports_df['findings'].isna() & reports_df['impression'].isna()).sum()}")

# Check image-report matching
print(f"\n--- Image-Report matching ---")
report_uids = set(reports_df['uid'].unique())
projection_uids = set(projections_df['uid'].unique())
print(f"Unique UIDs in reports: {len(report_uids)}")
print(f"Unique UIDs in projections: {len(projection_uids)}")
print(f"UIDs in projections but not in reports: {len(projection_uids - report_uids)}")
print(f"UIDs in reports but not in projections: {len(report_uids - projection_uids)}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
