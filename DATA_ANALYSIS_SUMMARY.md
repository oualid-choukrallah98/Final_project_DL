# Indiana X-Ray Dataset Analysis Summary

**Generated:** 2025-11-26

---

## Dataset Overview

### Files Structure
- **indiana_reports.csv**: 3,851 radiology reports
- **indiana_projections.csv**: 7,466 image records (mapping images to reports)
- **images_normalized/**: 7,470 PNG X-ray images

### Key Statistics
- **Reports**: 3,851 unique radiology reports
- **Images**: 7,470 chest X-ray images
- **Average images per report**: 1.94
- **Image distribution**:
  - 83.4% of reports have 2 images (frontal + lateral)
  - 11.6% have 1 image
  - 4.7% have 3 images
  - Small percentage have 4-5 images

### Projection Types
- **Frontal**: 3,818 images (51.1%)
- **Lateral**: 3,648 images (48.9%)

---

## Column Analysis: indiana_reports.csv

| Column | Description | Fill Rate | Avg Words | Recommended for Captioning? |
|--------|-------------|-----------|-----------|---------------------------|
| **uid** | Unique identifier | 100% | N/A | For indexing only |
| **MeSH** | Medical Subject Headings (structured terms) | 100% | N/A | ❌ Not natural language |
| **Problems** | Clinical diagnoses/problems | 100% | N/A | ❌ Labels, not descriptions |
| **image** | Type of X-ray exam | 100% | N/A | For metadata only |
| **indication** | Why X-ray was ordered | 97.8% | 4.7 | ❌ Clinical question, not image description |
| **comparison** | Reference to previous exams | 69.7% | 2.6 | ❌ Not about current image |
| **findings** | Detailed radiological description | 86.7% | 31.5 | ✅ **PRIMARY CHOICE** |
| **impression** | Summary/clinical interpretation | 99.2% | 10.6 | ✅ **SECONDARY CHOICE** |

---

## Text Field Examples

### Example 1: Normal Case
**UID**: 1
**Problems**: normal

**FINDINGS** (34 words):
> The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax.

**IMPRESSION** (3 words):
> Normal chest x-XXXX.

---

### Example 2: Abnormal Case with Pathology
**UID**: 10
**Problems**: Calcified Granuloma

**FINDINGS** (38 words):
> The cardiomediastinal silhouette is within normal limits for size and contour. The lungs are normally inflated without evidence of focal airspace disease, pleural effusion, or pneumothorax. Stable calcified granuloma within the right upper lung. No acute bone abnormality.

**IMPRESSION** (4 words):
> No acute cardiopulmonary process.

---

### Example 3: Complex Case (169 words in findings)
**UID**: 206
**Problems**: Density;Cardiomegaly;Aorta;Arthritis;Nodule

**FINDINGS** (169 words):
> Chest Comparison: There is a 2.6 cm diameter masslike density over the lingula partial obscuration left cardiac XXXX. There may be some ill-defined opacity in the right mid and lower lung zone. No pleural effusion is seen. The heart is borderline enlarged. The aorta is dilated and tortuous. Arthritic changes of the spine are present...

**IMPRESSION** (61 words):
> Chest. 1. Left lower lobe nodule which is worrisome. If there are no prior films available for comparison XXXX scan for further evaluation. Pelvis and left hip. Rotated subcapital fracture left hip.

---

## Data Quality Assessment

### Completeness
- ✅ All reports have UIDs and match with images (100% match)
- ✅ 99.2% of reports have IMPRESSION
- ⚠️ 86.7% of reports have FINDINGS (514 missing)
- ⚠️ 25 reports missing BOTH findings and impression
- ⚠️ Some text contains "XXXX" placeholders (anonymization)

### Text Length Distribution

**FINDINGS**:
- Mean: 31.5 words
- Median: 29 words
- Range: 7-169 words
- **Suitable for**: Medium-length captions (50-200 tokens with tokenization)

**IMPRESSION**:
- Mean: 10.6 words
- Median: 5 words
- Range: 1-130 words
- **Suitable for**: Short captions or summary task

---

## Recommendations for Image Captioning

### 🏆 Option 1: FINDINGS Only (RECOMMENDED for starting)
**Pros:**
- Most descriptive field (actual description of what's visible)
- Appropriate length (avg 31.5 words → 50-100 tokens)
- Matches Mamba's strength with medium-length sequences
- Clean training objective: describe what you see

**Cons:**
- 13.3% missing values (need to handle or filter)
- Some very long cases (>100 words)

**Use case:** Primary training target for your Mamba decoder

---

### 🥈 Option 2: IMPRESSION Only (Alternative for experiments)
**Pros:**
- Highest fill rate (99.2%)
- Shorter text (easier to train initially)
- Clear clinical conclusions

**Cons:**
- Very short (median 5 words) - may not fully utilize Mamba's capabilities
- More interpretation than pure description
- Less challenging for comparison with Transformer

**Use case:** Quick baseline experiments or short-caption ablation study

---

### 🌟 Option 3: FINDINGS + IMPRESSION (BEST for final model)
**Pros:**
- Most complete information
- Combines detailed description + clinical summary
- Longer sequences (perfect for showcasing Mamba's linear complexity advantage)
- More realistic clinical scenario

**Cons:**
- Still affected by missing FINDINGS (13.3%)
- Longer text (may need careful max_length tuning)

**Use case:** Final model after initial experiments, best for publication

---

## Suggested Implementation Strategy

### Phase 1: Initial Development (Week 2-3)
```python
# Use FINDINGS only
# Filter out reports without FINDINGS
dataset_df = reports_df[reports_df['findings'].notna()]  # 3,337 reports
max_caption_length = 100  # tokens (sufficient for avg 31.5 words)
```

### Phase 2: Full Model (Week 4-5)
```python
# Use FINDINGS + IMPRESSION concatenated
# Create combined caption:
def create_caption(row):
    caption = ""
    if pd.notna(row['findings']):
        caption += str(row['findings']).strip()
    if pd.notna(row['impression']):
        if caption:
            caption += " "
        caption += str(row['impression']).strip()
    return caption

max_caption_length = 200  # tokens (to handle longer combined text)
```

### Phase 3: Ablation Studies (Week 5)
```python
# Compare:
# 1. Short captions (IMPRESSION only)
# 2. Medium captions (FINDINGS only)
# 3. Long captions (FINDINGS + IMPRESSION)
# Hypothesis: Mamba advantage increases with caption length
```

---

## Important Data Preprocessing Notes

### 1. Handle Missing Data
```python
# Option A: Filter (recommended initially)
valid_reports = reports_df[reports_df['findings'].notna()]

# Option B: Use impression as fallback
reports_df['caption'] = reports_df['findings'].fillna(reports_df['impression'])
```

### 2. Text Cleaning Needed
- **XXXX placeholders**: Decide whether to keep, remove, or replace with special token
- **Multiple spaces**: Normalize whitespace
- **Special characters**: Keep medical punctuation

### 3. Image-Report Pairing Strategy
Since most reports have 2 images (frontal + lateral):

**Option A** (Simple - Recommended for Week 1):
- Use only FRONTAL images (3,818 images)
- 1 image → 1 caption relationship
- Simpler data pipeline

**Option B** (Advanced):
- Use both frontal and lateral
- Both views → same caption
- Better utilizes all data
- More complex data loading

---

## Next Steps (Week 1 Tasks from TODO.md)

Based on this analysis, you should:

1. ✅ **Data exploration** - COMPLETE
2. **Create train/val/test split** (70/15/15)
   - Use random seed for reproducibility
   - Stratify by normal/abnormal if possible
3. **Build vocabulary**
   - Use FINDINGS text
   - Consider medical tokenizers (BioBERT/ClinicalBERT)
4. **Create data pipeline**
   - PyTorch Dataset class
   - Handle missing data
   - Text preprocessing
5. **Set up vision encoder**
   - ViT-Base or ResNet-50
   - Freeze parameters

---

## Summary Statistics for Paper/Report

```
Dataset: Indiana University Chest X-Ray Collection
- Reports: 3,851
- Images: 7,470 (3,818 frontal, 3,648 lateral)
- Caption source: Radiologist-written reports
- Caption length: 31.5 ± 24.3 words (FINDINGS)
- Vocabulary: Medical/clinical domain
- Train/Val/Test: 2,696/578/577 reports (70/15/15 split)
```

---

## Questions for Decision

Before proceeding, please decide:

1. **Which caption field(s) to use?**
   - [ ] FINDINGS only (recommended to start)
   - [ ] IMPRESSION only
   - [ ] FINDINGS + IMPRESSION combined

2. **Which images to use?**
   - [ ] Frontal only (simpler)
   - [ ] Both frontal and lateral (more data)

3. **Handle missing FINDINGS?**
   - [ ] Filter out (3,337 reports remaining)
   - [ ] Use IMPRESSION as fallback (3,826 reports)

**My recommendation:** Start with FINDINGS only + Frontal images only for simplicity, then expand later.
