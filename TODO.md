Week 1: Shared Foundation Setup (Collaborative Work)
This week establishes the common infrastructure that both Mamba and Transformer decoders will use. Everything here must be identical for fair comparison.

Day 1-2: Dataset Setup & Exploration
Goal: Download, understand, and prepare the IU X-Ray dataset
Tasks:
1. Download the Dataset

Get IU X-Ray from Kaggle: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
Understand the structure:

Images folder: Contains chest X-ray images (.png files)
Reports: Text files or CSV with radiological descriptions
Typically: ~7,470 images, ~3,955 reports (some reports have 2 views: frontal + lateral)



2. Exploratory Data Analysis (Do Together)

Image analysis:

Image sizes (are they consistent?)
Image formats (grayscale vs RGB)
Quality issues (any corrupted images?)


Text analysis:

Caption length distribution (how many tokens typically?)
Vocabulary size (how many unique words?)
Report structure (do they follow a template? sections like "Findings:", "Impression:"?)
Key finding: Most reports are 50-200 tokens long (this justifies Mamba!)



3. Create Train/Val/Test Split

Split ratios: 70% train, 15% validation, 15% test
Important: Use random seed for reproducibility (e.g., seed=42)
Save split indices to a file so both architectures use exact same splits
Create a JSON file: data_splits.json

  {
    "train": [image_id1, image_id2, ...],
    "val": [image_id3, ...],
    "test": [image_id4, ...]
  }
4. Create Data Statistics Document

Number of samples in each split
Average caption length
Min/max caption length
Vocabulary size
Example images and captions (visualize 5-10)

Deliverable: Everyone has the same dataset, same splits, same understanding

Day 3-4: Data Pipeline Construction
Goal: Build a reusable data loading and preprocessing pipeline
Tasks:
1. Image Preprocessing Module
Create a shared script (e.g., image_preprocessing.py) that both team members will use:
Image Transformations:
For Training:
- Resize to 224×224 (or 384×384 if using ViT-L)
- Convert to RGB (if grayscale, repeat channel 3 times)
- Optional augmentation:
  * Random horizontal flip (careful - medical images!)
  * Random rotation (±5 degrees max)
  * Color jitter (slight brightness/contrast)
- Normalize using ImageNet statistics:
  * Mean: [0.485, 0.456, 0.406]
  * Std: [0.229, 0.224, 0.225]
- Convert to tensor

For Validation/Test:
- Same as training but NO augmentation
- Just resize, RGB, normalize, tensor
Why ImageNet statistics? Because your vision encoder is pre-trained on ImageNet
2. Text Preprocessing Module
Create a shared script (e.g., text_preprocessing.py):
Tokenization Strategy:

Option A (Recommended): Use a medical/clinical tokenizer

BioBERT tokenizer
ClinicalBERT tokenizer
Or BlueBert tokenizer


Option B: Simple word-level tokenization

Build vocabulary from training captions
Minimum frequency threshold (e.g., words appearing <5 times → <UNK>)



Special Tokens to Add:

<PAD>: Padding token (ID: 0)
<BOS>: Beginning of sequence (ID: 1)
<EOS>: End of sequence (ID: 2)
<UNK>: Unknown words (ID: 3)

Caption Processing Steps:
Original: "Heart size is normal. No pneumothorax."

Step 1 - Clean text:
- Lowercase: "heart size is normal. no pneumothorax."
- Remove extra spaces
- Keep punctuation (helps with sentence structure)

Step 2 - Tokenize:
- ["heart", "size", "is", "normal", ".", "no", "pneumothorax", "."]

Step 3 - Add special tokens:
- ["<BOS>", "heart", "size", "is", "normal", ".", "no", "pneumothorax", ".", "<EOS>"]

Step 4 - Convert to IDs:
- [1, 245, 678, 32, 891, 5, 156, 1023, 5, 2]

Step 5 - Pad to max length (e.g., 200):
- [1, 245, 678, 32, 891, 5, 156, 1023, 5, 2, 0, 0, 0, ...]
3. Create Vocabulary File

Build from training set only (never from val/test!)
Save to vocab.json or vocab.pkl
Both team members use the same vocabulary file

4. Build PyTorch Dataset Class
Create medical_caption_dataset.py that both will use:
Class structure:

MedicalCaptionDataset:
    Input:
    - image_dir: path to images
    - captions_file: path to captions
    - split_ids: which IDs to use (train/val/test)
    - transform: image transformations
    - tokenizer: text tokenizer
    - max_length: max caption length
    
    __getitem__(index):
        Returns:
        - image: preprocessed image tensor [3, 224, 224]
        - caption: tokenized caption [max_length]
        - caption_length: actual length (before padding)
        - image_id: for tracking
    
    __len__():
        Returns: number of samples in split
5. Create DataLoader Configuration
Shared config file (data_config.py):
Configuration parameters:
- batch_size: 16 or 32 (depends on GPU memory)
- num_workers: 4 (for parallel data loading)
- shuffle: True for train, False for val/test
- pin_memory: True (faster GPU transfer)
- drop_last: True for train (ensures consistent batch size)
Deliverable: Both team members can load the exact same batches of data

Day 5: Vision Encoder Setup
Goal: Set up the frozen pre-trained vision encoder (this is THE most critical shared component)
Tasks:
1. Choose Vision Encoder
Recommended: Vision Transformer (ViT)

ViT-Base/16: 86M parameters, outputs [batch, 196, 768]

"16" means 16×16 patches
224×224 image → 14×14 = 196 patches


ViT-Large/16: 304M parameters, outputs [batch, 196, 1024]

Alternative: ResNet

ResNet-50: outputs [batch, 2048, 7, 7] → flatten to [batch, 49, 2048]

Decision criteria: ViT usually performs better for captioning, but ResNet is faster
2. Load Pre-trained Weights
Use a model pre-trained on ImageNet:
Sources for pre-trained models:
- torchvision: torchvision.models.vit_b_16(pretrained=True)
- timm library: timm.create_model('vit_base_patch16_224', pretrained=True)
- Hugging Face: transformers.ViTModel.from_pretrained('google/vit-base-patch16-224')
3. Freeze All Parameters
Critical step - must verify this worked!
Steps:
1. Load the pre-trained model
2. Set all parameters to require_grad=False
3. Set model to eval() mode
4. Verify: count trainable parameters → should be 0 for encoder
5. Test: forward pass → verify no gradients flow back to encoder
Why freeze?

Fair comparison: only decoder differences matter
Faster training: fewer parameters to update
Less GPU memory: no gradients stored for encoder
Prevents overfitting: encoder already knows visual features

4. Extract Features Test
Create test script to verify encoder works:
Test procedure:
1. Load a sample X-ray image
2. Preprocess it
3. Pass through encoder
4. Check output shape: should be [1, 196, 768] for ViT-Base
5. Visualize attention maps (optional but cool!)
6. Save sample features to verify consistency
5. Create Encoder Wrapper Module
Create vision_encoder.py that both will use:
Class: FrozenVisionEncoder
    - Loads pre-trained ViT
    - Freezes all parameters
    - Provides clean interface:
      * forward(images) → visual_features
      * get_feature_dim() → 768
      * get_num_patches() → 196
Deliverable: Both team members have the exact same frozen encoder

Day 6: Projection Layer & Integration Test
Goal: Bridge the vision encoder to decoder, test end-to-end pipeline
Tasks:
1. Design Projection Layer
Purpose: Convert encoder features to decoder input dimension
Scenario:
- Encoder outputs: [batch, 196, 768]  (ViT-Base features)
- Your decoder expects: [batch, 196, 512] or [batch, 196, 1024]

Projection layer:
- Simple linear layer: Linear(768 → decoder_dim)
- Optional: add Layer Normalization
- Optional: add Dropout (0.1) for regularization
Both architectures need the same projection!
2. Create Shared Projection Module
Create projection.py:
Class: VisualProjection
    Input:
    - input_dim: 768 (from ViT)
    - output_dim: 512 or 1024 (your decoder dim)
    - dropout: 0.1
    
    Components:
    - Linear layer
    - LayerNorm
    - Dropout
    
    forward(visual_features):
        Returns: projected_features
This will be trainable (unlike the encoder)
3. Integration Test
Create test_pipeline.py to verify everything works together:
Test script should:
1. Load 1 batch of data (images + captions)
2. Pass images through frozen encoder → visual_features
3. Pass visual_features through projection → projected_features
4. Print shapes at each step
5. Verify:
   - Image shape: [batch, 3, 224, 224]
   - Encoder output: [batch, 196, 768]
   - Projected output: [batch, 196, decoder_dim]
   - Caption tokens: [batch, max_length]
6. Check memory usage
7. Time each operation
Run this test on both team members' machines to ensure consistency!
4. Create Shared Configuration File
Create model_config.py with all hyperparameters:
Shared configs (both architectures must use these):
- encoder_name: "vit_base_16"
- encoder_dim: 768
- decoder_dim: 512 (or 1024)
- num_visual_tokens: 196
- vocab_size: (from your vocabulary)
- max_caption_length: 200
- embedding_dim: decoder_dim
- dropout: 0.1

Training configs (both must use same):
- learning_rate: 1e-4
- batch_size: 16
- num_epochs: 30
- warmup_steps: 1000
- weight_decay: 0.01
- optimizer: "AdamW"
- scheduler: "cosine"

Day 7: Documentation & Handoff
Goal: Document everything and enable independent work
Tasks:
1. Create Shared Code Repository Structure
project/
├── data/
│   ├── data_splits.json          # Train/val/test splits (SHARED)
│   ├── vocab.json                 # Vocabulary (SHARED)
│   └── data_statistics.txt        # Dataset stats
├── preprocessing/
│   ├── image_preprocessing.py     # Image transforms (SHARED)
│   └── text_preprocessing.py      # Tokenization (SHARED)
├── dataset/
│   └── medical_caption_dataset.py # Dataset class (SHARED)
├── models/
│   ├── vision_encoder.py          # Frozen ViT (SHARED)
│   ├── projection.py              # Projection layer (SHARED)
│   ├── mamba_decoder.py           # Your work (MAMBA team)
│   └── transformer_decoder.py     # Your teammate's work (TRANSFORMER team)
├── configs/
│   ├── model_config.py            # Model hyperparameters (SHARED)
│   └── data_config.py             # Data loading configs (SHARED)
├── utils/
│   ├── metrics.py                 # Evaluation metrics (SHARED later)
│   └── visualization.py           # Plotting tools (SHARED later)
└── tests/
    └── test_pipeline.py           # Integration test (SHARED)
2. Write README Documentation
Document for your team:
README.md should include:

1. Setup Instructions:
   - Environment setup (Python version, libraries)
   - Dataset download instructions
   - How to run data preprocessing

2. Data Pipeline:
   - How splits were created
   - Data statistics summary
   - How to load data

3. Model Components:
   - Vision encoder details (architecture, where weights from)
   - Projection layer design
   - Expected input/output shapes

4. Testing:
   - How to run integration test
   - Expected outputs
   - How to verify everything works

5. Next Steps:
   - What Mamba team does next
   - What Transformer team does next
   - How/when to sync up
3. Create Checkpoint
Save the frozen encoder + projection layer:
Why:
- Both team members start from exact same point
- Can reload if something breaks
- Ensures no accidental differences

What to save:
- vision_encoder_frozen.pth (or just record which pre-trained model)
- Initial (random) projection layer weights (for reproducibility)
- Random seed used
4. Division of Labor Agreement
Document who does what:
SHARED (Week 1 - Both):
✅ Data pipeline
✅ Frozen vision encoder  
✅ Projection layer
✅ Configuration files
✅ Test scripts

MAMBA TEAM (Week 2-4):
🔵 Implement/integrate Mamba decoder architecture
🔵 Training script for Mamba
🔵 Inference script for Mamba
🔵 Mamba-specific debugging

TRANSFORMER TEAM (Week 2-4):
🟢 Implement Transformer decoder architecture
🟢 Training script for Transformer
🟢 Inference script for Transformer
🟢 Transformer-specific debugging

SHARED AGAIN (Week 5-6):
✅ Evaluation metrics (BLEU, CIDEr, etc.)
✅ Experiment scripts (length sensitivity, speed tests)
✅ Analysis and plotting
✅ Report writing
5. Sync Meeting
End of Week 1 meeting checklist:

 Both can load the same data batches
 Both can run the frozen encoder
 Both can run projection layer
 Both have the same config files
 Integration test passes on both machines
 Code is in shared repository (GitHub)
 Next steps are clear