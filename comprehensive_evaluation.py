import os
import sys
import json
import time
import gc
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

#import nltk
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

import evaluate
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

from transformers import BlipForConditionalGeneration, BlipProcessor
from peft import PeftModel

from models.mamba2_decoder import MambaDecoder
from preprocessing.text_preprocessing import CaptionTokenizer
from preprocessing.image_preprocessing import get_train_transform, get_val_transform
from dataset.medical_caption_dataset import MedicalCaptionDataset


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    use_rope: bool = False
    max_seq_len: int = 200
    data_percentage: float = 1.0  # 1.0 = 100%, 0.1 = 10%, etc.
    batch_size: int = 4
    epochs: int = 5
    learning_rate: float = 1e-4
    dropout: float = 0.2  # Increased dropout for regularization
    early_stopping_patience: int = 3  # Stop if val loss doesn't improve for 3 epochs
    device: str = "cuda"
    image_size: int = 224
    vocab_path: str = "data/vocab.json"
    train_csv: str = "data/train_data.csv"
    val_csv: str = "data/val_data.csv"
    test_csv: str = "data/test_data.csv"
    image_dir: str = "data/chest-xrays-indiana-university/images/images_normalized"
    output_dir: str = "evaluation_results"
    checkpoint_dir: str = "checkpoints"


class MetricsCalculator:
    
    
    def __init__(self):
        self.bleu_metric = evaluate.load("bleu")
        self.cider_scorer = Cider()

        try:
            self.meteor_scorer = Meteor()
            self.meteor_available = True
        except Exception:
            self.meteor_scorer = None
            self.meteor_available = False
    
    def calculate_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        filtered_preds = []
        filtered_refs = []
        for pred, ref in zip(predictions, references):
            # Skip if either is empty, whitespace-only, or placeholder
            if (pred and ref and pred.strip() and ref.strip() and 
                pred != "<empty>" and ref != "<empty>"):
                filtered_preds.append(pred)
                filtered_refs.append(ref)
        
        if len(filtered_preds) == 0:
            print("  Warning: All predictions/references were empty. Returning zero BLEU scores.")
            return {
                'bleu': 0.0,
                'precisions': [0.0, 0.0, 0.0, 0.0],
                'brevity_penalty': 1.0,
                'length_ratio': 1.0,
                'translation_length': 0,
                'reference_length': 0
            }
        
        refs = [[ref] for ref in filtered_refs]
        try:
            results = self.bleu_metric.compute(
                predictions=filtered_preds,
                references=refs,
                max_order=4
            )
            return results
        except (ZeroDivisionError, ValueError) as e:
            print(f"  Warning: BLEU calculation failed ({e}). Returning zero scores.")
            return {
                'bleu': 0.0,
                'precisions': [0.0, 0.0, 0.0, 0.0],
                'brevity_penalty': 1.0,
                'length_ratio': 1.0,
                'translation_length': len(filtered_preds),
                'reference_length': len(filtered_refs)
            }
    
    def calculate_cider(self, predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
        gts = {str(i): [ref] for i, ref in enumerate(references)}
        res = {str(i): [pred] for i, pred in enumerate(predictions)}
        score, scores = self.cider_scorer.compute_score(gts, res)
        return float(score), [float(s) for s in scores]
    
    def calculate_meteor(self, predictions: List[str], references: List[str]) -> Tuple[float, List[float]]:
        if not self.meteor_available:
            return 0.0, [0.0] * len(predictions)
        
        gts = {str(i): [ref] for i, ref in enumerate(references)}
        res = {str(i): [pred] for i, pred in enumerate(predictions)}
        score, scores = self.meteor_scorer.compute_score(gts, res)
        return float(score), [float(s) for s in scores]
    
    def calculate_radgraph_f1(self, predictions: List[str], references: List[str]) -> Dict:
        def extract_entities(text):
            medical_terms = [
                'pneumothorax', 'effusion', 'consolidation', 'cardiomegaly',
                'atelectasis', 'edema', 'pleural', 'mediastinal', 'pulmonary',
                'cardiac', 'lung', 'heart', 'chest', 'normal', 'abnormal',
                'opacity', 'infiltrate', 'nodule', 'mass', 'fracture'
            ]
            text_lower = text.lower()
            found = [term for term in medical_terms if term in text_lower]
            return set(found)
        
        all_precision = []
        all_recall = []
        all_f1 = []
        
        for pred, ref in zip(predictions, references):
            pred_entities = extract_entities(pred)
            ref_entities = extract_entities(ref)
            
            if len(pred_entities) == 0 and len(ref_entities) == 0:
                precision = recall = f1 = 1.0
            elif len(pred_entities) == 0:
                precision = recall = f1 = 0.0
            else:
                intersection = pred_entities & ref_entities
                precision = len(intersection) / len(pred_entities) if pred_entities else 0.0
                recall = len(intersection) / len(ref_entities) if ref_entities else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
        
        return {
            'f1': np.mean(all_f1),
            'precision': np.mean(all_precision),
            'recall': np.mean(all_recall),
            'individual_f1': all_f1
        }
    
    def calculate_all_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        if len(predictions) != len(references):
            print(f"  Warning: Mismatch - {len(predictions)} predictions vs {len(references)} references")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        empty_preds = sum(1 for p in predictions if not p or not p.strip())
        if empty_preds > 0:
            print(f"  Warning: {empty_preds}/{len(predictions)} predictions are empty")
        
        print("  Calculating BLEU...")
        bleu = self.calculate_bleu(predictions, references)
        
        print("  Calculating CIDEr...")
        cider_avg, cider_scores = self.calculate_cider(predictions, references)
        
        print("  Calculating METEOR...")
        if self.meteor_available:
            meteor_avg, meteor_scores = self.calculate_meteor(predictions, references)
        else:
            meteor_avg = 0.0
            meteor_scores = [0.0] * len(predictions)
        
        print("  Calculating RadGraph F1...")
        radgraph = self.calculate_radgraph_f1(predictions, references)
        
        return {
            'bleu': bleu,
            'cider': cider_avg,
            'meteor': meteor_avg,
            'radgraph_f1': radgraph['f1'],
            'radgraph_precision': radgraph['precision'],
            'radgraph_recall': radgraph['recall'],
            'cider_scores': cider_scores,
            'meteor_scores': meteor_scores,
            'radgraph_f1_scores': radgraph['individual_f1'],
            'meteor_available': self.meteor_available  # Track availability
        }


class BLIPMambaModel(nn.Module):
    """BLIP encoder + Mamba decoder model"""
    
    def __init__(self, encoder, decoder, processor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.processor = processor
    
    def forward(self, images, caption_ids=None, attention_mask=None):
        """Training forward pass"""
        output = self.encoder(images)
        
        # Extract tensor from BaseModelOutputWithPooling object
        if hasattr(output, 'last_hidden_state'):
            visual_features = output.last_hidden_state
        elif isinstance(output, tuple):
            visual_features = output[0]
        else:
            visual_features = output
        
        # Ensure correct shape [batch, seq_len, dim]
        if len(visual_features.shape) == 2:
            visual_features = visual_features.unsqueeze(1)
        
        return self.decoder(visual_features, caption_ids, attention_mask)
    
    def generate(self, images, max_length=200, temperature=1.0):
        """Generate captions"""
        self.eval()
        with torch.no_grad():
            output = self.encoder(images)
            
            # Extract tensor from BaseModelOutputWithPooling object
            if hasattr(output, 'last_hidden_state'):
                visual_features = output.last_hidden_state
            elif isinstance(output, tuple):
                visual_features = output[0]
            else:
                visual_features = output
                
            if len(visual_features.shape) == 2:
                visual_features = visual_features.unsqueeze(1)
            
            if visual_features.shape[1] > 1:
                visual_context = visual_features.mean(dim=1)
            else:
                visual_context = visual_features.squeeze(1)
            
            visual_context = self.decoder.visual_projection(visual_context)
            generated_ids = self.decoder._generate(
                visual_context, max_length=max_length, temperature=temperature
            )
        return generated_ids


class ExperimentRunner:
    """Main class to run all experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.metrics_calc = MetricsCalculator()
        
        # directories here
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        
        self.tokenizer = CaptionTokenizer(config.vocab_path)
        self.vocab_size = self.tokenizer.vocab_size
        
        
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
        
        
        self.results = defaultdict(dict)
        
        print("ok! ExperimentRunner initialized")
        print(f"  Device: {self.device}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  RoPE: {config.use_rope}")
    
    def setup_blip_encoder(self):
        """Setup BLIP encoder (similar to notebook)"""
        print("\n" + "="*80)
        print("SETTING UP BLIP ENCODER")
        print("="*80)
        
        base_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        encoder = base_blip.vision_model
        
        
        encoder = encoder.to(self.device)
        
        # Freeze this is police
        for param in encoder.parameters():
            param.requires_grad = False
        
        if hasattr(encoder, 'blocks'):
            num_blocks = len(encoder.blocks)
            for param in encoder.blocks[-1].parameters():
                param.requires_grad = True
            print(f"ok! Unfrozen last BLIP encoder block ({num_blocks-1})")
        
        # Get feature dimension
        dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = encoder(dummy_img)
            
            # Extract last_hidden_state from BaseModelOutputWithPooling
            if hasattr(output, 'last_hidden_state'):
                feats = output.last_hidden_state
            elif isinstance(output, tuple):
                feats = output[0]
            else:
                feats = output
                
            visual_feature_dim = feats.shape[-1]
        
        print(f"ok! BLIP encoder feature dimension: {visual_feature_dim}")
        return encoder, visual_feature_dim
    
    def setup_mamba_decoder(self, visual_feature_dim: int):
        """Setup Mamba decoder"""
        print("\n" + "="*80)
        print("SETTNG UP MAMBA DECODER")
        print("="*80)
        

        #overfitting present
        decoder = MambaDecoder(
            vocab_size=self.vocab_size,
            visual_feature_dim=visual_feature_dim,
            d_model=512,  # need to Update to match BLIP decoder size
            n_layers=6,  # need to Updates to match BLIP decoder size
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,  
            use_rope=self.config.use_rope
        ).to(self.device)
        
        total_params = sum(p.numel() for p in decoder.parameters())
        trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print("ok! Mamba Decoder:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parametrs: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  RoPE: {self.config.use_rope}")
        
        return decoder
    
    def create_dataloaders(self, data_percentage: float = 1.0):
        """Create data loaders with specified data percntage"""
        print(f"\nCreating dataloaders ({data_percentage*100:.0f}% of data)..")
        
        # BLIP normalization
        from torchvision import transforms
        blip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        train_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            blip_normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            blip_normalize
        ])
        
        # Load data
        train_df = pd.read_csv(self.config.train_csv)
        val_df = pd.read_csv(self.config.val_csv)
        test_df = pd.read_csv(self.config.test_csv)
        
        # Subsample training data
        if data_percentage < 1.0:
            n_samples = int(len(train_df) * data_percentage)
            train_df = train_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            print(f"  Using {len(train_df)} training samples ({data_percentage*100:.0f}%)")
        
        # Save temporary CSVs for dataset loading
        temp_train_csv = os.path.join(self.config.output_dir, f"temp_train_{data_percentage}.csv")
        temp_val_csv = os.path.join(self.config.output_dir, "temp_val.csv")
        temp_test_csv = os.path.join(self.config.output_dir, "temp_test.csv")
        
        train_df.to_csv(temp_train_csv, index=False)
        val_df.to_csv(temp_val_csv, index=False)
        test_df.to_csv(temp_test_csv, index=False)
        
        # Create datasets
        train_dataset = MedicalCaptionDataset(
            data_csv=temp_train_csv,
            image_dir=self.config.image_dir,
            tokenizer=self.tokenizer,
            image_transform=train_transform,
            max_caption_length=self.config.max_seq_len
        )
        
        val_dataset = MedicalCaptionDataset(
            data_csv=temp_val_csv,
            image_dir=self.config.image_dir,
            tokenizer=self.tokenizer,
            image_transform=val_transform,
            max_caption_length=self.config.max_seq_len
        )
        
        test_dataset = MedicalCaptionDataset(
            data_csv=temp_test_csv,
            image_dir=self.config.image_dir,
            tokenizer=self.tokenizer,
            image_transform=val_transform,
            max_caption_length=self.config.max_seq_len
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=2
        )
        
        print("ok! Datasets created:")
        print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader, test_df
    
    def train_model(self, model, train_loader, val_loader, experiment_name: str):
        """Train the model and track speed/memory"""
        print(f"\n{'='*80}")
        print(f"TRAINING: {experiment_name}")
        print(f"{'='*80}")
        
        from torch.optim import Adam
        from torch.nn import CrossEntropyLoss
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = CrossEntropyLoss(ignore_index=self.tokenizer.pad_idx, reduction='mean')
        
        # Learning rate scheduler - reduces LR when val loss plateaus
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, 
            min_lr=1e-6
        )
        
        train_losses = []
        val_losses = []
        epoch_times = []
        memory_usage = []
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Training
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                caption_ids = batch['caption_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                decoder_input = caption_ids[:, :-1]
                decoder_target = caption_ids[:, 1:]
                target_mask = attention_mask[:, 1:]
                
                logits = model(images, decoder_input, target_mask)
                
                logits = logits.reshape(-1, logits.shape[-1])
                targets = decoder_target.reshape(-1)
                target_mask = target_mask.reshape(-1).float()
                
                loss = criterion(logits, targets)
                loss = (loss * target_mask).sum() / (target_mask.sum() + 1e-8)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_train_loss = epoch_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    caption_ids = batch['caption_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    decoder_input = caption_ids[:, :-1]
                    decoder_target = caption_ids[:, 1:]
                    target_mask = attention_mask[:, 1:]
                    
                    logits = model(images, decoder_input, target_mask)
                    
                    logits = logits.reshape(-1, logits.shape[-1])
                    targets = decoder_target.reshape(-1)
                    target_mask = target_mask.reshape(-1).float()
                    
                    loss = criterion(logits, targets)
                    loss = (loss * target_mask).sum() / (target_mask.sum() + 1e-8)
                    
                    epoch_val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
            val_losses.append(avg_val_loss)
            
            
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                memory_usage.append(memory_mb)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.2f}s")
            
            
            if patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs "
                      f"(val loss didn't improve for {self.config.early_stopping_patience} epochs)")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    model.to(self.device)
                break
        
        avg_epoch_time = np.mean(epoch_times)
        total_training_time = sum(epoch_times)
        peak_memory = max(memory_usage) if memory_usage else 0
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epoch_times': epoch_times,
            'avg_epoch_time': avg_epoch_time,
            'total_training_time': total_training_time,
            'peak_memory_mb': peak_memory,
            'memory_usage': memory_usage
        }
    
    def generate_captions(self, model, test_loader, max_length: int = None):
        """Generate captions and measure inference speed"""
        if max_length is None:
            max_length = self.config.max_seq_len
        
        print(f"\nGenerating captions (max_length={max_length})...")
        
        model.eval()
        all_captions = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating"):
                images = batch['image'].to(self.device)
                
                start_time = time.time()
                generated_ids = model.generate(images, max_length=max_length, temperature=1.0)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Decode
                for gen_ids in generated_ids:
                    caption = self.tokenizer.decode(gen_ids)
                    # Strip whitespace and ensure non-empty
                    caption = caption.strip() if caption else ""
                    # If empty after decoding, use a placeholder
                    if not caption:
                        caption = "<empty>"
                    all_captions.append(caption)
        
        avg_inference_time = np.mean(inference_times)
        total_inference_time = sum(inference_times)
        tokens_per_second = len(all_captions) * max_length / total_inference_time if total_inference_time > 0 else 0
        
        return {
            'captions': all_captions,
            'avg_inference_time': avg_inference_time,
            'total_inference_time': total_inference_time,
            'tokens_per_second': tokens_per_second,
            'inference_times': inference_times
        }
    
    def experiment_1_performance_comparison(self, blip_model, mamba_model, test_df):
        """Performance comparison across metrics."""
        
        # Generate BLIP captions
        print("\nGenerating BLIP captions...")
        blip_captions = []
        blip_model.eval()
        test_images = [Image.open(f"{self.config.image_dir}/{f}").convert('RGB') 
                      for f in test_df['filename']]
        
        for img in tqdm(test_images, desc="BLIP generation"):
            inputs = self.blip_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = blip_model.generate(**inputs, max_new_tokens=200)
                caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                blip_captions.append(caption)
        
            # Generate Mamba captions
        print("\nGenerating Mamba captions...")
        # Create test loader
        test_df_full = pd.read_csv(self.config.test_csv)
        temp_test_csv = os.path.join(self.config.output_dir, "temp_test_full.csv")
        test_df_full.to_csv(temp_test_csv, index=False)
        
        from torchvision import transforms
        blip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        test_transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            blip_normalize
        ])
        
        test_dataset = MedicalCaptionDataset(
            data_csv=temp_test_csv,
            image_dir=self.config.image_dir,
            tokenizer=self.tokenizer,
            image_transform=test_transform,
            max_caption_length=200
        )
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2)
        
        mamba_result = self.generate_captions(mamba_model, test_loader, max_length=200)
        mamba_captions = mamba_result['captions']
        
        # Calculate metrics
        references = test_df['findings'].tolist()
        
        print("\nCalculating metrics for BLIP...")
        blip_metrics = self.metrics_calc.calculate_all_metrics(blip_captions, references)
        
        print("\nCalculating metrics for Mamba...")
        mamba_metrics = self.metrics_calc.calculate_all_metrics(mamba_captions, references)
        
        results = {
            'blip': blip_metrics,
            'mamba': mamba_metrics,
            'blip_captions': blip_captions,
            'mamba_captions': mamba_captions
        }
        
        print("\nPerformance comparison results")
        print(f"{'Metric':<20} {'BLIP':<15} {'Mamba':<15} {'Difference':<15}")
        print("-"*80)
        print(f"{'BLEU-1':<20} {blip_metrics['bleu']['precisions'][0]:<15.4f} "
              f"{mamba_metrics['bleu']['precisions'][0]:<15.4f} "
              f"{mamba_metrics['bleu']['precisions'][0] - blip_metrics['bleu']['precisions'][0]:<15.4f}")
        print(f"{'BLEU-4':<20} {blip_metrics['bleu']['bleu']:<15.4f} "
              f"{mamba_metrics['bleu']['bleu']:<15.4f} "
              f"{mamba_metrics['bleu']['bleu'] - blip_metrics['bleu']['bleu']:<15.4f}")
        print(f"{'CIDEr':<20} {blip_metrics['cider']:<15.4f} "
              f"{mamba_metrics['cider']:<15.4f} "
              f"{mamba_metrics['cider'] - blip_metrics['cider']:<15.4f}")
        print(f"{'METEOR':<20} {blip_metrics['meteor']:<15.4f} "
              f"{mamba_metrics['meteor']:<15.4f} "
              f"{mamba_metrics['meteor'] - blip_metrics['meteor']:<15.4f}")
        print(f"{'RadGraph F1':<20} {blip_metrics['radgraph_f1']:<15.4f} "
              f"{mamba_metrics['radgraph_f1']:<15.4f} "
              f"{mamba_metrics['radgraph_f1'] - blip_metrics['radgraph_f1']:<15.4f}")
        
        return results
    
    def run_all_experiments(self):
        """Run all experiments for this configuration."""
        print(
            f"Running experiments (RoPE={self.config.use_rope}, "
            f"data={self.config.data_percentage*100:.0f}%, "
            f"max_len={self.config.max_seq_len})"
        )

        blip_encoder, visual_feature_dim = self.setup_blip_encoder()
        mamba_decoder = self.setup_mamba_decoder(visual_feature_dim)
        mamba_model = BLIPMambaModel(blip_encoder, mamba_decoder, self.blip_processor).to(self.device)

        print("\nLoading BLIP model...")
        base_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        if os.path.exists("blip-chest-xray-lora"):
            blip_model = PeftModel.from_pretrained(base_blip, "blip-chest-xray-lora")
        else:
            blip_model = base_blip
        blip_model.to(self.device)
        blip_model.eval()

        train_loader, val_loader, test_loader, test_df = self.create_dataloaders(
            data_percentage=self.config.data_percentage
        )

        training_results = self.train_model(
            mamba_model,
            train_loader,
            val_loader,
            f"Mamba (RoPE={self.config.use_rope}, {self.config.data_percentage*100:.0f}% data)",
        )

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"mamba_rope_{self.config.use_rope}_data_{self.config.data_percentage}.pth",
        )
        torch.save(
            {
                "model_state_dict": mamba_model.state_dict(),
                "config": asdict(self.config),
                "training_results": training_results,
            },
            checkpoint_path,
        )
        print(f"\nok! Checkpoint saved: {checkpoint_path}")

        exp1_results = self.experiment_1_performance_comparison(blip_model, mamba_model, test_df)
        self.results["experiment_1"] = exp1_results

        # Speed analysis
        blip_times = []
        test_images = [
            Image.open(f"{self.config.image_dir}/{f}").convert("RGB")
            for f in test_df["filename"][:100]
        ]
        for img in tqdm(test_images, desc="BLIP speed test"):
            inputs = self.blip_processor(images=img, return_tensors="pt").to(self.device)
            start = time.time()
            with torch.no_grad():
                _ = blip_model.generate(**inputs, max_new_tokens=200)
            blip_times.append(time.time() - start)

        blip_avg_time = np.mean(blip_times)
        blip_tokens_per_sec = 200 / blip_avg_time if blip_avg_time > 0 else 0
        mamba_avg_time = exp1_results.get("mamba_inference_time", 0)

        speed_results = {
            "blip_avg_inference_time": blip_avg_time,
            "blip_tokens_per_second": blip_tokens_per_sec,
            "mamba_avg_inference_time": mamba_avg_time,
            "mamba_tokens_per_second": training_results.get("tokens_per_second", 0),
            "training_time": training_results["total_training_time"],
            "avg_epoch_time": training_results["avg_epoch_time"],
        }
        self.results["experiment_2"] = speed_results

        # Sequence length sensitivity (Mamba only)
        token_lengths = [20, 100, 200]
        references = test_df["findings"].tolist()

        mamba_seq_results = {}
        for max_len in token_lengths:
            print(f"Mamba max_length={max_len}")
            mamba_out = self.generate_captions(mamba_model, test_loader, max_length=max_len)
            metrics_len = self.metrics_calc.calculate_all_metrics(mamba_out["captions"], references)
            mamba_seq_results[f"{max_len}_tokens"] = {
                "metrics": metrics_len,
                "captions": mamba_out["captions"],
                "avg_inference_time": mamba_out["avg_inference_time"],
                "tokens_per_second": mamba_out["tokens_per_second"],
            }

        self.results["experiment_3"] = {"mamba": mamba_seq_results}

        # Qualitative assessment
        sample_indices = [0, 10, 20, 30, 40]
        qualitative_samples = []
        for idx in sample_indices:
            if idx < len(test_df):
                sample = {
                    "image_id": test_df.iloc[idx]["uid"],
                    "filename": test_df.iloc[idx]["filename"],
                    "ground_truth": test_df.iloc[idx]["findings"],
                    "blip_prediction": exp1_results["blip_captions"][idx],
                    "mamba_prediction": exp1_results["mamba_captions"][idx],
                }
                qualitative_samples.append(sample)

        self.results["experiment_5"] = {
            "samples": qualitative_samples,
            "total_samples": len(qualitative_samples),
        }

        # Memory profiling
        memory_results = {
            "training_peak_memory_mb": training_results["peak_memory_mb"],
            "training_memory_usage": training_results["memory_usage"],
            "inference_memory_mb": 0,
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = self.generate_captions(mamba_model, test_loader, max_length=200)
            memory_results["inference_memory_mb"] = (
                torch.cuda.max_memory_allocated() / 1024**2
            )

        self.results["experiment_6"] = memory_results
        self.results["config"] = asdict(self.config)
        self.results["training_results"] = training_results

        return self.results
    
    def save_results(self, results: Dict, test_df: pd.DataFrame = None):
        """Save results to disk."""
        
        # Save JSON
        json_path = os.path.join(
            self.config.output_dir,
            f"results_rope_{self.config.use_rope}_data_{self.config.data_percentage}.json"
        )
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"ok! JSON saved: {json_path}")
        
        # Load test_df if not provided
        if test_df is None:
            test_df = pd.read_csv(self.config.test_csv)
        
        # Create summary DataFrame
        summary_data = []
        
        if 'experiment_1' in results:
            blip_metrics = results['experiment_1']['blip']
            mamba_metrics = results['experiment_1']['mamba']
            
            summary_data.append({
                'Model': 'BLIP',
                'BLEU-1': blip_metrics['bleu']['precisions'][0],
                'BLEU-4': blip_metrics['bleu']['bleu'],
                'CIDEr': blip_metrics['cider'],
                'METEOR': blip_metrics['meteor'],
                'RadGraph F1': blip_metrics['radgraph_f1'],
                'RoPE': 'N/A',
                'Data %': 'N/A'
            })
            
            summary_data.append({
                'Model': 'Mamba',
                'BLEU-1': mamba_metrics['bleu']['precisions'][0],
                'BLEU-4': mamba_metrics['bleu']['bleu'],
                'CIDEr': mamba_metrics['cider'],
                'METEOR': mamba_metrics['meteor'],
                'RadGraph F1': mamba_metrics['radgraph_f1'],
                'RoPE': self.config.use_rope,
                'Data %': f"{self.config.data_percentage*100:.0f}%"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_path = os.path.join(self.config.output_dir, f"summary_rope_{self.config.use_rope}_data_{self.config.data_percentage}.csv")
            summary_df.to_csv(csv_path, index=False)
            print(f"ok! CSV saved: {csv_path}")
            
            # Save to Excel
            excel_path = os.path.join(self.config.output_dir, f"results_rope_{self.config.use_rope}_data_{self.config.data_percentage}.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add detailed sheets
                if 'experiment_1' in results:
                    blip_df = pd.DataFrame([results['experiment_1']['blip_captions']], 
                                           columns=[f'Image_{i}' for i in range(len(results['experiment_1']['blip_captions']))])
                    mamba_df = pd.DataFrame([results['experiment_1']['mamba_captions']],
                                          columns=[f'Image_{i}' for i in range(len(results['experiment_1']['mamba_captions']))])
                    
                    
                    min_len = min(len(test_df), len(results['experiment_1']['blip_captions']))
                    captions_df = pd.DataFrame({
                        'Image_ID': test_df['uid'].tolist()[:min_len],
                        'Ground_Truth': test_df['findings'].tolist()[:min_len],
                        'BLIP_Prediction': results['experiment_1']['blip_captions'][:min_len],
                        'Mamba_Prediction': results['experiment_1']['mamba_captions'][:min_len]
                    })
                    captions_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            print(f"ok! Excel saved: {excel_path}")
        
        
        self.create_plots(results)
    
    def create_plots(self, results: Dict):
        """Create visualization plots"""
        print("\nCreating plots...")
        
        fig_dir = os.path.join(self.config.output_dir, 'plots')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Plot 1: Metrics comparison (BLIP vs Mamba)
        if 'experiment_1' in results:
            blip_metrics = results['experiment_1']['blip']
            mamba_metrics = results['experiment_1']['mamba']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_names = ['BLEU-1', 'BLEU-4', 'CIDEr', 'METEOR', 'RadGraph F1']
            blip_values = [
                blip_metrics['bleu']['precisions'][0],
                blip_metrics['bleu']['bleu'],
                blip_metrics['cider'],
                blip_metrics['meteor'],
                blip_metrics['radgraph_f1']
            ]
            mamba_values = [
                mamba_metrics['bleu']['precisions'][0],
                mamba_metrics['bleu']['bleu'],
                mamba_metrics['cider'],
                mamba_metrics['meteor'],
                mamba_metrics['radgraph_f1']
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            ax.bar(x - width/2, blip_values, width, label='BLIP', alpha=0.8)
            ax.bar(x + width/2, mamba_values, width, label='Mamba', alpha=0.8)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title(f'Performance Comparison (RoPE={self.config.use_rope})')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'metrics_comparison_rope_{self.config.use_rope}.png'), dpi=300)
            plt.close()
        
        # Plot 2: Training loss curves
        if 'training_results' in results:
            train_losses = results['training_results']['train_losses']
            val_losses = results['training_results']['val_losses']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2)
            ax.plot(epochs, val_losses, 's-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Curves (RoPE={self.config.use_rope}, {self.config.data_percentage*100:.0f}% data)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'training_curves_rope_{self.config.use_rope}_data_{self.config.data_percentage}.png'), dpi=300)
            plt.close()

            # Plot 3: Epoch time vs epoch (training speed)
            epoch_times = results['training_results']['epoch_times']
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(epoch_times) + 1)
            ax.plot(epochs, epoch_times, 'o-', color='tab:purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time per epoch (s)')
            ax.set_title(f'Training Speed (RoPE={self.config.use_rope}, {self.config.data_percentage*100:.0f}% data)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'epoch_times_rope_{self.config.use_rope}_data_{self.config.data_percentage}.png'), dpi=300)
            plt.close()

            # Plot 4: Training memory usage vs epoch (GPU memory)
            memory_usage = results['training_results'].get('memory_usage', [])
            if len(memory_usage) == len(epoch_times) and len(memory_usage) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(epochs, memory_usage, 'o-', color='tab:green', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Peak GPU memory per epoch (MB)')
                ax.set_title(f'Training Memory Usage (RoPE={self.config.use_rope}, {self.config.data_percentage*100:.0f}% data)')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f'training_memory_rope_{self.config.use_rope}_data_{self.config.data_percentage}.png'), dpi=300)
                plt.close()

        # Plot 5: Inference speed comparison (BLIP vs Mamba)
        if 'experiment_2' in results:
            speed = results['experiment_2']
            blip_tps = speed.get('blip_tokens_per_second', 0)
            mamba_tps = speed.get('mamba_tokens_per_second', 0)

            fig, ax = plt.subplots(figsize=(8, 6))
            models = ['BLIP', 'Mamba']
            tps_values = [blip_tps, mamba_tps]
            x = np.arange(len(models))

            ax.bar(x, tps_values, color=['tab:blue', 'tab:orange'], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_ylabel('Tokens per second')
            ax.set_title(f'Inference Throughput (RoPE={self.config.use_rope}, {self.config.data_percentage*100:.0f}% data)')
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'inference_speed_rope_{self.config.use_rope}_data_{self.config.data_percentage}.png'), dpi=300)
            plt.close()

        # Plot 6: Training vs inference peak memory (Mamba)
        if 'experiment_6' in results:
            mem = results['experiment_6']
            train_peak = mem.get('training_peak_memory_mb', 0)
            infer_peak = mem.get('inference_memory_mb', 0)

            fig, ax = plt.subplots(figsize=(8, 6))
            labels = ['Training peak', 'Inference peak']
            values = [train_peak, infer_peak]
            x = np.arange(len(labels))

            ax.bar(x, values, color=['tab:red', 'tab:cyan'], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylabel('GPU memory (MB)')
            ax.set_title(f'Mamba GPU Memory Usage (RoPE={self.config.use_rope}, {self.config.data_percentage*100:.0f}% data)')
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'memory_comparison_rope_{self.config.use_rope}_data_{self.config.data_percentage}.png'), dpi=300)
            plt.close()
        
        print(f"ok! Plots saved to: {fig_dir}")


def create_cross_run_plots(all_results: Dict[str, Dict]):
    """Create plots that compare across different runs."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd

    if not all_results:
        return

    # Use output_dir from any run (they should all share it)
    sample_run = next(iter(all_results.values()))
    cfg = sample_run.get("config", {})
    output_dir = cfg.get("output_dir", "evaluation_results")
    fig_dir = os.path.join(output_dir, "plots")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) BLIP vs Mamba at 10%, 50%, 100% data, 200 tokens (RoPE=False)
    target_data = [0.1, 0.5, 1.0]
    for dp in target_data:
        run = None
        for r in all_results.values():
            c = r.get("config", {})
            if (
                c.get("use_rope") is False
                and abs(c.get("data_percentage", 0.0) - dp) < 1e-6
                and "experiment_1" in r
            ):
                run = r
                break

        if run is None:
            continue

        exp1 = run["experiment_1"]
        blip_metrics = exp1["blip"]
        mamba_metrics = exp1["mamba"]

        metrics_names = ["BLEU-1", "BLEU-4", "CIDEr", "METEOR", "RadGraph F1"]
        blip_values = [
            blip_metrics["bleu"]["precisions"][0],
            blip_metrics["bleu"]["bleu"],
            blip_metrics["cider"],
            blip_metrics["meteor"],
            blip_metrics["radgraph_f1"],
        ]
        mamba_values = [
            mamba_metrics["bleu"]["precisions"][0],
            mamba_metrics["bleu"]["bleu"],
            mamba_metrics["cider"],
            mamba_metrics["meteor"],
            mamba_metrics["radgraph_f1"],
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, blip_values, width, label="BLIP", alpha=0.8)
        ax.bar(x + width / 2, mamba_values, width, label="Mamba", alpha=0.8)
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_title(f"BLIP vs Mamba (RoPE=False, {int(dp*100)}% data, 200 tokens)")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                fig_dir,
                f"performance_blip_vs_mamba_rope_False_data_{dp}_tokens_200.png",
            ),
            dpi=300,
        )
        plt.close()

    # 2) BLIP vs Mamba at 20 vs 200 tokens, 100% data (RoPE=False)
    # Use existing 200-token metrics; compute BLIP@20 by truncating captions.
    run_100 = None
    for r in all_results.values():
        c = r.get("config", {})
        if (
            c.get("use_rope") is False
            and abs(c.get("data_percentage", 0.0) - 1.0) < 1e-6
            and "experiment_1" in r
            and "experiment_3" in r
        ):
            run_100 = r
            break

    if run_100 is not None:
        cfg_100 = run_100["config"]
        exp1 = run_100["experiment_1"]
        seq_results = run_100.get("experiment_3", {}).get("mamba", {})
        mamba_20 = seq_results.get("20_tokens")

        if mamba_20 is not None:
            # BLIP and Mamba at 200 tokens
            blip_200 = exp1["blip"]
            mamba_200 = exp1["mamba"]

            # BLIP@20: truncate existing BLIP captions and recompute metrics
            metrics_calc = MetricsCalculator()
            test_df = pd.read_csv(cfg_100["test_csv"])
            references = test_df["findings"].tolist()
            blip_captions = exp1["blip_captions"]
            blip_captions_20 = [
                " ".join(c.split()[:20]) if isinstance(c, str) else ""
                for c in blip_captions
            ]
            blip_20 = metrics_calc.calculate_all_metrics(
                blip_captions_20, references
            )

            # Plot BLIP vs Mamba at 20 tokens
            metrics_names = ["BLEU-1", "BLEU-4", "CIDEr", "METEOR", "RadGraph F1"]
            blip20_vals = [
                blip_20["bleu"]["precisions"][0],
                blip_20["bleu"]["bleu"],
                blip_20["cider"],
                blip_20["meteor"],
                blip_20["radgraph_f1"],
            ]
            mamba20_vals = [
                mamba_20["metrics"]["bleu"]["precisions"][0],
                mamba_20["metrics"]["bleu"]["bleu"],
                mamba_20["metrics"]["cider"],
                mamba_20["metrics"]["meteor"],
                mamba_20["metrics"]["radgraph_f1"],
            ]

            x = np.arange(len(metrics_names))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width / 2, blip20_vals, width, label="BLIP (20)", alpha=0.8)
            ax.bar(x + width / 2, mamba20_vals, width, label="Mamba (20)", alpha=0.8)
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Score")
            ax.set_title("BLIP vs Mamba (RoPE=False, 100% data, 20 tokens)")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    fig_dir,
                    "performance_blip_vs_mamba_rope_False_data_1.0_tokens_20.png",
                ),
                dpi=300,
            )
            plt.close()

            # (The 200-token BLIP vs Mamba plot already exists per-run;
            # we leave it as-is.)

    # 3) Mamba with RoPE vs without RoPE at 200 tokens, 100% data
    run_no_rope = None
    run_rope = None
    for r in all_results.values():
        c = r.get("config", {})
        if abs(c.get("data_percentage", 0.0) - 1.0) >= 1e-6:
            continue
        if "experiment_1" not in r:
            continue
        if c.get("use_rope") is False and run_no_rope is None:
            run_no_rope = r
        if c.get("use_rope") is True and run_rope is None:
            run_rope = r

    if run_no_rope is not None and run_rope is not None:
        mamba_no_rope = run_no_rope["experiment_1"]["mamba"]
        mamba_rope = run_rope["experiment_1"]["mamba"]

        metrics_names = ["BLEU-1", "BLEU-4", "CIDEr", "METEOR", "RadGraph F1"]
        no_rope_vals = [
            mamba_no_rope["bleu"]["precisions"][0],
            mamba_no_rope["bleu"]["bleu"],
            mamba_no_rope["cider"],
            mamba_no_rope["meteor"],
            mamba_no_rope["radgraph_f1"],
        ]
        rope_vals = [
            mamba_rope["bleu"]["precisions"][0],
            mamba_rope["bleu"]["bleu"],
            mamba_rope["cider"],
            mamba_rope["meteor"],
            mamba_rope["radgraph_f1"],
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, no_rope_vals, width, label="Mamba (no RoPE)", alpha=0.8)
        ax.bar(x + width / 2, rope_vals, width, label="Mamba (RoPE)", alpha=0.8)
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_title("Mamba: RoPE vs no RoPE (200 tokens, 100% data)")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                fig_dir,
                "performance_mamba_rope_vs_no_rope_tokens_200_data_1.0.png",
            ),
            dpi=300,
        )
        plt.close()

def main():
    """Run all configured experiments."""
    experiments = [
        # Full experiments with different RoPE settings
        ExperimentConfig(use_rope=False, max_seq_len=200, data_percentage=1.0),
        ExperimentConfig(use_rope=True, max_seq_len=200, data_percentage=1.0),
        
        # Data efficiency experiments
        ExperimentConfig(use_rope=False, max_seq_len=200, data_percentage=0.1),
        ExperimentConfig(use_rope=False, max_seq_len=200, data_percentage=0.5),
        ExperimentConfig(use_rope=True, max_seq_len=200, data_percentage=0.1),
        ExperimentConfig(use_rope=True, max_seq_len=200, data_percentage=0.5),
    ]
    
    all_results = {}
    
    for i, config in enumerate(experiments, 1):
        print(f"\nExperiment {i}/{len(experiments)}")
        print(f"Config: RoPE={config.use_rope}, Data={config.data_percentage*100:.0f}%, MaxLen={config.max_seq_len}")

        runner = ExperimentRunner(config)
        results = runner.run_all_experiments()

        # Get test_df for saving
        test_df = pd.read_csv(config.test_csv)
        runner.save_results(results, test_df)
        all_results[f"run_{i}_rope_{config.use_rope}_data_{config.data_percentage}"] = results
        print(f"\nok! Experiment {i} completed")
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create final summary
    summary_path = os.path.join("evaluation_results", "final_summary.xlsx")
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        # Aggregate all results
        all_summaries = []
        for run_name, results in all_results.items():
            if 'experiment_1' in results:
                config = results.get('config', {})
                blip_metrics = results['experiment_1']['blip']
                mamba_metrics = results['experiment_1']['mamba']
                
                all_summaries.append({
                    'Run': run_name,
                    'RoPE': config.get('use_rope', False),
                    'Data %': f"{config.get('data_percentage', 1.0)*100:.0f}%",
                    'Model': 'BLIP',
                    'BLEU-4': blip_metrics['bleu']['bleu'],
                    'CIDEr': blip_metrics['cider'],
                    'METEOR': blip_metrics['meteor'],
                    'RadGraph F1': blip_metrics['radgraph_f1']
                })
                
                all_summaries.append({
                    'Run': run_name,
                    'RoPE': config.get('use_rope', False),
                    'Data %': f"{config.get('data_percentage', 1.0)*100:.0f}%",
                    'Model': 'Mamba',
                    'BLEU-4': mamba_metrics['bleu']['bleu'],
                    'CIDEr': mamba_metrics['cider'],
                    'METEOR': mamba_metrics['meteor'],
                    'RadGraph F1': mamba_metrics['radgraph_f1']
                })
        
        if all_summaries:
            final_df = pd.DataFrame(all_summaries)
            final_df.to_excel(writer, sheet_name='All Results', index=False)
            print(f"ok! Final summary saved: {summary_path}")
    
    # Cross-run comparison plots (data % sweeps, token length, RoPE vs no RoPE)
    create_cross_run_plots(all_results)


if __name__ == "__main__":
    main()

