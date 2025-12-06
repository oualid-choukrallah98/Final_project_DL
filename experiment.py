"""
Experimentation Script for BLIP vs Mamba Decoder Comparison
Generates performance plots for different configurations
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from PIL import Image
import random

from models.blip_model import BLIPModel
from models.mamba_decoder import MambaDecoder
from data_utils import get_data_loader
from evaluation_metrics import evaluate_all_metrics
from transformers import BlipProcessor

# Set seed for reproducibility
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed
set_seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ExperimentRunner:
    """Run experiments comparing BLIP and Mamba decoders"""
    
    def __init__(
        self,
        train_csv: str = "data/train_data.csv",
        test_csv: str = "data/test_data.csv",
        image_dir: str = "data/chest-xrays-indiana-university/images/images_normalized",
        device: str = None
    ):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.image_dir = image_dir
        
        # Ensure CUDA is used if available
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
            else:
                self.device = torch.device("cpu")
                print("WARNING: CUDA not available, using CPU")
        else:
            self.device = torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
            print(f"Using device: {self.device}")
        
        # Load processor with fast tokenizer
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_fast=True
        )
        
        # Results storage
        self.results = {}  # Legacy format for backward compatibility
        self.training_history = {}  # Store training/validation losses
        # Cache for unique configurations: key = (data_fraction, max_tokens, model_type)
        self.config_cache = {}  # Stores trained models and metrics
    
    def train_blip(
        self,
        data_fraction: float = 1.0,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 5e-5,
        max_length: int = 512,
        save_path: str = None,
        track_validation: bool = False,
        val_fraction: float = 0.1
    ) -> Tuple[BLIPModel, Dict]:
        """Train BLIP model with optional validation tracking"""
        print(f"Training BLIP with {data_fraction*100}% of data...")
        
        # Create model
        model = BLIPModel(use_lora=True)
        model.to(self.device)
        model.train()
        
        # Get data loaders
        train_loader, train_df = get_data_loader(
            self.image_dir,
            self.train_csv,
            self.processor,
            batch_size=batch_size,
            shuffle=True,
            data_fraction=data_fraction,
            max_length=max_length
        )
        
        # Create validation loader if needed
        val_loader = None
        if track_validation:
            val_loader, _ = get_data_loader(
                self.image_dir,
                self.train_csv,
                self.processor,
                batch_size=batch_size,
                shuffle=False,
                data_fraction=val_fraction,
                max_length=max_length
            )
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=lr)
        
        # Track losses
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"BLIP Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if track_validation and val_loader is not None:
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        pixel_values = batch["pixel_values"].to(self.device)
                        attention_mask = batch.get("attention_mask", None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        
                        outputs = model(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids,
                            attention_mask=attention_mask
                        )
                        epoch_val_loss += outputs.loss.item()
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")
        
        if save_path:
            model.save_pretrained(save_path)
            print(f"BLIP model saved to {save_path}")
        
        history = {"train_losses": train_losses, "val_losses": val_losses if track_validation else None}
        return model, history
    
    def train_mamba(
        self,
        use_rope: bool = False,
        data_fraction: float = 1.0,
        epochs: int = 5,
        batch_size: int = 4,
        lr: float = 5e-5,
        max_length: int = 512,
        d_model: int = 768,
        n_layers: int = 6,
        save_path: str = None,
        track_validation: bool = False,
        val_fraction: float = 0.1
    ) -> Tuple[MambaDecoder, Dict]:
        """Train Mamba decoder with optional validation tracking"""
        rope_str = "RoPE" if use_rope else "Standard"
        print(f"Training Mamba ({rope_str}) with {data_fraction*100}% of data...")
        
        # Create model
        model = MambaDecoder(
            vocab_size=30522,  # BLIP vocab size
            d_model=d_model,
            n_layers=n_layers,
            use_rope=use_rope
        )
        model.to(self.device)
        model.train()
        
        # Get data loaders
        train_loader, _ = get_data_loader(
            self.image_dir,
            self.train_csv,
            self.processor,
            batch_size=batch_size,
            shuffle=True,
            data_fraction=data_fraction,
            max_length=max_length
        )
        
        # Create validation loader if needed
        val_loader = None
        if track_validation:
            val_loader, _ = get_data_loader(
                self.image_dir,
                self.train_csv,
                self.processor,
                batch_size=batch_size,
                shuffle=False,
                data_fraction=val_fraction,
                max_length=max_length
            )
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=lr)
        
        # Track losses
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Mamba ({rope_str}) Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                pixel_values = batch["pixel_values"].to(self.device)
                
                # Create labels (shifted input_ids)
                labels = input_ids.clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
                
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels
                )
                
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if track_validation and val_loader is not None:
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        pixel_values = batch["pixel_values"].to(self.device)
                        
                        labels = input_ids.clone()
                        labels[labels == self.processor.tokenizer.pad_token_id] = -100
                        
                        outputs = model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            labels=labels
                        )
                        epoch_val_loss += outputs["loss"].item()
                
                avg_val_loss = epoch_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} Average Loss: {avg_train_loss:.4f}")
        
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Mamba model saved to {save_path}")
        
        history = {"train_losses": train_losses, "val_losses": val_losses if track_validation else None}
        return model, history
    
    def evaluate_model(
        self,
        model,
        max_new_tokens: int = 50,
        model_type: str = "blip"
    ) -> Tuple[List[str], List[str]]:
        """Evaluate model on test set"""
        print(f"Evaluating {model_type} model...")
        
        # Load test data
        df_test = pd.read_csv(self.test_csv)
        
        predictions = []
        references = []
        
        model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(len(df_test))):
                # Load image
                image_path = f"{self.image_dir}/{df_test['filename'].iloc[i]}"
                image = Image.open(image_path).convert('RGB')
                
                # Get reference
                reference = df_test['findings'].iloc[i]
                references.append(reference)
                
                # Generate prediction
                if model_type == "blip":
                    inputs = self.processor(images=image, return_tensors="pt")
                    # Move to device
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in inputs.items()}
                    generated_ids = model.generate(
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=max_new_tokens
                    )
                    prediction = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )[0]
                else:  # mamba
                    # Process image
                    inputs = self.processor(images=image, return_tensors="pt")
                    # Move to device
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in inputs.items()}
                    generated_ids = model.generate(
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=max_new_tokens
                    )
                    # Decode
                    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                        prediction = model.tokenizer.decode(
                            generated_ids[0], 
                            skip_special_tokens=True
                        )
                    else:
                        # Fallback: use processor tokenizer
                        prediction = self.processor.tokenizer.decode(
                            generated_ids[0], 
                            skip_special_tokens=True
                        )
                
                predictions.append(prediction)
        
        return predictions, references
    
    def _get_config_key(self, data_fraction: float, max_tokens: int, model_type: str) -> tuple:
        """Generate a unique key for a configuration"""
        return (data_fraction, max_tokens, model_type)
    
    def _train_and_evaluate_model(
        self,
        model_type: str,  # "blip", "mamba_standard", "mamba_rope"
        data_fraction: float,
        max_tokens: int,
        epochs: int = 5,
        batch_size: int = 4
    ) -> Dict:
        """Train and evaluate a single model configuration (with caching)"""
        config_key = self._get_config_key(data_fraction, max_tokens, model_type)
        
        # Check if already cached
        if config_key in self.config_cache:
            print(f"Using cached results for {model_type} at {data_fraction*100}% data, {max_tokens} tokens")
            return self.config_cache[config_key]
        
        print(f"\nTraining {model_type} at {data_fraction*100}% data, {max_tokens} tokens...")
        
        # Train model
        if model_type == "blip":
            model, _ = self.train_blip(
                data_fraction=data_fraction,
                epochs=epochs,
                batch_size=batch_size,
                max_length=max_tokens,
                track_validation=False
            )
        elif model_type == "mamba_standard":
            model, _ = self.train_mamba(
                use_rope=False,
                data_fraction=data_fraction,
                epochs=epochs,
                batch_size=batch_size,
                max_length=max_tokens,
                track_validation=False
            )
        elif model_type == "mamba_rope":
            model, _ = self.train_mamba(
                use_rope=True,
                data_fraction=data_fraction,
                epochs=epochs,
                batch_size=batch_size,
                max_length=max_tokens,
                track_validation=False
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Evaluate model
        model_type_for_eval = "blip" if model_type == "blip" else "mamba"
        preds, refs = self.evaluate_model(
            model,
            max_new_tokens=max_tokens,
            model_type=model_type_for_eval
        )
        metrics = evaluate_all_metrics(preds, refs)
        
        # Cache results
        result = {
            'model': model,
            'metrics': metrics,
            'predictions': preds,
            'references': refs
        }
        self.config_cache[config_key] = result
        
        # Map model_type to display name
        display_name = {
            "blip": "BLIP",
            "mamba_standard": "Mamba (Standard)",
            "mamba_rope": "Mamba (RoPE)"
        }[model_type]
        
        print(f"{display_name} Metrics: {metrics}")
        return result
    
    def run_experiment(
        self,
        experiment_name: str,
        data_fraction: float,
        max_tokens: int,
        epochs: int = 5,
        batch_size: int = 4,
        train_blip: bool = True,
        train_mamba_standard: bool = True,
        train_mamba_rope: bool = False
    ) -> Dict:
        """Run a single experiment configuration (uses caching to avoid duplicate runs)"""
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Data Fraction: {data_fraction*100}%, Max Tokens: {max_tokens}")
        print(f"{'='*60}\n")
        
        results = {}
        
        # Train and evaluate BLIP
        if train_blip:
            blip_result = self._train_and_evaluate_model(
                model_type="blip",
                data_fraction=data_fraction,
                max_tokens=max_tokens,
                epochs=epochs,
                batch_size=batch_size
            )
            results['BLIP'] = blip_result['metrics']
        
        # Train and evaluate Mamba (Standard)
        if train_mamba_standard:
            mamba_result = self._train_and_evaluate_model(
                model_type="mamba_standard",
                data_fraction=data_fraction,
                max_tokens=max_tokens,
                epochs=epochs,
                batch_size=batch_size
            )
            results['Mamba (Standard)'] = mamba_result['metrics']
        
        # Train and evaluate Mamba (RoPE)
        if train_mamba_rope:
            mamba_rope_result = self._train_and_evaluate_model(
                model_type="mamba_rope",
                data_fraction=data_fraction,
                max_tokens=max_tokens,
                epochs=epochs,
                batch_size=batch_size
            )
            results['Mamba (RoPE)'] = mamba_rope_result['metrics']
        
        self.results[experiment_name] = results
        return results
    
    def plot_comparison(self, experiment_name: str, save_path: str = None):
        """Plot comparison for a single experiment"""
        if experiment_name not in self.results:
            print(f"No results found for {experiment_name}")
            return
        
        results = self.results[experiment_name]
        models = list(results.keys())
        metrics = ['BLEU', 'CIDEr', 'METEOR', 'RadGraph F1']
        
        # Prepare data
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric,
                    'Score': results[model].get(metric, 0.0)
                })
        
        df_plot = pd.DataFrame(data)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Define distinct colors for each model
        model_colors = {
            'BLIP': '#1f77b4',  # Blue
            'Mamba (Standard)': '#ff7f0e',  # Orange
            'Mamba (RoPE)': '#2ca02c',  # Green
        }
        # Default colors if model not in dict
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            metric_data = df_plot[df_plot['Metric'] == metric]
            
            # Assign colors to each model
            colors = [model_colors.get(model, default_colors[i % len(default_colors)]) 
                     for i, model in enumerate(metric_data['Model'])]
            
            bars = ax.bar(metric_data['Model'], metric_data['Score'], 
                         alpha=0.7, color=colors)
            ax.set_title(f'{metric} Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_xlabel('Model', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        plt.suptitle(f'{experiment_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_experiments(self, save_dir: str = "plots"):
        """Plot all experiments"""
        os.makedirs(save_dir, exist_ok=True)
        
        for exp_name in self.results.keys():
            save_path = os.path.join(save_dir, f"{exp_name.replace(' ', '_')}.png")
            self.plot_comparison(exp_name, save_path)
    
    def run_training_validation_experiment(
        self,
        data_fraction: float = 1.0,
        max_tokens: int = 200,
        epochs: int = 5,
        batch_size: int = 4,
        val_fraction: float = 0.1
    ):
        """Run training/validation comparison for BLIP and Mamba at 100% data, 200 tokens"""
        print(f"\n{'='*60}")
        print(f"Training/Validation Comparison: 100% Data, {max_tokens} Tokens")
        print(f"{'='*60}\n")
        
        # Train BLIP with validation tracking
        print("Training BLIP with validation tracking...")
        blip_model, blip_history = self.train_blip(
            data_fraction=data_fraction,
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_tokens,
            track_validation=True,
            val_fraction=val_fraction
        )
        
        # Train Mamba with validation tracking
        print("\nTraining Mamba (Standard) with validation tracking...")
        mamba_model, mamba_history = self.train_mamba(
            use_rope=False,
            data_fraction=data_fraction,
            epochs=epochs,
            batch_size=batch_size,
            max_length=max_tokens,
            track_validation=True,
            val_fraction=val_fraction
        )
        
        # Store history
        self.training_history = {
            'BLIP': blip_history,
            'Mamba': mamba_history
        }
        
        return blip_model, mamba_model, blip_history, mamba_history
    
    def plot_training_validation_curves(self, save_path: str = None):
        """Plot training and validation curves for BLIP and Mamba on the same figure"""
        if not self.training_history:
            print("No training history found. Run training_validation_experiment first.")
            return
        
        blip_history = self.training_history.get('BLIP', {})
        mamba_history = self.training_history.get('Mamba', {})
        
        if not blip_history.get('train_losses') or not mamba_history.get('train_losses'):
            print("Training history incomplete.")
            return
        
        # Get epochs
        epochs = range(1, len(blip_history['train_losses']) + 1)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot BLIP training and validation
        plt.plot(epochs, blip_history['train_losses'], 'b-', label='BLIP Training', linewidth=2, marker='o')
        if blip_history.get('val_losses'):
            plt.plot(epochs, blip_history['val_losses'], 'b--', label='BLIP Validation', linewidth=2, marker='s')
        
        # Plot Mamba training and validation
        plt.plot(epochs, mamba_history['train_losses'], 'r-', label='Mamba Training', linewidth=2, marker='o')
        if mamba_history.get('val_losses'):
            plt.plot(epochs, mamba_history['val_losses'], 'r--', label='Mamba Validation', linewidth=2, marker='s')
        
        # Formatting
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Training and Validation Loss Comparison\n(BLIP vs Mamba Decoder, 100% Data, 200 Tokens)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training/validation plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, save_path: str = "results.json"):
        """Save results to JSON"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {save_path}")
    
    def export_to_excel(self, save_path: str = "results.xlsx"):
        """Export all results to Excel with a comprehensive table"""
        # Collect all unique configurations from cache
        rows = []
        
        for (data_fraction, max_tokens, model_type), result in self.config_cache.items():
            metrics = result['metrics']
            
            # Map model_type to display name
            display_name = {
                "blip": "BLIP",
                "mamba_standard": "Mamba (Standard)",
                "mamba_rope": "Mamba (RoPE)"
            }[model_type]
            
            row = {
                'Data Fraction (%)': data_fraction * 100,
                'Max Tokens': max_tokens,
                'Model': display_name,
                'BLEU': metrics.get('BLEU', 0.0),
                'CIDEr': metrics.get('CIDEr', 0.0),
                'METEOR': metrics.get('METEOR', 0.0),
                'RadGraph F1': metrics.get('RadGraph F1', 0.0),
            }
            
            # Add any additional metrics if they exist
            for key, value in metrics.items():
                if key not in ['BLEU', 'CIDEr', 'METEOR', 'RadGraph F1']:
                    row[key] = value
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by Data Fraction, Max Tokens, then Model
        df = df.sort_values(['Data Fraction (%)', 'Max Tokens', 'Model'])
        
        # Export to Excel
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Results']
            from openpyxl.utils import get_column_letter
            for idx, col in enumerate(df.columns, start=1):
                max_length = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.column_dimensions[get_column_letter(idx)].width = min(max_length, 50)
        
        print(f"Results exported to {save_path}")
        print(f"\nSummary Table:")
        print(df.to_string(index=False))
        return df


def main():
    """Main function to run all experiments"""
    # Initialize runner
    runner = ExperimentRunner()
    n_epochs = 2
    
    # Experiment 1: 10% data, 200 tokens
    runner.run_experiment(
        experiment_name="10% Data, 200 Tokens",
        data_fraction=0.1,
        max_tokens=200,
        epochs=n_epochs,  
        batch_size=16,
        train_blip=True,
        train_mamba_standard=True,
        train_mamba_rope=False
    )
    
    # Experiment 2: 100% data, 20 tokens
    runner.run_experiment(
        experiment_name="100% Data, 20 Tokens",
        data_fraction=1.0,
        max_tokens=20,
        epochs=n_epochs,
        batch_size=16,
        train_blip=True,
        train_mamba_standard=True,
        train_mamba_rope=False
    )
    
    # Experiment 3: 100% data, 200 tokens
    runner.run_experiment(
        experiment_name="100% Data, 200 Tokens",
        data_fraction=1.0,
        max_tokens=200,
        epochs=n_epochs,
        batch_size=16,
        train_blip=True,
        train_mamba_standard=True,
        train_mamba_rope=False
    )
    
    # Experiment 4: Mamba variants comparison (100% data, 200 tokens)
    runner.run_experiment(
        experiment_name="Mamba Variants, 100% Data, 200 Tokens",
        data_fraction=1.0,
        max_tokens=200,
        epochs=n_epochs,
        batch_size=16,
        train_blip=False,
        train_mamba_standard=True,
        train_mamba_rope=True
    )
    
    # Experiment 5: Training/Validation curves (100% data, 200 tokens)
    runner.run_training_validation_experiment(
        data_fraction=1.0,
        max_tokens=200,
        epochs=n_epochs,
        batch_size=16,
        val_fraction=0.1
    )
    
    # Generate plots
    runner.plot_all_experiments()
    
    # Generate training/validation plot
    runner.plot_training_validation_curves(save_path="plots/training_validation_curves.png")
    
    # Save results
    runner.save_results()
    
    # Export to Excel
    runner.export_to_excel("results.xlsx")


if __name__ == "__main__":
    main()

