import os

# Suppress tokenizers parallelism warning when using DataLoader with num_workers > 0
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from comprehensive_evaluation import (
    ExperimentConfig,
    ExperimentRunner as BaseRunner,
    BLIPMambaModel,
)


class FullComparisonRunner(BaseRunner):
    """Train and evaluate BLIP and BLIP+Mamba."""

    def train_blip_model(self, blip_model, train_loader, val_loader, experiment_name: str):
        print(f"\nTraining BLIP: {experiment_name}")

        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        trainable_params = [p for p in blip_model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found in BLIP model! "
                             "Make sure model parameters have requires_grad=True.")
        
        optimizer = Adam(trainable_params, lr=self.config.learning_rate)
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2,
            min_lr=1e-6
        )
        
        train_losses = []
        val_losses = []
        epoch_times = []
        memory_usage = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            blip_model.train()
            epoch_train_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"[BLIP] Epoch {epoch + 1}/{self.config.epochs}",
            )
            for batch in progress_bar:
                images = batch["image"].to(self.device)
                caption_ids = batch["caption_ids"]

                captions_text = [
                    self.tokenizer.decode(ids.tolist())
                    for ids in caption_ids
                ]

                text_inputs = self.blip_processor.tokenizer(
                    captions_text,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(self.device)
                attention_mask = text_inputs.attention_mask.to(self.device)
                labels = input_ids.clone()

                outputs = blip_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=images,
                    labels=labels,
                )
                loss = outputs.loss
                
                if not loss.requires_grad:
                    print(f"  Warning: Loss does not require grad. Checking model state...")
                    blip_model.train()
                    sample_param = next(blip_model.parameters())
                    if not sample_param.requires_grad:
                        raise RuntimeError("Model parameters don't require grad! "
                                         "Model may be frozen or in eval mode.")
                    outputs = blip_model(
                        input_ids=input_ids,
                        pixel_values=images,
                        labels=labels,
                    )
                    loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                epoch_train_loss += float(loss.item())
                num_batches += 1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = epoch_train_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)

            blip_model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    caption_ids = batch["caption_ids"]

                    captions_text = [
                        self.tokenizer.decode(ids.tolist())
                        for ids in caption_ids
                    ]

                    text_inputs = self.blip_processor.tokenizer(
                        captions_text,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_seq_len,
                        return_tensors="pt",
                    )
                    input_ids = text_inputs.input_ids.to(self.device)
                    attention_mask = text_inputs.attention_mask.to(self.device)
                    labels = input_ids.clone()

                    outputs = blip_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=images,
                        labels=labels,
                    )
                    loss = outputs.loss

                    epoch_val_loss += float(loss.item())
                    num_val_batches += 1

            avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in blip_model.state_dict().items()}
            else:
                patience_counter += 1

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                memory_usage.append(memory_mb)

            print(
                f"[BLIP] Epoch {epoch + 1}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"LR: {current_lr:.2e}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            if patience_counter >= self.config.early_stopping_patience:
                print(f"\n[BLIP] Early stopping after {epoch + 1} epochs")
                if best_model_state is not None:
                    blip_model.load_state_dict(best_model_state)
                    blip_model.to(self.device)
                break

        avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0
        total_training_time = float(sum(epoch_times))
        peak_memory = max(memory_usage) if memory_usage else 0.0

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch_times": epoch_times,
            "avg_epoch_time": avg_epoch_time,
            "total_training_time": total_training_time,
            "peak_memory_mb": peak_memory,
            "memory_usage": memory_usage,
        }

    def experiment_1_performance_comparison(self, blip_model, mamba_model, test_df):
        from PIL import Image
        import pandas as pd
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from dataset.medical_caption_dataset import MedicalCaptionDataset

        print("\nExperiment 1: performance comparison (BLIP vs Mamba)")

        print("\nGenerating BLIP captions...")
        blip_captions = []
        blip_model.eval()
        test_images = [
            Image.open(f"{self.config.image_dir}/{f}").convert("RGB")
            for f in test_df["filename"]
        ]

        for img in tqdm(test_images, desc="BLIP generation"):
            inputs = self.blip_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = blip_model.generate(**inputs, max_new_tokens=200)
                caption = self.blip_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
                blip_captions.append(caption)

        print("\nGenerating Mamba captions...")
        test_df_full = pd.read_csv(self.config.test_csv)
        temp_test_csv = os.path.join(self.config.output_dir, "temp_test_full.csv")
        test_df_full.to_csv(temp_test_csv, index=False)

        blip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                blip_normalize,
            ]
        )

        test_dataset = MedicalCaptionDataset(
            data_csv=temp_test_csv,
            image_dir=self.config.image_dir,
            tokenizer=self.tokenizer,
            image_transform=test_transform,
            max_caption_length=200,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
        )

        mamba_result = self.generate_captions(
            mamba_model, test_loader, max_length=200
        )
        mamba_captions = mamba_result["captions"]

        references = test_df["findings"].tolist()

        print("\nCalculating metrics for BLIP...")
        blip_metrics = self.metrics_calc.calculate_all_metrics(blip_captions, references)

        print("\nCalculating metrics for Mamba...")
        mamba_metrics = self.metrics_calc.calculate_all_metrics(
            mamba_captions, references
        )

        results = {
            "blip": blip_metrics,
            "mamba": mamba_metrics,
            "blip_captions": blip_captions,
            "mamba_captions": mamba_captions,
            




            "mamba_avg_inference_time": mamba_result["avg_inference_time"],
            "mamba_tokens_per_second": mamba_result["tokens_per_second"],
        }

        print("\nPerformance comparson (BLIP vs Mamba)")
        print(f"{'Metric':<20} {'BLIP':<15} {'Mamba':<15} {'Difference':<15}")
        print("-" * 80)
        print(
            f"{'BLEU-1':<20} {blip_metrics['bleu']['precisions'][0]:<15.4f} "
            f"{mamba_metrics['bleu']['precisions'][0]:<15.4f} "
            f"{mamba_metrics['bleu']['precisions'][0] - blip_metrics['bleu']['precisions'][0]:<15.4f}"
        )
        print(
            f"{'BLEU-4':<20} {blip_metrics['bleu']['bleu']:<15.4f} "
            f"{mamba_metrics['bleu']['bleu']:<15.4f} "
            f"{mamba_metrics['bleu']['bleu'] - blip_metrics['bleu']['bleu']:<15.4f}"
        )
        print(
            f"{'CIDEr':<20} {blip_metrics['cider']:<15.4f} "
            f"{mamba_metrics['cider']:<15.4f} "
            f"{mamba_metrics['cider'] - blip_metrics['cider']:<15.4f}"
        )
        print(
            f"{'METEOR':<20} {blip_metrics['meteor']:<15.4f} "
            f"{mamba_metrics['meteor']:<15.4f} "
            f"{mamba_metrics['meteor'] - blip_metrics['meteor']:<15.4f}"
        )
        print(
            f"{'RadGraph F1':<20} {blip_metrics['radgraph_f1']:<15.4f} "
            f"{mamba_metrics['radgraph_f1']:<15.4f} "
            f"{mamba_metrics['radgraph_f1'] - blip_metrics['radgraph_f1']:<15.4f}"
        )

        return results

    def run_all_experiments(self) -> Dict:
        """Run BLIP and Mamba experiments for this config."""
        from PIL import Image
        import pandas as pd
        from transformers import BlipForConditionalGeneration
        from peft import PeftModel

        print("\nFull comparison: BLIP (transformer) vs Mamba decoder")
        print(f"RoPE: {self.config.use_rope}")
        print(f"Max sequence length: {self.config.max_seq_len}")
        print(f"Data percentage: {self.config.data_percentage * 100:.0f}%")

        # 1) Load BLIP and fine-tune encoder + decoder
        print("\nLoading BLIP model (for training + comparison)...")
        base_blip = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        if os.path.exists("blip-chest-xray-lora"):
            blip_model = PeftModel.from_pretrained(
                base_blip, "blip-chest-xray-lora"
            )
        else:
            blip_model = base_blip
            
        blip_model.train()
        for param in blip_model.parameters():
            param.requires_grad = True

        blip_model.to(self.device)
            
        trainable_params = sum(p.numel() for p in blip_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in blip_model.parameters())
        print(f"  BLIP model: {trainable_params:,} trainable / {total_params:,} total parameters")

        train_loader, val_loader, test_loader, test_df = self.create_dataloaders(
            data_percentage=self.config.data_percentage
        )

        blip_training_results = self.train_blip_model(
            blip_model,
            train_loader,
            val_loader,
            f"BLIP (full, {self.config.data_percentage * 100:.0f}% data)",
        )
        print("\nok! Finished fine-tuning BLIP (encoder + decoder)")

        # 2) Build Mamba using the fine-tuned BLIP encoder (frozen)
        print("\nSetting up Mamba using the fine-tuned BLIP encoder...")
        blip_encoder = blip_model.vision_model.to(self.device)
        for p in blip_encoder.parameters():
            p.requires_grad = False

        visual_feature_dim = blip_encoder.config.hidden_size
        mamba_decoder = self.setup_mamba_decoder(visual_feature_dim)
        mamba_model = BLIPMambaModel(
            blip_encoder, mamba_decoder, self.blip_processor
        ).to(self.device)

        mamba_training_results = self.train_model(
            mamba_model,
            train_loader,
            val_loader,
            f"Mamba (RoPE={self.config.use_rope}, {self.config.data_percentage * 100:.0f}% data)",
        )

        self.results["training_results"] = mamba_training_results
        self.results["blip_training_results"] = blip_training_results
        self.results["mamba_training_results"] = mamba_training_results

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"mamba_rope_{self.config.use_rope}_data_{self.config.data_percentage}.pth",
        )
        torch.save(
            {
                "model_state_dict": mamba_model.state_dict(),
                "config": asdict(self.config),
                "training_results": mamba_training_results,
            },
            checkpoint_path,
        )
        print(f"\nok! Mamba checkpoint saved: {checkpoint_path}")

        # 3) Performance comparison
        exp1_results = self.experiment_1_performance_comparison(
            blip_model, mamba_model, test_df
        )
        self.results["experiment_1"] = exp1_results

        # 4) Speed analysis
        print("\nMeasuring BLIP inference speed...")
        blip_times = []
        test_images_speed = [
            Image.open(f"{self.config.image_dir}/{f}").convert("RGB")
            for f in test_df["filename"][:100]
        ]

        for img in tqdm(test_images_speed, desc="BLIP speed test"):
            inputs = self.blip_processor(images=img, return_tensors="pt").to(
                self.device
            )
            start = time.time()
            with torch.no_grad():
                _ = blip_model.generate(**inputs, max_new_tokens=200)
            blip_times.append(time.time() - start)

        blip_avg_time = float(np.mean(blip_times)) if blip_times else 0.0
        blip_tokens_per_sec = 200.0 / blip_avg_time if blip_avg_time > 0 else 0.0

        mamba_avg_time = exp1_results.get("mamba_avg_inference_time", 0.0)
        mamba_tokens_per_sec = exp1_results.get("mamba_tokens_per_second", 0.0)

        speed_results = {
            "blip_avg_inference_time": blip_avg_time,
            "blip_tokens_per_second": blip_tokens_per_sec,
            "mamba_avg_inference_time": mamba_avg_time,
            "mamba_tokens_per_second": mamba_tokens_per_sec,
            "mamba_training_time": mamba_training_results["total_training_time"],
            "mamba_avg_epoch_time": mamba_training_results["avg_epoch_time"],
            "blip_training_time": blip_training_results["total_training_time"],
            "blip_avg_epoch_time": blip_training_results["avg_epoch_time"],
        }
        self.results["experiment_2"] = speed_results

        # 5) Sequence length sensitivity (Mamba only)
        token_lengths = [20, 100, 200]
        references = test_df["findings"].tolist()

        mamba_seq_results = {}
        for max_len in token_lengths:
            print(f"Mamba max_length={max_len}")
            mamba_out = self.generate_captions(
                mamba_model, test_loader, max_length=max_len
            )
            metrics_len = self.metrics_calc.calculate_all_metrics(
                mamba_out["captions"], references
            )
            mamba_seq_results[f"{max_len}_tokens"] = {
                "metrics": metrics_len,
                "captions": mamba_out["captions"],
                "avg_inference_time": mamba_out["avg_inference_time"],
                "tokens_per_second": mamba_out["tokens_per_second"],
            }

        self.results["experiment_3"] = {"mamba": mamba_seq_results}

        # 6) Qualitative assessment
        sample_indices = [0, 10, 20, 30, 40]
        qualitative_samples = []

        for idx in sample_indices:
            if idx < len(test_df):
                qualitative_samples.append(
                    {
                        "image_id": test_df.iloc[idx]["uid"],
                        "filename": test_df.iloc[idx]["filename"],
                        "ground_truth": test_df.iloc[idx]["findings"],
                        "blip_prediction": exp1_results["blip_captions"][idx],
                        "mamba_prediction": exp1_results["mamba_captions"][idx],
                    }
                )

        self.results["experiment_5"] = {
            "samples": qualitative_samples,
            "total_samples": len(qualitative_samples),
        }

        # 7) Memory profiling
        memory_results = {
            "mamba_training_peak_memory_mb": mamba_training_results["peak_memory_mb"],
            "mamba_training_memory_usage": mamba_training_results["memory_usage"],
            "mamba_inference_memory_mb": 0.0,
            "blip_training_peak_memory_mb": blip_training_results["peak_memory_mb"],
            "blip_training_memory_usage": blip_training_results["memory_usage"],
            "blip_inference_memory_mb": 0.0,
        }

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = self.generate_captions(mamba_model, test_loader, max_length=200)
            memory_results["mamba_inference_memory_mb"] = (
                torch.cuda.max_memory_allocated() / 1024**2
            )

            torch.cuda.reset_peak_memory_stats()
            for img in tqdm(test_images_speed, desc="BLIP memory test"):
                inputs = self.blip_processor(
                    images=img, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    _ = blip_model.generate(**inputs, max_new_tokens=200)
            memory_results["blip_inference_memory_mb"] = (
                torch.cuda.max_memory_allocated() / 1024**2
            )

        self.results["experiment_6"] = memory_results
        self.results["config"] = asdict(self.config)

        return self.results

    def create_plots(self, results: Dict):
        import matplotlib.pyplot as plt
        import numpy as np

        super().create_plots(results)

        fig_dir = os.path.join(self.config.output_dir, "plots")
        os.makedirs(fig_dir, exist_ok=True)

        blip_tr = results.get("blip_training_results")
        mamba_tr = results.get("mamba_training_results")

        if blip_tr and mamba_tr:
            # Training loss curves
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs_blip = range(1, len(blip_tr["train_losses"]) + 1)
            epochs_mamba = range(1, len(mamba_tr["train_losses"]) + 1)
            ax.plot(
                epochs_blip,
                blip_tr["train_losses"],
                "o-",
                label="BLIP Train Loss",
                linewidth=2,
            )
            ax.plot(
                epochs_blip,
                blip_tr["val_losses"],
                "s-",
                label="BLIP Val Loss",
                linewidth=2,
            )
            ax.plot(
                epochs_mamba,
                mamba_tr["train_losses"],
                "o--",
                label="Mamba Train Loss",
                linewidth=2,
            )
            ax.plot(
                epochs_mamba,
                mamba_tr["val_losses"],
                "s--",
                label="Mamba Val Loss",
                linewidth=2,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(
                f"Training Curves BLIP vs Mamba "
                f"(RoPE={self.config.use_rope}, "
                f"{self.config.data_percentage * 100:.0f}% data)"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    fig_dir,
                    f"training_curves_blip_vs_mamba_rope_{self.config.use_rope}_data_{self.config.data_percentage}.png",
                ),
                dpi=300,
            )
            plt.close()

            # Training epoch time comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(mamba_tr["epoch_times"]) + 1)
            ax.plot(
                epochs,
                mamba_tr["epoch_times"],
                "o-",
                label="Mamba epoch time",
                linewidth=2,
            )
            if len(blip_tr["epoch_times"]) == len(mamba_tr["epoch_times"]):
                ax.plot(
                    epochs,
                    blip_tr["epoch_times"],
                    "s-",
                    label="BLIP epoch time",
                    linewidth=2,
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Time per epoch (s)")
            ax.set_title(
                f"Training Speed BLIP vs Mamba "
                f"(RoPE={self.config.use_rope}, "
                f"{self.config.data_percentage * 100:.0f}% data)"
            )
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    fig_dir,
                    f"epoch_times_blip_vs_mamba_rope_{self.config.use_rope}_data_{self.config.data_percentage}.png",
                ),
                dpi=300,
            )
            plt.close()

        # Additional plot: BLIP vs Mamba training/inference peak memory
        mem = results.get("experiment_6")
        if mem:
            labels = [
                "BLIP train",
                "BLIP infer",
                "Mamba train",
                "Mamba infer",
            ]
            values = [
                mem.get("blip_training_peak_memory_mb", 0.0),
                mem.get("blip_inference_memory_mb", 0.0),
                mem.get("mamba_training_peak_memory_mb", 0.0),
                mem.get("mamba_inference_memory_mb", 0.0),
            ]
            x = np.arange(len(labels))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x, values, color=["tab:blue", "tab:cyan", "tab:red", "tab:orange"])
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.set_ylabel("GPU memory (MB)")
            ax.set_title(
                f"GPU Memory BLIP vs Mamba "
                f"(RoPE={self.config.use_rope}, "
                f"{self.config.data_percentage * 100:.0f}% data)"
            )
            ax.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    fig_dir,
                    f"memory_blip_vs_mamba_rope_{self.config.use_rope}_data_{self.config.data_percentage}.png",
                ),
                dpi=300,
            )
            plt.close()


# Small helper alias so code using this file can do:
#   from new_revisions.full_comparison_runner import ExperimentConfig, FullComparisonRunner
ExperimentConfigAlias = ExperimentConfig


