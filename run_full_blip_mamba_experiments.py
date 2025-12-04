import os

# Suppress tokenizers parallelism warning when using DataLoader with num_workers > 0
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
from typing import List

import torch
import pandas as pd

from full_comparison_runner import ExperimentConfig, FullComparisonRunner


def run_single_experiment(config: ExperimentConfig, run_name: str) -> None:
    """Run a single configuration and save results."""
    print(f"\nRun: {run_name}")
    print(
        f"RoPE={config.use_rope}, "
        f"Data={config.data_percentage * 100:.0f}%, "
        f"MaxLen={config.max_seq_len}, "
        f"Epochs={config.epochs}, "
        f"BatchSize={config.batch_size}"
    )

    runner = FullComparisonRunner(config)
    results = runner.run_all_experiments()

    test_df = pd.read_csv(config.test_csv)
    runner.save_results(results, test_df)

    print(f"\nok! Experiment '{run_name}' completed and results saved.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("Full BLIP vs Mamba-2 experiments")

    max_seq_len = 200
    epochs = 2
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_percentage = 1.0
    rope_settings: List[bool] = [False, True]

    run_id = 1
    total_runs = len(rope_settings)

    for use_rope in rope_settings:
        run_name = (
            f"full_run_{run_id}_rope_{use_rope}_"
            f"data_{int(data_percentage * 100)}_maxlen_{max_seq_len}"
        )

        # Use relative paths (assumes running from project root)
        config = ExperimentConfig(
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            data_percentage=data_percentage,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
        )

        run_single_experiment(config, run_name)
        run_id += 1

    print("\nAll full comparison experiments completed")


if __name__ == "__main__":
    main()


