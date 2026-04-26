"""Signer-independent data splitting for unbiased evaluation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplitConfig:
    """Configuration for signer-independent splits."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    min_samples_per_signer: int = 10


class SignerIndependentSplitter:
    """
    Creates signer-independent train/val/test splits.

    Ensures no signer appears in multiple splits, providing
    unbiased evaluation of model generalization to new signers.
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()
        self._validate_ratios()

    def _validate_ratios(self):
        total = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def split(
        self, df: pd.DataFrame, signer_col: str = "participant_id"
    ) -> Dict[str, pd.DataFrame]:
        """
        Split dataframe by signer ID.

        Args:
            df: DataFrame with samples and signer information
            signer_col: Column name containing signer IDs

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if signer_col not in df.columns:
            raise ValueError(f"Column '{signer_col}' not found in dataframe")

        # Get unique signers and their sample counts
        signer_counts = df[signer_col].value_counts()

        # Filter signers with minimum samples
        valid_signers = signer_counts[
            signer_counts >= self.config.min_samples_per_signer
        ].index.tolist()

        if len(valid_signers) < 3:
            raise ValueError(
                f"Need at least 3 signers with >= {self.config.min_samples_per_signer} "
                f"samples, got {len(valid_signers)}"
            )

        # Shuffle signers
        np.random.seed(self.config.random_seed)
        shuffled_signers = np.random.permutation(valid_signers)

        # Calculate split points
        n_signers = len(shuffled_signers)
        n_train = int(n_signers * self.config.train_ratio)
        n_val = int(n_signers * self.config.val_ratio)

        # Assign signers to splits
        train_signers = set(shuffled_signers[:n_train])
        val_signers = set(shuffled_signers[n_train : n_train + n_val])
        test_signers = set(shuffled_signers[n_train + n_val :])

        # Filter valid signers only
        df_valid = df[df[signer_col].isin(valid_signers)]

        # Create splits
        splits = {
            "train": df_valid[df_valid[signer_col].isin(train_signers)].copy(),
            "val": df_valid[df_valid[signer_col].isin(val_signers)].copy(),
            "test": df_valid[df_valid[signer_col].isin(test_signers)].copy(),
        }

        # Store metadata
        self.split_info = {
            "n_signers": {
                k: len(s)
                for k, s in [
                    ("train", train_signers),
                    ("val", val_signers),
                    ("test", test_signers),
                ]
            },
            "n_samples": {k: len(v) for k, v in splits.items()},
            "signers": {
                "train": sorted(train_signers),
                "val": sorted(val_signers),
                "test": sorted(test_signers),
            },
        }

        return splits

    def get_split_info(self) -> Dict:
        """Return information about the last split performed."""
        if not hasattr(self, "split_info"):
            raise RuntimeError("No split has been performed yet")
        return self.split_info

    def print_split_summary(self):
        """Print a summary of the split."""
        info = self.get_split_info()
        print("Signer-Independent Split Summary")
        print("=" * 40)
        for split_name in ["train", "val", "test"]:
            print(
                f"{split_name.capitalize():>8}: {info['n_signers'][split_name]:>3} signers, "
                f"{info['n_samples'][split_name]:>5} samples"
            )


def create_signer_splits(
    csv_path: str, base_path: Optional[str] = None, config: Optional[SplitConfig] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Convenience function to create signer-independent splits from a CSV file.

    Args:
        csv_path: Path to the train.csv file
        base_path: Base path for landmark files (prepended to 'path' column)
        config: Split configuration

    Returns:
        Tuple of (splits dict, split info dict)
    """
    df = pd.read_csv(csv_path)

    # Add full path if base_path provided
    if base_path:
        df["full_path"] = df["path"].apply(lambda p: str(Path(base_path) / p))

    splitter = SignerIndependentSplitter(config)
    splits = splitter.split(df)

    splitter.print_split_summary()

    return splits, splitter.get_split_info()


def merge_datasets_with_pseudo_signers(
    datasets: List[Tuple[pd.DataFrame, str]], signer_col: str = "participant_id"
) -> pd.DataFrame:
    """
    Merge multiple datasets, creating pseudo-signer IDs for datasets without signer info.

    Args:
        datasets: List of (DataFrame, dataset_name) tuples
        signer_col: Column name for signer IDs

    Returns:
        Merged DataFrame with consistent signer IDs
    """
    merged_dfs = []

    for df, dataset_name in datasets:
        df = df.copy()
        df["dataset"] = dataset_name

        # If no signer column, create pseudo-signers from paths
        if signer_col not in df.columns:
            # Use unique prefixes from file paths as pseudo-signer IDs
            df[signer_col] = df["path"].apply(
                lambda p: f"{dataset_name}_" + Path(p).stem[:4]
            )
        else:
            # Prefix signer IDs with dataset name to avoid collisions
            df[signer_col] = df[signer_col].apply(lambda s: f"{dataset_name}_{s}")

        merged_dfs.append(df)

    return pd.concat(merged_dfs, ignore_index=True)
