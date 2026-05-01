import hashlib
import io
import json
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import List, Optional, Tuple
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from .augmentation import augment_sample
from .preprocessing import frame_stacked_data
from ..config import ALL_COLUMNS

try:
    import lmdb
    _LMDB_AVAILABLE = True
except ImportError:
    _LMDB_AVAILABLE = False

# Hash of the exact column list serialized into each cached tensor.
# Any change to ALL_COLUMNS (face selection, depth flag, ordering) forces a
# clean rebuild so stale incompatible tensors are never silently reused.
_CACHE_VERSION = hashlib.md5("|".join(ALL_COLUMNS).encode()).hexdigest()[:8]


def _lmdb_key(path: str) -> bytes:
    """LMDB lookup key: version-prefixed so stale archives never shadow new configs."""
    return f"{_CACHE_VERSION}:{path}".encode()


class ASLDataset(Dataset):
    """
    ASL landmark dataset with three-tier caching: LMDB → per-sample .pt → parquet.

    LMDB is the preferred backend on RunPod / network-attached storage: a single
    file avoids the per-open() overhead of 94k individual .pt files.  Build it
    once with `python -m cnn_transformer.data.build_lmdb`.

    The .pt cache is the fallback when LMDB hasn't been built yet.

    Velocity is computed at __getitem__ time after augmentation so augmented
    coordinates produce the correct body-relative velocity.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_path: str,
        cache_dir: Optional[str] = None,
        lmdb_path: Optional[str] = None,
        max_frames: int = 128,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_frames = max_frames
        self.augment = augment

        # LMDB: open with lock=False so multiple DataLoader workers can share
        # the same environment after fork() without deadlocking.
        # Guard: data.mdb must exist (empty dirs and interrupted builds crash lmdb.open).
        self.lmdb_env = None
        if lmdb_path and _LMDB_AVAILABLE:
            lmdb_dir = Path(lmdb_path)
            if lmdb_dir.exists() and (lmdb_dir / "data.mdb").exists():
                try:
                    self.lmdb_env = lmdb.open(
                        str(lmdb_path),
                        readonly=True,
                        lock=False,
                        readahead=False,
                        meminit=False,
                    )
                except lmdb.Error as exc:
                    import warnings
                    warnings.warn(
                        f"Could not open LMDB at {lmdb_path}: {exc}. "
                        "Falling back to .pt / parquet cache."
                    )
            elif lmdb_dir.exists():
                import warnings
                warnings.warn(
                    f"LMDB directory {lmdb_path} exists but data.mdb is missing "
                    "(empty or interrupted build). Falling back to .pt / parquet cache."
                )

        self.lengths = self._load_or_compute_lengths()

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path(self, idx: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        rel = self.df.iloc[idx]["path"]
        return self.cache_dir / Path(rel).with_suffix(f".{_CACHE_VERSION}.pt")

    def _fast_length_from_parquet(self, idx: int) -> int:
        """Count unique frames using only the 'frame' column — avoids reading all landmarks."""
        full_path = str(self.base_path / self.df.iloc[idx]["path"])
        try:
            return min(
                pd.read_parquet(full_path, columns=["frame"])["frame"].nunique(),
                self.max_frames,
            )
        except Exception:
            return self.max_frames

    def _load_coords(self, idx: int) -> torch.Tensor:
        """Return raw position coordinates (T, D_pos) as a float32 tensor."""
        # Tier 1: LMDB — single file, one open() per training run on network storage
        if self.lmdb_env is not None:
            key = _lmdb_key(self.df.iloc[idx]["path"])
            with self.lmdb_env.begin(buffers=True) as txn:
                val = txn.get(key)
                if val is not None:
                    return torch.load(io.BytesIO(bytes(val)), weights_only=True)

        # Tier 2: per-sample .pt cache
        cp = self._cache_path(idx)
        if cp is not None and cp.exists():
            return torch.load(cp, weights_only=True)

        # Tier 3: parse parquet and populate .pt cache for future runs
        full_path = str(self.base_path / self.df.iloc[idx]["path"])
        coords = torch.tensor(frame_stacked_data(full_path), dtype=torch.float32)
        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(coords, cp)
        return coords

    def _load_or_compute_lengths(self) -> List[int]:
        """Return per-sample sequence lengths, loading from sidecar JSON if available."""
        lengths_file = (
            (self.cache_dir / f"_lengths_{_CACHE_VERSION}.json")
            if self.cache_dir else None
        )
        if lengths_file is not None and lengths_file.exists():
            with open(lengths_file) as f:
                lengths = json.load(f)
            if len(lengths) == len(self.df):
                return lengths

        # Compute lengths. Prefer fast paths over full tensor loads:
        #   .pt exists → load tensor (still O(N) but avoids parquet parsing)
        #   otherwise  → read only the 'frame' column (columnar parquet is fast)
        # LMDB is intentionally skipped here: deserializing each tensor to check
        # len() is slower than reading one parquet column, and the sidecar makes
        # this O(N) scan a one-time cost.
        lengths = []
        for i in range(len(self.df)):
            cp = self._cache_path(i)
            if cp is not None and cp.exists():
                t = torch.load(cp, weights_only=True)
                lengths.append(min(len(t), self.max_frames))
            else:
                lengths.append(self._fast_length_from_parquet(i))

        if lengths_file is not None:
            lengths_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lengths_file, "w") as f:
                json.dump(lengths, f)
        return lengths

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        coords = self._load_coords(idx)  # (T, D_pos)
        label = int(self.df.iloc[idx]["sign"])

        if self.augment:
            coords = torch.tensor(augment_sample(coords.numpy()), dtype=torch.float32)

        if coords.shape[0] > self.max_frames:
            idxs = torch.linspace(0, coords.shape[0] - 1, self.max_frames).long()
            coords = coords[idxs]

        vel = torch.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]

        return torch.cat([coords, vel], dim=-1), label  # (T, 2*D_pos)


def collate_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences])
    max_len = int(lengths.max())
    B, D = len(sequences), sequences[0].shape[1]
    padded = torch.zeros(B, max_len, D)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        T = seq.shape[0]
        padded[i, :T] = seq
        mask[i, :T] = True
    return padded, mask, torch.tensor(labels)


class BucketBatchSampler(Sampler):
    """Groups sequences by length to minimise padding waste within each batch."""

    def __init__(self, lengths: List[int], batch_size: int, drop_last: bool = False):
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        sorted_idxs = np.argsort(self.lengths)
        buckets = [
            sorted_idxs[i : i + self.batch_size]
            for i in range(0, len(sorted_idxs), self.batch_size)
        ]
        if self.drop_last and len(buckets[-1]) < self.batch_size:
            buckets = buckets[:-1]
        np.random.shuffle(buckets)
        for b in buckets:
            yield list(b)

    def __len__(self) -> int:
        n = len(self.lengths)
        return (
            n // self.batch_size
            if self.drop_last
            else (n + self.batch_size - 1) // self.batch_size
        )


def get_data_loaders(
    data_dir: str,
    cache_dir: Optional[str] = None,
    lmdb_path: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    max_frames: int = 128,
) -> Tuple[DataLoader, DataLoader]:
    """
    Args:
        data_dir:    Directory with train.csv and sign_to_prediction_index_map.json.
        cache_dir:   Directory for per-sample .pt files (fallback when LMDB not built).
        lmdb_path:   Path to LMDB archive (recommended on network storage). Build with
                     `python -m cnn_transformer.data.build_lmdb`.
        batch_size:  Samples per batch.
        num_workers: DataLoader worker processes.
        max_frames:  Truncate sequences longer than this.
    """
    sign_map_file = Path(data_dir) / "sign_to_prediction_index_map.json"
    train_csv = Path(data_dir) / "train.csv"

    with open(sign_map_file) as f:
        sign2idx = json.load(f)

    df = pd.read_csv(train_csv)
    df["sign"] = df["sign"].map(sign2idx)

    # Signer-independent split: ensures no participant appears in both train and val,
    # matching the Kaggle evaluation setup. Falls back to stratified random split
    # if participant_id is absent.
    if "participant_id" in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df["participant_id"]))
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        print(
            f"Signer-independent split: "
            f"{train_df['participant_id'].nunique()} train signers, "
            f"{test_df['participant_id'].nunique()} val signers"
        )
    else:
        train_df, test_df = train_test_split(
            df, test_size=0.1, stratify=df["sign"], random_state=42
        )

    train_cache = str(Path(cache_dir) / "train") if cache_dir else None
    test_cache = str(Path(cache_dir) / "test") if cache_dir else None

    train_dataset = ASLDataset(
        train_df, data_dir, cache_dir=train_cache, lmdb_path=lmdb_path,
        max_frames=max_frames, augment=True,
    )
    test_dataset = ASLDataset(
        test_df, data_dir, cache_dir=test_cache, lmdb_path=lmdb_path,
        max_frames=max_frames, augment=False,
    )

    worker_kwargs = (
        dict(persistent_workers=True, prefetch_factor=2)
        if num_workers > 0
        else {}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketBatchSampler(train_dataset.lengths, batch_size),
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        **worker_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=BucketBatchSampler(test_dataset.lengths, batch_size),
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True,
        **worker_kwargs,
    )
    return train_loader, test_loader
