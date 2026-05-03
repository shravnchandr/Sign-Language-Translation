"""
FingerspellingDataset: serves (coords, mask, char_indices) triples for CTC
pre-training of AnatomicalConformer.

Reads from the LMDB built by build_fingerspelling_lmdb.py.
Tensor layout matches ASLDataset: (T, 2*COORD_FEAT) = [positions | Δ1 velocities].
"""
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import torch
from torch.utils.data import Dataset

from ..config import COORD_FEAT
from .dataset import _open_lmdb_env  # reuse the fork-safe env cache


class FingerspellingDataset(Dataset):
    """
    Each sample is one fingerspelling phrase sequence.

    Returns:
      coords:       (T, 2*COORD_FEAT)  float32  [positions | Δ1 velocities]
      mask:         (T,)               bool     True = valid frame
      char_indices: List[int]                   CTC target character indices
    """

    def __init__(
        self,
        lmdb_path: str,
        csv_path: str,
        char_to_idx: Dict[str, int],
        max_frames: int = 384,
    ):
        import pandas as pd

        self.lmdb_path   = str(Path(lmdb_path).resolve())
        self.char_to_idx = char_to_idx
        self.max_frames  = max_frames

        df = pd.read_csv(csv_path)
        self.sequence_ids   = df["sequence_id"].tolist()
        self.participant_ids = df["participant_id"].tolist()
        self.phrases        = df["phrase"].tolist()

    # ------------------------------------------------------------------
    # LMDB access (lazy, fork-safe — each worker opens its own handle)
    # ------------------------------------------------------------------
    def _get_env(self) -> lmdb.Environment:
        return _open_lmdb_env(self.lmdb_path)

    def _load_coords(self, sequence_id: int) -> torch.Tensor:
        key = f"fs:{sequence_id}".encode()
        env = self._get_env()
        with env.begin() as txn:
            val = txn.get(key)
        if val is None:
            raise KeyError(f"sequence_id {sequence_id} not found in LMDB")
        return torch.load(io.BytesIO(bytes(val)), weights_only=True)  # (T, COORD_FEAT)

    # ------------------------------------------------------------------
    # Character encoding
    # ------------------------------------------------------------------
    def _encode_phrase(self, phrase: str) -> List[int]:
        """Map phrase string → list of char indices (skipping unknown chars)."""
        return [self.char_to_idx[c] for c in str(phrase) if c in self.char_to_idx]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sid = self.sequence_ids[idx]
        coords = self._load_coords(sid)        # (T, COORD_FEAT)

        # Temporal downsampling to max_frames (same as ASLDataset)
        if coords.shape[0] > self.max_frames:
            idxs   = torch.linspace(0, coords.shape[0] - 1, self.max_frames).long()
            coords = coords[idxs]

        T = coords.shape[0]

        # Δ1 velocity (computed after downsampling so it's consistent)
        vel      = torch.zeros_like(coords)
        vel[1:]  = coords[1:] - coords[:-1]
        coords   = torch.cat([coords, vel], dim=-1)  # (T, 2*COORD_FEAT)

        mask         = torch.ones(T, dtype=torch.bool)
        char_indices = self._encode_phrase(self.phrases[idx])
        return coords, mask, char_indices


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_ctc(
    batch: List[Tuple[torch.Tensor, torch.Tensor, List[int]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a batch of variable-length sequences for CTC training.

    Returns:
      coords:         (B, T_max, 2*COORD_FEAT)
      mask:           (B, T_max)  bool
      targets:        (sum(target_lengths),)  concatenated char indices
      input_lengths:  (B,)  valid frame counts
      target_lengths: (B,)  phrase character counts
    """
    coords_list, mask_list, targets_list = zip(*batch)
    B      = len(coords_list)
    T_max  = max(c.shape[0] for c in coords_list)
    D      = coords_list[0].shape[1]

    padded = torch.zeros(B, T_max, D)
    masks  = torch.zeros(B, T_max, dtype=torch.bool)
    for i, (c, m) in enumerate(zip(coords_list, mask_list)):
        T = c.shape[0]
        padded[i, :T] = c
        masks[i, :T]  = m

    input_lengths  = masks.sum(dim=1).long()
    target_lengths = torch.tensor([len(t) for t in targets_list], dtype=torch.long)
    targets        = torch.cat(
        [torch.tensor(t, dtype=torch.long) for t in targets_list]
    )
    return padded, masks, targets, input_lengths, target_lengths


# ---------------------------------------------------------------------------
# Helper: load char map and return (char_to_idx, blank_idx)
# ---------------------------------------------------------------------------

def load_char_map(data_dir: str) -> Tuple[Dict[str, int], int]:
    char_map_path = Path(data_dir) / "character_to_prediction_index.json"
    with open(char_map_path) as f:
        char_to_idx = json.load(f)
    blank_idx = len(char_to_idx)  # blank = one past the last real char
    return char_to_idx, blank_idx
