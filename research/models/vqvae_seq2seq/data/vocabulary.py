"""Gloss vocabulary management for sign language translation."""

import json
from typing import Dict, List, Optional, Set
from pathlib import Path
from collections import Counter


class GlossVocabulary:
    """
    Manages gloss vocabulary for sign language translation.

    Handles:
    - Building vocabulary from datasets
    - Special tokens (PAD, BOS, EOS, UNK)
    - Index <-> gloss mapping
    - Vocabulary persistence
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(
        self,
        glosses: Optional[List[str]] = None,
        min_count: int = 1,
    ):
        """
        Args:
            glosses: Optional list of glosses to initialize vocabulary
            min_count: Minimum occurrence count for a gloss to be included
        """
        self.min_count = min_count
        self._gloss_to_idx: Dict[str, int] = {}
        self._idx_to_gloss: Dict[int, str] = {}

        # Initialize with special tokens
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self._gloss_to_idx[token] = i
            self._idx_to_gloss[i] = token

        if glosses is not None:
            self.add_glosses(glosses)

    @property
    def pad_idx(self) -> int:
        return self._gloss_to_idx[self.PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self._gloss_to_idx[self.BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self._gloss_to_idx[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self._gloss_to_idx[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self._gloss_to_idx)

    def __contains__(self, gloss: str) -> bool:
        return gloss in self._gloss_to_idx

    def add_glosses(self, glosses: List[str]) -> None:
        """Add glosses to vocabulary, respecting min_count."""
        # Count occurrences
        counts = Counter(glosses)

        # Add glosses that meet threshold
        for gloss, count in counts.items():
            if count >= self.min_count and gloss not in self._gloss_to_idx:
                idx = len(self._gloss_to_idx)
                self._gloss_to_idx[gloss] = idx
                self._idx_to_gloss[idx] = gloss

    def gloss_to_idx(self, gloss: str) -> int:
        """Convert gloss to index, returns UNK for unknown glosses."""
        return self._gloss_to_idx.get(gloss, self.unk_idx)

    def idx_to_gloss(self, idx: int) -> str:
        """Convert index to gloss."""
        return self._idx_to_gloss.get(idx, self.UNK_TOKEN)

    def encode(
        self, glosses: List[str], add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """
        Encode a sequence of glosses to indices.

        Args:
            glosses: List of gloss strings
            add_bos: Whether to prepend BOS token
            add_eos: Whether to append EOS token

        Returns:
            List of indices
        """
        indices = [self.gloss_to_idx(g) for g in glosses]

        if add_bos:
            indices = [self.bos_idx] + indices
        if add_eos:
            indices = indices + [self.eos_idx]

        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """
        Decode a sequence of indices to glosses.

        Args:
            indices: List of indices
            remove_special: Whether to remove special tokens

        Returns:
            List of gloss strings
        """
        glosses = [self.idx_to_gloss(i) for i in indices]

        if remove_special:
            special_set = set(self.SPECIAL_TOKENS)
            glosses = [g for g in glosses if g not in special_set]

        return glosses

    def get_all_glosses(self, include_special: bool = False) -> List[str]:
        """Get all glosses in vocabulary."""
        if include_special:
            return list(self._gloss_to_idx.keys())
        return [g for g in self._gloss_to_idx.keys() if g not in self.SPECIAL_TOKENS]

    def save(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            "gloss_to_idx": self._gloss_to_idx,
            "min_count": self.min_count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GlossVocabulary":
        """Load vocabulary from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        vocab = cls(min_count=data.get("min_count", 1))
        vocab._gloss_to_idx = data["gloss_to_idx"]
        vocab._idx_to_gloss = {int(v): k for k, v in data["gloss_to_idx"].items()}

        return vocab

    @classmethod
    def from_sign_to_prediction_map(cls, path: str) -> "GlossVocabulary":
        """
        Create vocabulary from a sign_to_prediction_index_map.json file.

        This is the format used by the Kaggle competition.
        """
        with open(path, "r") as f:
            sign_map = json.load(f)

        # Sort by index to maintain order
        sorted_signs = sorted(sign_map.items(), key=lambda x: x[1])
        glosses = [sign for sign, _ in sorted_signs]

        return cls(glosses=glosses)

    def merge(self, other: "GlossVocabulary") -> "GlossVocabulary":
        """
        Merge another vocabulary into this one.

        Returns a new vocabulary containing all glosses from both.
        """
        all_glosses = self.get_all_glosses() + other.get_all_glosses()
        return GlossVocabulary(glosses=all_glosses, min_count=1)


def build_combined_vocabulary(
    dataset_paths: List[str],
    output_path: Optional[str] = None,
) -> GlossVocabulary:
    """
    Build a combined vocabulary from multiple sign_to_prediction_index_map.json files.

    Args:
        dataset_paths: List of paths to sign mapping JSON files
        output_path: Optional path to save the combined vocabulary

    Returns:
        Combined GlossVocabulary
    """
    all_glosses: Set[str] = set()

    for path in dataset_paths:
        with open(path, "r") as f:
            sign_map = json.load(f)
        all_glosses.update(sign_map.keys())

    vocab = GlossVocabulary(glosses=sorted(all_glosses))

    if output_path:
        vocab.save(output_path)
        print(f"Saved vocabulary with {len(vocab)} tokens to {output_path}")

    return vocab
