"""
SyntheticPhraseDataset: pseudo-continuous signing stitched from ASL Signs clips.

A phrase is a gloss sequence from PhraseGrammar, signed by ONE signer: each
gloss is a random clip of that sign by that signer, so a change of person can
never mark a sign boundary. Phrases are built on the fly from the ASL Signs
LMDB already used for training — nothing new is stored, and every epoch draws
new phrases. Only the phrase *specs* (glosses, clip ids, transition lengths)
are drawn up front, per epoch, so batches can be bucketed by length.

Stitching, per phrase:
  1. Each clip: hand_presence() (detection flags + hold-fill), then trimmed to
     its first..last frame with any hand detected. Clips with fewer than
     min_detected detected frames are swapped for another clip of the same
     sign and signer when one exists.
  2. Training only: each clip resampled by its own tempo factor (signs vary
     in speed within a sentence).
  3. Transitions of a few frames are inserted between clips. Pose and face
     are linearly interpolated; a hand is interpolated when detected at both
     ends, otherwise held with presence 0 (exactly how a tracking gap is
     stored).
  4. Phrase-level hold-fill: a hand missing in one clip holds its nearest
     detection elsewhere in the phrase instead of sitting at the origin
     (0 = nose). Hands never detected in the whole phrase stay 0.

Known gaps vs real continuous signing: citation-form signs (no
co-articulation), synthetic transitions, no grammatical facial expressions,
and toddler vocabulary. Evaluate on real continuous signing before trusting
phrase-level numbers.

Output matches FingerspellingDataset, so collate_ctc is reused:
  coords (T, IN_FEAT) = [positions | Δ1 | presence], mask (T,), gloss ids.
"""

import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import COORDS_PER_LM, LH_START, N_LH, RH_START
from .augmentation import augment_sample, bone_scale
from .phrase_grammar import PhraseGrammar
from .preprocessing import hand_presence

_HAND_SLICES = [slice(hs, hs + N_LH * COORDS_PER_LM) for hs in (LH_START, RH_START)]


class SyntheticPhraseDataset(Dataset):
    """
    Args:
        source:      ASLDataset whose df has 'sign' (int) and 'participant_id';
                     only _load_coords() and lengths are used. Build it with a
                     large max_frames so lengths are uncapped.
        idx2sign:    class index → sign name (inverse of the sign map).
        n_phrases:   phrases per epoch.
        seed:        phrase specs depend only on (seed, epoch, index).
        augment:     per-clip tempo, bone scaling, noise / rigid shift.
        max_frames:  longer phrases are uniformly downsampled.
        random_frac: share of random sign sequences (rest: grammar templates).
        max_signs:   longest phrase in signs.
        transition:  (min, max) inserted frames between clips.
        tempo:       (min, max) per-clip resampling factor, training only.
        min_detected: clips with fewer frames with a detected hand are swapped.
        bone_aug:    (prob, hand_range, bone_range) for bone_scale(); None = off.
        holdout_frac: share of multi-sign gloss sequences (by a fixed hash of the
                     sequence) never drawn for training; validation reports them
                     separately as unseen combinations. Single signs are never held out.
    """

    def __init__(
        self,
        source,
        idx2sign: Dict[int, str],
        n_phrases: int,
        seed: int = 0,
        augment: bool = False,
        max_frames: int = 384,
        random_frac: float = 0.3,
        max_signs: int = 5,
        transition: Tuple[int, int] = (3, 8),
        tempo: Tuple[float, float] = (0.8, 1.25),
        min_detected: int = 4,
        bone_aug: Optional[Tuple[float, float, float]] = None,
        holdout_frac: float = 0.1,
    ):
        self.source = source
        self.idx2sign = idx2sign
        self.sign2idx = {s: i for i, s in idx2sign.items()}
        self.grammar = PhraseGrammar(list(self.sign2idx))
        self.n_phrases = n_phrases
        self.seed = seed
        self.augment = augment
        self.max_frames = max_frames
        self.random_frac = random_frac
        self.max_signs = max_signs
        self.transition = transition
        self.tempo = tempo
        self.min_detected = min_detected
        self.bone_aug = bone_aug
        self.holdout_frac = holdout_frac

        df = source.df
        self.clips: Dict[Tuple[str, int], List[int]] = {}
        for i, (p, s) in enumerate(zip(df["participant_id"].astype(str), df["sign"].astype(int))):
            self.clips.setdefault((p, s), []).append(i)
        self.signers = sorted({p for p, _ in self.clips})
        self.available = {
            p: {idx2sign[s] for (q, s) in self.clips if q == p} for p in self.signers
        }
        self.raw_lengths = np.asarray(source.lengths)
        self.set_epoch(0)

    # ------------------------------------------------------------------
    # Phrase specs
    # ------------------------------------------------------------------
    def held_out(self, glosses: List[str]) -> bool:
        """Fixed (seed-independent) membership of the held-out sequence set."""
        if len(glosses) < 2:
            return False
        return zlib.crc32(" ".join(glosses).encode()) % 1000 < self.holdout_frac * 1000

    def set_epoch(self, epoch: int) -> None:
        """Draw this epoch's phrases: (signer, gloss ids, english, clip ids, transitions,
        kind), kind = "seen" / "unseen" (held-out template sequence) / "random"."""
        self.epoch = epoch
        rng = np.random.default_rng([self.seed, epoch])
        specs = []
        for _ in range(self.n_phrases):
            signer = self.signers[rng.integers(len(self.signers))]
            while True:
                glosses, english = self.grammar.sample(
                    rng, self.available[signer], self.random_frac, self.max_signs
                )
                held = self.held_out(glosses)
                if not (self.augment and held):  # training never sees held-out sequences
                    break
            is_random = english == " ".join(g.upper() for g in glosses)
            kind = "random" if is_random and len(glosses) > 1 else "unseen" if held else "seen"
            ids = [self.sign2idx[g] for g in glosses]
            clips = [int(rng.choice(self.clips[(signer, s)])) for s in ids]
            trans = rng.integers(self.transition[0], self.transition[1] + 1, len(ids) - 1).tolist()
            specs.append((signer, ids, english, clips, trans, kind))
        self.specs = specs

    def lengths(self) -> List[int]:
        """Approximate frames per phrase (untrimmed clips + transitions), capped —
        for BucketBatchSampler."""
        return [
            min(int(self.raw_lengths[c].sum()) + sum(t), self.max_frames)
            for c, t in ((np.asarray(s[3]), s[4]) for s in self.specs)
        ]

    def __len__(self) -> int:
        return len(self.specs)

    # ------------------------------------------------------------------
    # Stitching
    # ------------------------------------------------------------------
    def _clip(self, rng, signer: str, sign: int, j: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trimmed (coords, presence) of one clip, swapped if poorly tracked."""
        pool = self.clips[(signer, sign)]
        for attempt in range(4):
            coords, pres = hand_presence(self.source._load_coords(j))
            det = pres.sum(1) > 0
            if int(det.sum()) >= self.min_detected or attempt == 3 or len(pool) == 1:
                break
            j = int(rng.choice(pool))
        if det.any():
            nz = torch.nonzero(det).flatten()
            coords, pres = coords[nz[0] : nz[-1] + 1], pres[nz[0] : nz[-1] + 1]
        if self.augment and coords.shape[0] >= 2:
            f = rng.uniform(*self.tempo)
            n = max(2, int(round(coords.shape[0] * f)))
            if n != coords.shape[0]:
                src = torch.linspace(0, coords.shape[0] - 1, n)
                coords = F.interpolate(coords.T[None], size=n, mode="linear", align_corners=True)[0].T
                pres = pres[src.round().long()]
                # Interpolation would blend a held hand into a real one; hold instead.
                for h, sl in enumerate(_HAND_SLICES):
                    held = pres[:, h] == 0
                    if held.any() and (~held).any():
                        coords[:, sl] = _hold_fill(coords[:, sl], ~held)
        return coords, pres

    def build(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stitched (coords (T, COORD_FEAT), presence (T, 2)) of phrase idx, no augmentation
        beyond tempo. Deterministic for a given (seed, epoch, idx) when augment=False."""
        signer, ids, _, clips, trans, _ = self.specs[idx]
        rng = np.random.default_rng([self.seed, self.epoch, idx, 1])
        segs = [self._clip(rng, signer, s, j) for s, j in zip(ids, clips)]

        parts, pres_parts, gaps = [], [], []  # gaps: (start, length) of each transition
        T = 0
        for k, (c, p) in enumerate(segs):
            parts.append(c)
            pres_parts.append(p)
            T += c.shape[0]
            if k < len(trans):
                n = trans[k]
                both = (p[-1] > 0) & (segs[k + 1][1][0] > 0)  # hand detected at both ends
                parts.append(c[-1:].expand(n, -1).clone())
                pres_parts.append(both.float().expand(n, -1).clone())
                gaps.append((T, n))
                T += n
        coords = torch.cat(parts)
        pres = torch.cat(pres_parts)

        # Phrase-level hold-fill: a hand absent from a whole clip takes its nearest
        # detection in the phrase rather than 0 (= at the nose).
        for h, sl in enumerate(_HAND_SLICES):
            det = pres[:, h] > 0
            if det.any() and not det.all():
                coords[:, sl] = _hold_fill(coords[:, sl], det)

        # Transitions: interpolate between the frames on either side. Hands only
        # where flagged present (both ends detected); held hands stay held.
        for start, n in gaps:
            a, b = coords[start - 1], coords[start + n]
            w = torch.arange(1, n + 1, dtype=coords.dtype)[:, None] / (n + 1)
            interp = a + w * (b - a)
            keep = coords[start : start + n].clone()
            coords[start : start + n] = interp
            for h, sl in enumerate(_HAND_SLICES):
                if not bool(pres[start, h] > 0):
                    coords[start : start + n, sl] = keep[:, sl]
        return coords, pres

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        coords, pres = self.build(idx)
        ids = self.specs[idx][1]

        if self.augment:
            if self.bone_aug and self.bone_aug[0] > 0:
                coords = bone_scale(coords, *self.bone_aug)
            coords = torch.tensor(augment_sample(coords.numpy()), dtype=torch.float32)
            for h, sl in enumerate(_HAND_SLICES):
                if not pres[:, h].any():  # never detected in the phrase → stays exactly zero
                    coords[:, sl] = 0.0

        if coords.shape[0] > self.max_frames:
            keep = torch.linspace(0, coords.shape[0] - 1, self.max_frames).long()
            coords, pres = coords[keep], pres[keep]

        vel = torch.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]
        x = torch.cat([coords, vel, pres], dim=-1)
        return x, torch.ones(x.shape[0], dtype=torch.bool), list(ids)

    def describe(self, idx: int) -> Dict:
        signer, ids, english, clips, _, kind = self.specs[idx]
        return {"signer": signer, "glosses": [self.idx2sign[i] for i in ids],
                "english": english, "clips": clips, "kind": kind}


def _hold_fill(block: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Frames where present is False take the nearest earlier present frame
    (the first present frame for leading ones)."""
    t = torch.arange(block.shape[0])
    src = torch.cummax(torch.where(present, t, -1), 0).values
    src[src < 0] = int(t[present][0])
    return block[src]


def phrase_datasets(train_source, val_source, idx2sign: Dict[int, str], n_train: int,
                    n_val: int, seed: int = 0, **kw) -> Tuple["SyntheticPhraseDataset", "SyntheticPhraseDataset"]:
    """Train phrases from training signers (augmented, redrawn per epoch via
    set_epoch) and a fixed set of validation phrases from held-out signers."""
    aug_only = {"bone_aug", "tempo"}
    train = SyntheticPhraseDataset(train_source, idx2sign, n_train, seed, augment=True, **kw)
    val = SyntheticPhraseDataset(val_source, idx2sign, n_val, seed + 1000, augment=False,
                                 **{k: v for k, v in kw.items() if k not in aug_only})
    return train, val
