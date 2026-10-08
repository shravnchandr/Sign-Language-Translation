# Experiment Log — LandmarkConformer (cnn_transformer)

Tracking every training run, the config used, and the result. Goal: 250-class ASL sign recognition on Google ASL Signs dataset (signer-independent val split).

**Baseline to beat:** 1st-place Kaggle solution — 0.8929 accuracy (1D CNN + Transformer).

---

## Run 001 — Baseline LandmarkConformer
**Date:** ~2025-04 (pre-session)  
**Hardware:** RunPod — NVIDIA A40  
**Config:** d_model=512, n_layers=6, n_heads=8, dropout=0.1, params=~49M  
**Face landmarks:** 134 (eyebrows + mouth + nose + eyes + face_oval)  
**Augmentation:** flip, noise, time-stretch, rotation, finger dropout, mixup  
**Phase 2:** light augmentation (heavy_augment=False, use_mixup=False)  
**Timing:**
- Throughput (before vectorization): ~4.05 it/s
- Throughput (after vectorization):  ~11.4 it/s
- Per-epoch time: unknown (not logged)
- Total training time: unknown (not logged)

**Result:**
- Best val acc (deterministic): **0.7462**
- Best val acc (TTA):           **0.7463**

**Analysis:**
- Severe overfitting: train acc ~99.9%, val acc 72–74% → 25% gap
- Phase 2 "gentle augmentation trap": turning off augmentation caused model to memorize 18 training signers
- Model was too large (49M params) for the dataset size
- Face landmarks included identity-encoding features (nose, eyes, face_oval)
- Mixup cross-dominance bug: paired lh-dominant with rh-dominant samples

---

## Changes Applied After Run 001

### Architecture
| Change | Rationale |
|--------|-----------|
| d_model 512→256, n_layers 6→4, n_heads 8→4 | Reduce overfitting; ~6.5M params vs 49M |
| Remove nose/eyes/face_oval from face landmarks | Remove signer-identity features (134→56 face lms) |
| Add WristNormalization (dual-stream) | lm0=location, lm1-20=shape; rotation invariant for hand shape |
| Add stochastic depth (drop_path_max=0.1) | Force each layer to be independently useful |
| Add GRL signer-invariance (grl_lambda=0.1) | Adversarial loss forces features to discard signer identity |
| Increase dropout 0.1→0.2 | More regularisation |

### Training
| Change | Rationale |
|--------|-----------|
| Phase 2: keep heavy_augment=True, use_mixup=True | Prevent memorisation during fine-tuning |
| Fix dominance-aware mixup pairing | Avoid cross-dominance mixed tensors for HandDominanceModule |
| Fix OneCycleLR steps (÷ accumulation_steps) | Schedule was 4× too slow |

### Data / Preprocessing
| Change | Rationale |
|--------|-----------|
| LMDB cache with version hash | Fast I/O on RunPod network storage |
| Signer-independent split (GroupShuffleSplit) | Match Kaggle evaluation protocol |

---

## Run 002 — Regularised LandmarkConformer
**Date:** 2026-05-02  
**Hardware:** RunPod — NVIDIA A40  
**Config:**
- d_model=256, n_layers=4, n_heads=4, dropout=0.2, params=~6.5M
- drop_path_max=0.1, grl_lambda=0.1
- Face landmarks: 56 (eyebrows + mouth only)
- Phase 1: 80 epochs (OneCycleLR, accumulation×4), Phase 2: 20 epochs (CosineAnnealing, accumulation×4)
- heavy_augment=True, use_mixup=True throughout

**Timing:**
- Throughput: ~12–13 it/s (LMDB cache warm); first epoch slow due to cache build (~8m 50s)
- Phase 1: 2h 19m 32s (80 epochs, avg 1m 44s/epoch)
- Phase 2: 33m 35s (20 epochs, avg 1m 40s/epoch)
- Total:   2h 53m 08s

**Result:**
- Best val acc (deterministic): **0.7555** (Phase 1 epoch 74)
- Best val acc (TTA):           **0.7569**
- Train acc at convergence:     ~0.42

**Analysis:**
- *(2026-10-08 correction: train acc here is measured on augmented, mixup-mixed inputs and is not
  comparable to val — this bullet does not establish anything about overfitting. See "Review
  corrections".)*
- Overfitting eliminated: train acc ~0.42 vs val acc ~0.755 — inverse of Run 001's 99.9% / 74.6% pattern. Heavy aug + mixup + smaller model all contributing.
- Phase 2 didn't improve on Phase 1 best (P2 peak 0.7551 vs 0.7555); warmdown acted as polishing, not exploration. `best_final.pth` is the Phase 1 checkpoint.
- Phase 2 train loss climbed 0.40 → 0.65 over the first ~6 epochs — expected warm restart at LR=1e-4 after Phase 1 finished at LR~8e-6. Val acc held steady throughout.
- TTA gain negligible (+0.14%) — model predictions stable under augmentation; stochastic variance already low.
- Disc acc not logged — this run predates the discriminator accuracy logging commit. GRL activity cannot be confirmed from logs alone.
- Expected 0.82–0.86; achieved 0.7555. Gap vs expectation likely reflects GRL not being verified as active, and the 0.82+ target requiring additional feature improvements (multi-scale velocity, cross-part attention).

---

## Changes Applied After Run 002

### Architecture
| Change | Rationale |
|--------|-----------|
| Multi-scale velocity: Δ1/Δ2/Δ5 per part | Δ2/Δ5 computed inside forward() from body-relative positions, divided by time delta to normalize units |
| Equal velocity budget: vel_proj upgraded d_model//8 → d_model//4 per part | Velocity and position now get symmetric d_model budget before fusion |
| Face split: eyebrow_proj (d_model//8) + mouth_proj (d_model//8) | Grammatical and phonological face streams specialize independently |
| INCLUDE_DEPTH=True | Add z-coordinates for palm orientation signal |
| GRL Phase 2 fix: continues Ganin ramp from Phase 1 endpoint | Previously reset λ to 0 at Phase 2 start |

### Training / Tooling
| Change | Rationale |
|--------|-----------|
| Discriminator accuracy logging | Verify GRL is actively confusing discriminator (disc_acc near 1/n_signers = chance) |
| Parallel LMDB build (ProcessPoolExecutor) | ~12× faster build (~20 min vs 2+ hrs) |

---

## Run 003 — Multi-Scale Velocity + Equal Budget + Face Split + Depth
**Date:** 2026-05-03  
**Hardware:** RunPod — NVIDIA A40  
**Config:**
- d_model=256, n_layers=4, n_heads=4, dropout=0.2, params=~6.5M
- drop_path_max=0.1, grl_lambda=0.1, INCLUDE_DEPTH=True
- Face landmarks: 56 (eyebrows 16 + mouth 40), split into two projections
- Multi-scale velocity: Δ1 (dataset) + Δ2/Δ5 (in-model, /2 and /5 normalised)
- Phase 1: 100 epochs (OneCycleLR, accumulation×4), Phase 2: 20 epochs

**Timing:**
- Throughput: ~12–13 it/s
- Phase 1: 1h 41m 26s (100 epochs, avg 1m 00s/epoch)
- Phase 2: 17m 51s (20 epochs, avg 0m 53s/epoch)
- Total:   1h 59m 18s

**Result:**
- Best val acc (deterministic): **0.7432** (Phase 1 epoch ~90)
- Best val acc (TTA):           **0.7468**
- Train acc at convergence:     ~0.41
- Disc acc at convergence:      ~0.120 vs 0.0556 chance (18 signers)

**Analysis:**
- Slight regression vs Run 002 (0.7432 vs 0.7555). Most likely cause: expanded feature set (depth + more velocity + face split) requires more gradient steps to converge than 100 epochs allows; d_model=512 produced similar results suggesting information ceiling, not capacity.
- GRL confirmed active: disc acc ~12% ≈ 2× chance — feature extractor is confusing discriminator but not fully suppressing signer signal.
- Phase 2 made no improvement: Phase 2 best was 0.7390, below Phase 1's 0.7432. Root cause: Phase 2 force-resets LR to 1e-4 regardless of Phase 1 end (~2e-9) — 5 orders of magnitude jump that undoes Phase 1 fine-tuning.
- Val acc plateaued ~0.742–0.743 from epoch 85+; model fully converged.
- Key insight from capacity ablation (d_model=512 ≈ d_model=256): the ~26% error rate reflects an **information ceiling** — the features don't make all 250-sign distinctions accessible, not insufficient model capacity. The core gaps are palm orientation (monocular z is noisy) and explicit hand shape (raw XYZ buries fine-grained joint angle differences).

---

## Changes Applied After Run 003

### Architecture
| Change | Rationale |
|--------|-----------|
| Geometry stream: `_hand_geometry()` | 15 joint-angle cosines (3 per finger at MCP/PIP/DIP) + 10 fingertip pairwise distances per hand = 25 explicit hand-shape features. Invariant to wrist rotation and signer hand scale. Projected via lh_geo_proj / rh_geo_proj each d_model//8 = d_model//4 total. |
| `feat_fuse` input: 2·d_model + d_model//4 | Adds geometry budget without reducing position/velocity allocation |

### Normalization fix
| Change | Rationale |
|--------|-----------|
| `normalize_values` in preprocessing: nose→shoulder→hip→0 fallback | Previously fillna(0) silently used origin=0 for missing nose; `RobustNormalization` in model always fell to shoulder branch (nose appeared missing post-preprocessing). Single consistent path now. |
| `RobustNormalization` removed from model | Normalization done once at LMDB build time; no per-forward-pass cost |
| `_NORM_VERSION = "v2_fallback"` in `_cache_keys.py` | Auto-invalidates old LMDB on next build |

### Training
| Change | Rationale |
|--------|-----------|
| Phase 2 removed | Phase 2 never improved on Phase 1 best across two runs; LR reset to 1e-4 is the root cause. Single-phase OneCycleLR is sufficient. |

---

## Diagnosis Before Run 004 — Augmentation & Canonicalization Bugs
**Date:** 2026-10-06

The Run 003 "information ceiling" conclusion (d_model 256 ≈ 512) is also consistent with a
**corrupted-input ceiling**: several augmentations destroyed the fine hand-shape signal the
model needs. Measured on real MediaPipe hands (fingerspelling shard, ~91k hand frames):
finger segment ≈ 0.053, per-frame motion ≈ 0.011, joint-angle-cosine spread across hand
shapes ≈ 0.28.

| Issue | Measured effect | Fix |
|-------|-----------------|-----|
| `augment_sample` shift was per feature column (±0.02 per joint, not rigid) | joint-angle cos changed by 0.22 (≈ 80% of natural spread), 50% of samples | One rigid offset per axis |
| Train-loop noise σ=0.01 on every feature, every frame | joint-angle cos changed by 0.19; noise ≈ entire Δ1 velocity signal | Removed (σ=3e-3 dataset noise before velocity kept → 0.045) |
| `HandDominanceModule` swapped slots without mirroring | Dominant slot mixed left- and right-hand anatomy (and palm normal sign) across signers | Full horizontal mirror via `MIRROR_PERM` |
| `random_flip` swapped hands only | Pose limbs / eyebrows / lip corners not mirrored → impossible training poses never seen at val | Removed — canonicalization mirrors every input, so a flip is undone except on near-ties |
| Never-detected hand stored as 0 = nose origin; FS LMDB writes 0 per missing frame | Missing hand indistinguishable from a hand at the face; hands missing in 30–70% of frames | Per-frame presence flags (`hand_presence`, derived from stored data — no LMDB rebuild) fed to `dist_proj`; FS gaps held like ASL |
| `time_stretch` cropped back to T | Last 10–23% of nearly every sign removed whenever stretch > 1 | Return longer batch |
| GRL λ applied twice | Backbone got −λ² = −0.01 (not −0.1) → disc acc 2× chance in Run 003 | `loss = sign_loss + adv_loss` |
| Train acc scored vs `y_a` under mixup | Logged ~0.42 was ≈ half the real value — "no overfitting" conclusion unverified | Mixup-weighted accuracy |

**Evaluation caveat:** the GroupShuffleSplit val set is only **3 signers** (2044, 37779, 53618;
14,248 samples). Run-to-run differences of ~1 pt (0.7555 vs 0.7432) are within signer-sampling
noise. The 0.8929 Kaggle score is a hidden-test-set (ensemble) number, not directly comparable.

**Re-run implications:** canonicalization and the presence inputs change the input
distribution — existing `backbone_best.pth` checkpoints should be re-pre-trained (an old
backbone still loads: `dist_proj` changed shape and is re-initialised).
LMDBs do **not** need rebuilding. GRL is now ~10× stronger on the
backbone at the same `--grl-lambda`; watch disc acc vs chance and sign val acc.

---

## Run 004 — Geometry Stream + Normalization Fix + Augmentation Fixes
**Date:** 2026-10-07
**Hardware:** RunPod — NVIDIA A40 (no persistent volume; `setup_runpod.sh`)
**Config changes vs Run 003:**
- Geometry stream (joint angles + fingertip distances + palm normal) and distance stream with hand presence flags
- Proper preprocessing normalization (nose→shoulder→hip fallback)
- All fixes from "Diagnosis Before Run 004" (rigid shift, no loop noise, full-mirror canonicalisation, no flip, no time-stretch crop, GRL λ once, mixup-weighted train acc)
- d_model=256, n_layers=4, n_heads=4, ~6.5M params; Phase 1 only, 80 epochs, `--num-workers 8`, fp16 AMP

**Timing:** ~16–17 it/s after a cold-cache first epoch (4.6 it/s) — avg 1m23s/epoch.
Early-stopped at epoch 70 (20 epochs without improvement after the NaN); total 1h 37m.

**Result:**
- Best val acc (deterministic): **0.7590** (epoch 50) — above Run 002's 0.7555 (within 3-signer noise)
- Train acc at epoch 50: 0.75 (mixup-weighted, now trustworthy) — no overfitting; train < val under heavy aug
- Disc acc ~0.10 vs 0.056 chance — stable, ~2× chance as in Run 003
- Best val acc (TTA): **0.7610**

**Curve** (val acc still rising when the NaN hit — the LR-annealing phase where Run 002
peaked, epoch 74, was lost):

| Epoch | Train acc | Val acc | Disc acc |
|------:|----------:|--------:|---------:|
| 5  | 0.171 | 0.330 | 0.106 |
| 10 | 0.418 | 0.561 | 0.112 |
| 20 | 0.575 | 0.659 | 0.109 |
| 30 | 0.657 | 0.729 | 0.107 |
| 40 | 0.705 | 0.739 | 0.104 |
| 45 | 0.732 | 0.741 | 0.100 |
| **50** | 0.752 | **0.759** | 0.096 |
| 55–70 | 0.767 → 0.794 | 0.0042 (NaN-poisoned BN) | 0.098 → 0.094 |

NaN-loss epochs: 55 and 60. Disc acc trended down 0.112 → 0.094 (chance 0.056): GRL is
slowly working. Train ≈ val at epoch 50 — no overfitting.

**Failure at epoch 55:** one batch produced a NaN loss under fp16 AMP. GradScaler skipped
the step (weights intact, train acc kept rising to 0.78) but the NaN forward had already
updated the Conformer conv-module BatchNorm running stats → eval mode (running stats)
output NaN → **val acc collapsed to 0.0042 (chance) for every later epoch**, while train
mode (batch stats) was unaffected. Reproduced on CPU. Best checkpoint (epoch 50) predates
it, so the reported result is valid, but epochs 51–80 could not improve it.

**Fixes for Run 005:** bf16 autocast on Ampere+ (fp32 exponent range — no overflow, no loss
scaling); non-finite-loss guard in `train_epoch` restores BN buffers from a pre-forward
snapshot and skips the backward, with a per-epoch warning count. The NaN's root cause
under fp16 is unconfirmed (GRL now acts at full strength — a candidate); if warnings
persist under bf16 it is a real numerical bug, not overflow.

---

## Run 005 — Run 004 config + bf16 / non-finite guard / split loss logging
**Date:** 2026-10-07
**Hardware:** RunPod — NVIDIA A40
**Config:** identical to Run 004 except bf16 autocast (no GradScaler), the non-finite
batch guard, and separate sign/adv loss logging. Default val split; 80 epochs;
`--checkpoint-dir checkpoints/run005`.

**Timing:** avg 1m 18s/epoch; 80 epochs in 1h 44m (no early stop).

**Result:**
- Best val acc (deterministic): **0.7665** (epoch 79)
- Best val acc (TTA): **0.7669**
- No NaN epochs, **0 non-finite batches skipped** — bf16 alone removed the Run 004 failure

| Epoch | Sign loss | Adv loss | Train acc | Val acc | Disc acc |
|------:|----------:|---------:|----------:|--------:|---------:|
| 10 | 2.815 | 2.729 | 0.433 | 0.555 | 0.111 |
| 20 | 2.200 | 2.736 | 0.582 | 0.682 | 0.109 |
| 30 | 1.907 | 2.749 | 0.657 | 0.718 | 0.108 |
| 40 | 1.736 | 2.765 | 0.705 | 0.738 | 0.103 |
| 50 | 1.550 | 2.781 | 0.751 | 0.750 | 0.099 |
| 60 | 1.398 | 2.795 | 0.787 | 0.762 | 0.094 |
| 70 | 1.340 | 2.794 | 0.802 | 0.765 | 0.096 |
| **79** | 1.383 | 2.798 | 0.799 | **0.7665** | 0.092 |

**Analysis:**
- **Seed noise ≈ 1 pt.** At epoch 50 Run 004 had 0.7590 and Run 005 0.7499 with the
  same recipe — differences under ~1 pt on this split are not evidence. Most of Run 005's
  gain over Run 004 is the LR-annealing phase it got to use (+1.7 pt from epoch 50 to 79).
- **Plateau:** +0.3 pt over the last 20 epochs at this recipe/length.
- **GRL working, slowly:** adv loss 2.73 → 2.80 (chance ln 18 = 2.89), disc acc 0.111 → 0.092.
  *(Correction: low discriminator accuracy — measured against original ids under mixup — does not
  show signer invariance; needs a GRL on/off ablation and a frozen-embedding signer probe.)*
- **Still underfitting:** train 0.80 ≈ val 0.77 *(correction: not comparable — augmented
  mixup-weighted train acc vs clean val; use the new "Clean Train" metric)* — capacity is not being used; next levers are
  training length and lighter regularisation (see 1st-place comparison below), measured
  with `--val-fold` / per-signer accuracy.

---

## Signer-fold validation & per-signer diagnostics (2026-10-07)
**Setup:** Run 005 recipe, `--val-fold k` (GroupKFold by signer, 7 folds × 3 signers), per-signer val
accuracy logged every epoch. Folds 0–1 run; fold 1 was interrupted at epoch 33/80.

| Split | Val signers | Best val | Per-signer |
|---|---|---|---|
| default (Run 005) | 2044, 37779, 53618 | 0.7665 | (not logged) |
| fold 0 | 34503, 49445, 62590 | **0.6680** (TTA 0.6674) | 0.557 / 0.654 / 0.794 |
| fold 1 (ep 33, partial) | 27610, 37055, 61333 | ~0.70 | 0.659 / 0.687 / 0.789 |

- **Signer variance dominates:** ~24 pt spread inside fold 0; the default split scores ~10 pt above fold 0 with
  the same recipe — it happens to hold the shortest-clip signers (median 9–16 frames). Fold 0 also shows a
  14 pt train–val gap (0.81 vs 0.67) that the default split did not.
- **`signer_diagnostics`** (500 clips/signer): every signer effectively uses one hand (the other is almost never
  detected; `dom_ratio` ≈ 1.0). Google clips end exactly at the last hand frame (`idle_trail` = 0 for all).
- **Spearman ρ with val acc over the 6 signers:** `no_hand` (frames with no hand detected) **−0.94**,
  `shoulder_w` −0.94, clip length −0.83, speed +0.20. `no_hand` ranks the six almost exactly and explains
  61333 (long clips but well tracked → 0.789), which broke the clip-length-only hypothesis (predicted to be
  fold 1's weakest; it was its strongest). n=6 and fold 1 partial — leads, not conclusions.
- Clip lengths over all 94,477 samples: median 22, 95th pct 135, 5.6% > 128 (subsampled), 0.3% > 256;
  highest over-128 share for 49445 (17.6%).

### Fold 0 + per-clip time stretch (`--stretch-mode sample --stretch-min 0.5 --stretch-max 2.0 --stretch-prob 0.8`)
| | Best | TTA | avg ep 61–70 | avg ep 71–80 |
|---|---|---|---|---|
| pooled | **0.6802** (+1.2) | 0.6786 (+1.1) | +1.3 | +1.1 |
| 34503 (slowest signer) | 0.5773 (+2.0) | +2.0 | +2.3 | +1.9 |
| 49445 | 0.6614 (+0.7) | +0.6 | +1.1 | +1.1 |
| 62590 | 0.8032 (+0.9) | +0.7 | +0.5 | +0.5 |

Consistent small gain on every signer and every summary, largest for the slowest signer; single seed vs
single seed (~1 pt run-to-run noise) so modest but likely real. Cost: 1m45s vs 1m21s per epoch (+29%).
Duration is a minor factor — it does not close 34503's ~22 pt gap to 62590. Next: hand dropout on top
(`--hand-drop-prob 0.5`), compared against this run.

### Fold 0 + stretch + hand dropout (`--hand-drop-prob 0.5`, span 0.1–0.4)
| | Baseline | Stretch | Stretch + drop | Δ drop vs stretch |
|---|---|---|---|---|
| best, pooled | 0.6680 | **0.6802** | 0.6774 | −0.3 |
| TTA, pooled | 0.6674 | 0.6786 | 0.6777 | −0.1 |
| avg ep 71–80, pooled | 0.6629 | 0.6743 | 0.6743 | ±0.0 |
| 34503 (avg 71–80) | 0.5521 | 0.5708 | 0.5674 | −0.3 |
| 49445 (avg 71–80) | 0.6458 | 0.6567 | 0.6658 | +0.9 |
| 62590 (avg 71–80) | 0.7919 | 0.7964 | 0.7901 | −0.6 |

**Null result.** Simulated tracking gaps do not help the signers with the most real gaps (34503 −0.3);
only 49445 (highest `no_hand`) moves, within noise. Train acc drops slightly (0.795 → 0.784). The
`no_hand` correlation is real but not addressable by robustness training: when MediaPipe loses the hand
(per prior work, mostly hand–face and hand–hand contact) the discriminative information is absent from
the input. *(Correction: this is a hypothesis, not a conclusion — six signers and one augmentation
run cannot separate tracking quality from duration, framing, signing style or the depth artifact
found afterwards.)* Hand dropout is
dropped from the recipe; current best = Run 005 recipe + per-clip stretch (fold 0: 0.6802).

---

## Review corrections & fixes (2026-10-08)
An external code review raised five pipeline issues and five over-strong conclusions. All
measurable claims reproduced locally (Run 005 checkpoint, local parquets):

| Issue | Measured | Fix (commit) |
|---|---|---|
| Nose-z subtracted from hand/face z (different MediaPipe depth frames) | raw wrist z 0.000 → stored 1.46–1.53; **52–82% of wrist velocity energy** is this artifact; hand–nose distance dominated by it | xy for dominance / hand–nose distance / shoulder scale; wrist z → 0; face z re-centred per frame; finger z kept (= raw wrist-relative z to 1e-6) (`70e651a`) |
| Finger dropout zeroed fingers before wrist subtraction → finger = −wrist | code | dropped finger collapses onto its wrist → exactly 0 in wrist frame (`5cf402c`) |
| Never-detected hands got noise/shift geometry in training only | code | re-zeroed after `augment_sample`; TTA noise skips them (`5cf402c`) |
| Batch stretch & TTA interpolated Δ1 with positions, and blended short clips' last frames with padding | 2× stretch kept Δ1 at 1.0 while positions moved 0.5 | all resampling via `_resample_prefix`: valid frames only, Δ1 rebuilt (`5cf402c`) |
| Padding leaked into depthwise conv; BatchNorm counted padded frames | +10 masked frames changed logits by up to **0.165** (6-frame clip) | re-mask before depthwise conv + `MaskedBatchNorm1d`; now ≤ 1e-3 (`b14f75e`) |
| `forward()` mutated its input | same tensor twice → max Δlogit **7.8** | clone at the model boundary (`b14f75e`) |
| No seeding; focal modulation from weighted smoothed CE; only augmented train acc | code; class counts 299–415 | `--seed` (default 42), `--loss ce`, `--train-eval-size` "Clean Train" metric (`b62a3e3`) |

**Corrected conclusions** (marked inline above):
- "No overfitting" / "underfitting" claims compared augmented, mixup-weighted train acc with clean val —
  unsupported. Use Clean Train.
- "GRL working" (low disc acc) — unsupported; needs on/off ablation + clean signer probe.
- Hand-dropout null result ≠ proof of an extraction ceiling — a hypothesis.
- Signer diagnostics "hard signers sign at normal per-frame speed" — that `speed` used wrist xyz, mostly
  the z artifact; recomputed in xy from now on. The duration/no-hand correlations used other columns
  and stand, but remain n=6.
- **All runs up to and including fold0_stretch_handdrop carry the depth artifact and the other bugs.**
  Their relative comparisons (same bugs on both sides) are informative, but the next seeded run with all
  fixes is a new baseline, not directly comparable to the 0.668 / 0.680 fold-0 numbers.

### Second review round (2026-10-08)
Reproduced and fixed: padding-dependent `dom_ratio` (0.998 → 0.969 with 100 pad frames; 0 for no
motion), per-signer summary parser regression (introduced by the Clean Train line), TTA noise breaking
Δ1, batch membership fixed across epochs, mixup on raw clips able to flip dominance, single-source
mixup tails, discriminator accuracy ignoring mixup weights, class weights from the full CSV.
Ablation flags added so the open questions can be **measured**: `--loss ce`, `--mixup-prob`,
`--finger-drop-prob`, `--zero-parts face,pose`, `--no-depth`; GRL off = `--grl-lambda 0`.
Open (needs runs, not code): focal vs CE, GRL on/off + frozen-feature signer probe, mixup / finger
dropout value, face / pose / depth value, presence-heuristic accuracy on sparse clips (raw masks were
not kept in the LMDB).

---

## Pending Ideas (not yet implemented)

### High priority
| ID | Idea | Expected impact | Notes |
|----|------|----------------|-------|
| 4A | Pre-training on fingerspelling data | High | Leverage unlabeled data for better hand-shape representations; fingerspelling has 30 distinct hand configs forcing fine-grained learning |

### Medium priority
| ID | Idea | Expected impact | Notes |
|----|------|----------------|-------|
| 1B | Cross-part attention (hand↔pose) | Medium | Spatial relationship between hand position and body |
| 4B | INCLUDE_DEPTH ablation | Low-medium | Confirm whether z helps or hurts; noisy monocular depth vs. palm orientation signal |

### Low priority / explored
| ID | Idea | Status | Notes |
|----|------|--------|-------|
| 2A | Multi-Scale Velocity (Δ2/Δ5) | Done (Run 003) | Implemented; contributed to run but converged slightly lower — may need more epochs |
| 2B | Fingertip distance features | Done (Run 004 prep) | Implemented as part of geometry stream |
| 1A | GCN stem | Skipped | Per-part projections already encode topology; GCN adds minor edge-level priors |
| 3B | STN (Spatial Transformer) | Skipped | WristNorm + geometry stream covers same invariance more cheaply |

---

## Notes on Kaggle Leaderboard Context

| Rank | Score | Key techniques (public info) |
|------|-------|------------------------------|
| 1st  | 0.8929 | 1D CNN + Transformer, ensemble (details below) |

The 0.8929 is an ensemble on Kaggle's hidden test set — not comparable to a single model on
our 3-signer val split. Whether the hidden test signers overlap the 21 training signers is
not stated in anything retrievable here (the Kaggle data page is JS-rendered); check the
competition's Data tab / host posts in a browser.

### 1st-place recipe — read from the published notebook (2026-10-07)
Source: `ISLR_1st_place_Hoyeol_Sohn.ipynb` in
github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution
(write-up: kaggle.com/competitions/asl-signs/discussion/406684).

| | 1st place | Ours (Run 004/005) |
|---|---|---|
| Model | stem Dense+BN → [3× Conv1DBlock (k=17) + Transformer] ×2, dim 192 (4× model: dim 384, ×4) | 4× Conformer block (k=31), d_model 256 |
| Inputs | x, y only; 118 lms (lips 40, hands 42, nose 4, eyes 32); pos + Δ1 + Δ2; nose-mean / std normalised | x, y, z; 131 lms (hands, pose, eyebrows, lips); pos + Δ1/Δ2/Δ5 + geometry + presence |
| Max length | **384** | **128** (longer clips are subsampled) |
| Epochs | **300** (comment: 400), cosine, lr 5e-4×8, batch 512, wd 0.1 | 80, OneCycle, lr 5e-4, batch 64 (accum 4) |
| Regularisation | dropout 0.2/block, **late dropout 0.8 before head from epoch 15**, **AWP λ=0.2 from epoch 15**, CE + label smoothing 0.1 | dropout 0.2, drop-path 0.1, **mixup every batch**, focal loss + LS 0.1 + class weights, GRL |
| Augmentation | resample 0.5–1.5× (p .8), flip (p .5), affine: scale .8–1.2 / shear .15 / shift .1 / rotate 30° (p .75), temporal mask 20–40% (p .5), spatial mask (p .5) | rigid shift, noise, time stretch 0.8–1.3×, rotation 15°, finger dropout, mixup; no flip (canonicalised) |
| Validation | 5 folds (`5fold`, plus a separate `5fold_randsplit` variant — fold construction not shown) | GroupShuffleSplit 3 signers; now `--val-fold` (GroupKFold by signer) |
| Final model | `train_folds(CFG, ['all'])` × seeds 42–45 → ensemble trained on all data | single model, 18 signers |

Largest differences: training length (300 vs 80 epochs), sequence length (384 vs 128), no
mixup / no focal / no GRL but AWP + late dropout, and a 4-seed all-data ensemble.
