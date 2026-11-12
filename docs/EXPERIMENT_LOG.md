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

### Fold 0, all review fixes + stretch, seed 42 (`fold0_fixed`) — new baseline
| | Stretch (pre-fix) | Fixed | Δ |
|---|---|---|---|
| best, pooled | 0.6802 | **0.6771** | −0.3 |
| TTA | 0.6786 | 0.6775 | −0.1 |
| avg ep 71–80 | 0.6743 | 0.6738 | ±0.0 |
| 34503 / 49445 / 62590 (best) | 0.577 / 0.661 / 0.803 | 0.574 / 0.656 / 0.803 | ≤ 0.5 |

Same final accuracy: the fixes sped up early learning (+9 pt at epochs 8–10) but not the endpoint.
0 non-finite batches; 1m54s/epoch.

**Clean train (3,000 un-augmented training clips, eval mode) vs val:**

| epoch | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|
| clean train | 0.684 | 0.813 | 0.884 | 0.924 | 0.950 | 0.963 | 0.970 | 0.970 |
| val | 0.550 | 0.612 | 0.647 | 0.662 | 0.669 | 0.671 | 0.670 | 0.675 |
| gap (pt) | 13.4 | 20.1 | 23.7 | 26.2 | 28.1 | 29.2 | 30.0 | 29.5 |

Best checkpoint (epoch 72): clean train **0.971** vs val 0.677 → **29.4 pt gap**. The earlier
"underfitting" reading (augmented train acc ≈ 0.80) was wrong: the model nearly memorises its 18
training signers while val plateaus from ~epoch 50. The bottleneck is signer generalisation — not
capacity, not training length (longer runs would widen the gap). Disc acc 0.142 → 0.109 (chance
0.056): the CLS sheds some signer identity, yet the gap still widens.

Next (budget ≈ 2 runs): `--grl-lambda 0` (does the GRL help at all?) and stronger signer-variation
augmentation (1st place: scale 0.8–1.2, shear 0.15, shift 0.1, rotate 30° vs our 15° / small shift).
Deprioritised: 160-epoch run, `--loss ce`.

Implemented for the next two runs (both off by default, compared against `fold0_fixed`):
- `--supcon-weight 0.2` — cross-signer supervised contrastive loss (same sign, different signer =
  positive; cross-batch memory queue; unmixed second pass). Success = val ≥ +2 pt, clean-train gap
  narrower, gains on the hard signers.
- `--aug-rotate 30 --aug-shear 0.15 --aug-scale 0.2 --aug-affine-prob 0.75` — stronger signer-variation
  spatial augmentation (closer to the 1st-place recipe).
GRL-off deferred: if the contrastive term works, the GRL may be redundant and that ablation becomes
more informative.

### Fold 0 + cross-signer contrastive loss (`fold0_supcon`, `--supcon-weight 0.2`, τ 0.1, queue 4096) — stopped at epoch 37
Same config as `fold0_fixed` (seed 42) plus the contrastive term (BN running stats frozen in its
pass). Behaved as designed: 100% of anchors had a cross-signer positive from epoch 1, Con Loss
8.23 → 3.98 by epoch 37, no non-finite batches, ~2m25s/epoch (1.3×).

| avg | ep 8–12 fixed / supcon | ep 22–27 fixed / supcon | ep 28–37 fixed / supcon |
|---|---|---|---|
| val | 0.552 / 0.538 (−1.3) | 0.633 / 0.638 (+0.5) | 0.649 / 0.647 (−0.2) |
| clean train | — | 0.850 / 0.860 (+1.0) | 0.895 / 0.903 (+0.8) |
| gap (pt) | 11.5 / 13.6 (+2.1) | 21.7 / 22.2 (+0.5) | 24.6 / 25.6 (+1.0) |
| 34503 / 49445 / 62590 | −3.1 / −0.3 / −0.6 | +1.1 / +0.9 / −0.6 | −0.7 / +0.4 / −0.5 |

Best val by epoch 37: 0.660 (fixed: 0.658). **Null at this setting**, stopped by the pre-agreed
epoch-35 rule (val not above fixed, gap not narrower) to fund the spatial-affine run. The
objective is learned — the 18 training signers' executions of each sign are pulled together — but
that alignment is itself a fit to the training signers and does not transfer: clean train rises
slightly faster and the gap is slightly wider. One setting only (w 0.2, τ 0.1, single seed);
other weights untested.

### Fold 0 + stronger spatial affine (`--aug-rotate 30 --aug-shear 0.15 --aug-scale 0.2 --aug-affine-prob 0.75`) — incomplete, no result
Same config and seed as `fold0_fixed` plus the stronger rotation/shear/anisotropic-scale augmentation.
The pod ran out of RunPod credit around epoch 30–40, before the log or checkpoint was retrieved, so
there is **no final or best score**. Last numbers seen (vs `fold0_fixed` at the same epochs):
neutral through epoch 12; epochs 13–16 val +0.4, clean train −1.0, **clean-train gap 1.4 pt
narrower**, hard signer 34503 +1.4. That is the pattern the augmentation was meant to produce, but
four early epochs are within noise (the gap is still small then). **Inconclusive — not a null.**
Worth re-running on the current base (`--arch graph`) if budget allows; it targets the same
signer-variation gap as bone-length augmentation (camera angle / body proportions vs hand
proportions), so the two could be combined after each is measured alone.

### GraphConformer (`--arch graph`) — design (2026-11-09); result in the `fold0_graph` entry below
The old `research/models/st_gcn/` pipeline was not salvageable as a comparison: its skeleton graph
was indexed for a face → LH → pose → RH layout while its data was LH → pose → RH → face (almost every
edge on the wrong joints), the mask downsampling crashed on most odd batch lengths, it used a
random by-sign split (signers shared with val), 2D positions only, and none of the input fixes above.

Instead the graph is a drop-in backbone on this pipeline (same LMDB, folds, augmentation, GRL,
logging). `GraphConformer` subclasses `LandmarkConformer` and replaces **only** the four flat hand
projections (`lh/rh_proj`, `lh/rh_vel_proj`) with:
- a **shared hand graph encoder** (21 joints, skeleton + knuckle-web adjacency plus a learned
  zero-initialised residual adjacency per layer, 3 pre-norm residual layers × 64, joint embeddings,
  flattened per joint — not pooled) over wrist-relative positions, bone vectors and Δ1/Δ2/Δ5;
  the non-dominant slot is x-negated first so both hands share weights with the same chirality;
- the **wrist location** (body-relative position + Δ1/Δ2/Δ5) kept outside the graph;
- per-slot output projections to the same `d_model//2` per hand.
Pose, face, geometry and distance streams, fusion, Conformer blocks and heads are unchanged, so the
comparison with `fold0_fixed` isolates the graph encoder. 6.68 M params (vs 6.61 M; graph 18 k);
~1.1× step time on CPU.

Checked: default `LandmarkConformer` logits bit-identical to before the refactor; padding
(+50 frames), input mutation, mirror canonicalisation all exact (Δlogit 0); mirrored hands reach
the shared encoder identically.

Run: `--arch graph --val-fold 0 --stretch-mode sample --stretch-min 0.5 --stretch-max 2.0
--stretch-prob 0.8 --checkpoint-dir checkpoints/fold0_graph` (fold0_fixed's flags + `--arch graph`).
Success = val ≥ +2 pt over 0.6771 or a narrower clean-train gap; otherwise the per-part
projections already capture what the skeleton prior adds (the 1A "GCN stem" hypothesis).
Either way, compare the per-signer ranking (34503 < 49445 < 62590) across architectures.

On the pod (bf16) the graph costs **~2m58s/epoch vs 1m54s** (1.55×, not the 1.1×
measured on CPU), so a full 80-epoch run is ~4 h. First 10 epochs: val 0.5865 vs 0.550 at epoch 10,
clean-train gap 10.8 vs 13.4 pt — early, but ahead on both.

**Next (implemented, off by default): bone-length augmentation** (`--bone-aug-prob 0.5`,
hand ±0.15, per-bone ±0.1). Hand size and finger proportions are a per-signer trait that shoulder
scaling does not remove; scaling bones along their own direction changes fingertip distances and
absolute joint positions but keeps every joint angle, so hand shape (the sign) is preserved.
Intended run: the `fold0_graph` command + `--bone-aug-prob 0.5`, compared against `fold0_graph`
(one change). Success = clean-train gap narrower with val ≥ `fold0_graph`.

**Speed: `--compile`.** It had never worked: torch 2.9.1 refuses `torch.compile` on Python 3.14 and
`train.py` fell back to eager silently. Upgraded to torch 2.10.0 (same CUDA 12.8 wheels; default
model logits within 3e-7 of 2.9.1 on CPU, so runs stay comparable). On CPU, compiled = eager to
4e-7 and nothing recompiles across lengths 19–128 or the per-epoch GRL λ. GPU speed-up unmeasured:
the bone-length run uses `--compile` and its it/s from epoch 2 on is compared with `fold0_graph`'s
~7.2 it/s (epoch 1 includes compilation). Speed only — no effect on the comparison beyond float
rounding.

**Update (2026-11-10): `--compile` crashes on the GPU.** First backward pass of `fold0_graph_bone`
(torch 2.10, bf16): `RuntimeError: (*bias): last dimension must be contiguous` in
`_scaled_dot_product_efficient_attention_backward`. `nn.MultiheadAttention` turns the key padding
mask into a broadcast (stride-0) bias; eager handles it, but the compiled graph passes it as is to
the CUDA memory-efficient attention backward, which needs a contiguous last dimension. CPU uses a
different SDPA kernel, so the local tests could not catch it. Not pursued: a fix (exclude the
attention call from compilation, or build a contiguous mask) can only be verified on a GPU, for an
unmeasured speed-up. Runs continue without `--compile`. torch stays at 2.10 (needed for compile,
harmless otherwise).

### Fold 0 GraphConformer (`fold0_graph`, `fold0_fixed` flags + `--arch graph`) — stopped at epoch 65, final extrapolated
Same config and seed as `fold0_fixed`; only the hand streams differ. ~2m57s/epoch (1.55×).
Stopped after epoch 65 (val 0.6768; log and `best_phase1.pth` = epoch 56 kept locally) to fund the bone-length run: LR had decayed to 5.9e-5 and val had been on
a plateau since ~epoch 54.

| | `fold0_fixed` | graph | Δ |
|---|---|---|---|
| val ep 10 / 20 | 0.550 / 0.612 | 0.587 / 0.635 | +3.7 / +2.3 |
| val ep 28–35 avg | ~0.645 | 0.660 | ~+1.5 |
| val ep 49–64 avg | ~0.670 | 0.679 | ~+0.9 |
| best (measured) | 0.6771 (ep 72) | **0.6859** (ep 56) | +0.9 |
| clean-train gap ep 50 / 60 | 28.1 / 29.2 pt | 27.4 / 28.9 pt | ~0.5 narrower |
| per-signer at best: 34503 / 49445 / 62590 | 0.574 / 0.656 / 0.803 | 0.583 / 0.664 / 0.813 | +0.9 / +0.8 / +1.0 |

`fold0_fixed` values between its logged epochs are interpolated. **Extrapolated final (not
measured):** `fold0_fixed` gained ~+0.6 pt of best from epoch 64 to its peak at 72 and its
epoch-71–80 average sat ~0.3 below its best; applied here → best ≈ 0.686–0.690, avg ep 71–80
≈ 0.680. No TTA number (the run was stopped before the final evaluation; TTA has matched the
deterministic score within ±0.3 in every run so far).

**Reading.** A small, steady gain (~+1 pt from epoch 35 on, single seed), below the +2 pt success
bar. It is uniform across the three val signers, does not target the hard signer, and the
clean-train gap is essentially unchanged (clean train still → 0.97): the explicit hand graph
models hand shape slightly better but does not reduce overfitting to the 18 training signers —
consistent with every other result pointing at signer variety, not architecture. Kept as the
base for the next runs (bone-length augmentation, then fingerspelling pre-training), which target
the gap directly. Comparisons with it use matching epochs (curves), since its endpoint is
extrapolated.

**What the learned adjacency added** (`best_phase1.pth`, epoch 56; symmetrised zero-initialised
residual `adj_residual` per graph layer, links not in the fixed skeleton). Mean |residual| 0.05–0.06,
max 0.27–0.33 — the same order as the fixed normalised skeleton weights (~0.31), so the layers were
substantially rewired, not left near zero.
- **Layer 0: a PIP–tip link on every finger** (two hops apart): index −0.28, middle −0.25, thumb
  MCP–tip −0.24, ring −0.23, pinky −0.22. A negative link acts as a difference (tip − PIP), i.e. a
  per-finger **curl** measure — learned, not given.
- **Layers 1–2: same-level links between neighbouring fingers** (index–middle and middle–ring PIP,
  index–middle, middle–ring and ring–pinky DIP; ±0.13–0.20) — finger **spread / adjacency**.
- The thumb tip ↔ index tip link anticipated in the design (pinch shapes) did not emerge strongly.
Reading: the residual adjacency extracts explicit hand-shape structure (curl, spread), consistent with
attributing the ~+1 pt to better hand-shape modelling — and it overlaps the hand-crafted geometry
stream (joint-angle cosines), which may now be partly redundant (untested).

**Offline re-evaluation on fold 0** (Mac CPU, fp32, both `best_phase1.pth`, 14,076 val clips, one pass
with both models per batch):

| | logged (pod, bf16) | re-run | TTA |
|---|---|---|---|
| `fold0_fixed` | 0.6771 | 0.6769 | 0.6773 |
| `fold0_graph` | 0.6859 | 0.6861 | **0.6858** |

Reproduces within 2–3 clips. TTA again adds nothing.
- **Paired test:** graph right / fixed wrong 1,030 clips, the reverse 901 → net +129; exact McNemar
  **p = 0.0036**. The two trained models differ significantly on this fold — but the test does not
  include training randomness, so "the architecture is better" still needs a second seed or fold.
  The models disagree on **26 %** of clips: a lot of churn for a net +0.9.
- **Per sign:** 129 up, 92 down, 29 unchanged (mean +1.0, sd 6.6 pt). Largest gains: thankyou
  0.56→0.78, grandma 0.55→0.75, like 0.72→0.91, child 0.43→0.59; largest losses: think 0.86→0.67,
  dryer 0.50→0.33, that 0.69→0.53, car 0.67→0.53.
- **Much of the churn is within near-identical pairs**, resolved one way and broken the other:
  duck↔goose (8 resolved / 14 introduced), wake↔awake (7 / 11), home↔kiss (7 / 6), plus dryer→dry,
  grandpa→dad. Some resolved confusions differ mainly in hand shape (thankyou vs taste, hen vs pen),
  but new ones (think→penny) cut the other way — **only weak support** for "the gain is hand shape".
- Side finding: part of the remaining error is concentrated in near-identical sign pairs —
  possibly a dataset-level ambiguity ceiling (not yet quantified).

**Ensemble `fold0_fixed` + `fold0_graph`** (average of softmax probabilities, equal weights, same
14,076 clips; offline, no retraining):

| | pooled | 34503 | 49445 | 62590 | top-5 |
|---|---|---|---|---|---|
| `fold0_fixed` | 0.6769 | 0.574 | 0.656 | 0.803 | 0.856 |
| `fold0_graph` | 0.6861 | 0.583 | 0.664 | 0.813 | 0.864 |
| **ensemble** | **0.7025** | **0.605** | **0.684** | **0.820** | **0.874** |

**First fold-0 result above 0.70**: +1.6 pt over the graph alone (net +231 clips, McNemar
p = 4e-14), +2.6 over `fold0_fixed`, and the largest gain on the hardest signer (34503 +2.2 over
graph). Averaging logits instead: 0.7041. Equal weights are also the optimum of a weight sweep
(w_graph 0.4 / 0.5 / 0.6 → 0.699 / 0.703 / 0.701), so the number is not inflated by tuning on val.
Cost: 2× inference. **Open question:** whether the gain comes from *architectural* diversity or
just from averaging two independently trained models — a 2-seed ensemble of one architecture would
separate the two (it would also give the missing seed-noise estimate).


### Fold 0 GraphConformer + bone-length augmentation (`fold0_graph_bone`, `--bone-aug-prob 0.5`) — null, stopped at epoch 32
`fold0_graph` + `--bone-aug-prob 0.5` (hand ±0.15, bone ±0.1); same seed and batch order, so the two
curves are tightly coupled and small effects would show.

| epochs | val Δ vs graph | clean-train gap graph → bone | 34503 Δ |
|---|---|---|---|
| 10–14 | +0.3 | 12.5 → 12.1 | +0.8 |
| 15–20 | +0.1 | 16.4 → 16.3 | +0.5 |
| 21–26 | +0.3 | 19.7 → 19.3 | +1.1 |
| 27–32 | −0.2 | 22.1 → 22.2 | +0.1 |

Best by epoch 32: 0.6609 (graph: 0.6665). **Null**: the gap grows at the same rate and the early
hard-signer edge vanished; stopped by the epoch-35 rule to fund fingerspelling pre-training.
Likely reason (untested): the model is already largely invariant to hand proportions — joint-angle
cosines are scale-free and fingertip distances are divided by each hand's own size — so simulated
proportions add no signer variety that matters. Low value as an ensemble member: it tracks
`fold0_graph` epoch by epoch (same seed and order), unlike `fold0_fixed` (26 % disagreement).


### Fold 0 GraphConformer + fingerspelling pre-training (`fold0_graph_fs`) — no gain, stopped at epoch 35
**Pre-training** (graph arch, seed 42, bucketed, augmented): 59,342 phrases / 84 signers, val 7,866 /
10 held-out signers; 5m30s/epoch (est. was ~8 min); CER 0.997 → 0.485 (ep 2) → 0.379 (5) → 0.331 (10)
→ **0.3098** (ep 20, best; ran all 20). Val loss flat from ~ep 12 (mild FS overfit; selection is by CER).

**Fine-tune** = `fold0_graph` flags + `--pretrained-backbone` (the standard recipe: 5 warmup epochs with
the backbone frozen, then 75 joint epochs with **backbone LR = 0.1 × head LR**). Warmup alone (frozen
FS features, new head) reached val 0.26 — real transfer. Then:

| | FS-pretrained | graph, same joint epochs | graph, same epochs |
|---|---|---|---|
| val ep 31–35 avg | **0.650** (best 0.652, ep 34) | 0.661 (ep 26–30) | 0.659 |
| clean-train gap | 16.0 pt | 20.9 pt | 24.0 pt |
| 34503 | 0.538 | 0.559 | 0.559 |
| signer-disc acc | ~0.19 | ~0.12 | ~0.12 |

At matched val (~0.65; graph ep 19/25/31) the FS gap is a little narrower (16.0 vs 17.4–23.9) and
34503 equal (0.538 vs 0.537–0.548); at ep 22 the FS gap had been the wider one. **Reading:** ~1 pt
behind at matched training, no hard-signer gain, features carry more signer identity — no support
for "84 extra FS signers fix signer generalisation" *with this recipe*. **Confound:** the 0.1× backbone
LR weakens both the backbone's adaptation and the GRL's reversed gradient on it (≈10× weaker
signer-invariance pressure), which fits the higher discriminator accuracy. Clean test still open:
re-fine-tune from the same `pretrain_fs/backbone_best.pth` with `--backbone-lr-ratio 1.0`
(fine-tune only, ~4 h). Checkpoint (0.652) kept as a candidate ensemble member — a backbone
pre-trained on other data may err differently from `fold0_fixed` and `fold0_graph`.

**Ensemble check (offline, Mac CPU, same 14,076 fold-0 val clips, equal-weight probability average):**
the FS model reproduces 0.6520 and is the *most different* member — prediction disagreement 32 % with
`fold0_fixed` and 30 % with `fold0_graph` (fixed/graph: 26 %); error overlap (both wrong / either
wrong) 0.58–0.60 vs 0.65 for fixed/graph.

| members | pooled | 34503 | 49445 | 62590 |
|---|---|---|---|---|
| graph + fs | 0.6971 | 0.592 | 0.682 | 0.819 |
| fixed + fs | 0.6920 | 0.583 | 0.682 | 0.811 |
| fixed + graph | 0.7025 | 0.605 | 0.684 | 0.820 |
| **fixed + graph + fs** | **0.7079** | **0.608** | **0.694** | **0.822** |

graph + fs beats graph alone by +1.1 (net +156 clips, McNemar p = 2e-11) although fs is 3.4 pt worse
alone; adding fs to fixed + graph: +0.5 (net +76, p = 6e-5). **Best fold-0 result so far: 0.7079.**
Revised reading: FS pre-training does not make a better single model (with this recipe) but yields a
genuinely different one — useful as an ensemble member. The fs member was stopped at epoch 34 of 80.

### VQ bottleneck (`--vq-codes`) — implemented, not yet run (2026-11-10)
Test (c) of the VQ-VAE plan: does a discrete bottleneck, trained end-to-end with classification,
help signer generalisation? The hypothesis that would make VQ useful here: snapping each frame to a
small codebook is an information bottleneck that may drop signer-specific detail — the 29-pt gap.
Kept from the old `vqvae_seq2seq/`: factorised codebooks per body part, EMA codebook updates,
dead-code reset. Dropped: its data pipeline and the seq2seq/CTC/beam-search decoder.

Design: the fusion input of each valid frame is quantised by four EMA codebooks (dominant hand,
non-dominant hand, body, face; 160/160/160/128 dims) → 4 codes per frame (≈32 bits at 256 codes),
straight-through gradient, commitment β 0.25, per-part code perplexity logged each epoch. Same
initial weights as the matching non-VQ run. Checked: default model unchanged; groups tile the fusion
input exactly; codebooks seeded from data, frozen in eval; eval output is a function of the codes
only (residual 1e-7 differences traced to two near-duplicate seed codes — now seeded from distinct
frames); `train.py` end-to-end on a synthetic fold.

Run: `fold0_graph` flags + `--vq-codes 256` (no pre-training, so it compares with `fold0_graph`).
**Reading:** gap narrower with val holding (≥ 0.6859 − 1) → quantisation helps; then the
reconstruction-trained tokenizer (b), possibly pre-trained on FS without labels, is worth building.
Val clearly lower (> 2 pt) → the bottleneck discards needed information; (b) — frozen tokens learned
for reconstruction — would be worse, so the VQ-VAE line stops there. Code perplexity far below 256
for a part means that codebook collapsed and the test is not fair for it (raise β or lower K).


### VQ run (`fold0_graph_vq`) aborted at epoch 5 — exposed a geometry-stream bug (2026-11-10)
The VQ commitment loss logged 54,745 / 15,220 / 19,133 / 11,837 for epochs 1–4 (expected ~0.02), val
0.160 at epoch 4 vs 0.290 for `fold0_graph`: the β-weighted squared error dominated every clipped
update. Not the quantiser: on real fold-0 clips the fusion features are ≤ 4.9 without augmentation.
**Cause:** `_hand_geometry` divided fingertip distances by the mean MCP distance + 1e-6. Finger
dropout (default p 0.5, in every run since Run 004) collapses dropped fingers onto the wrist; with all
four MCPs collapsed and the thumb intact the divisor is ~1e-6 → features up to **2.6e5 in 0.15 % of
training clips ≈ 9 % of batches**. The fusion LayerNorm hid it in the forward pass, but each such
batch's gradient was dominated by one corrupted clip and then clipped — roughly 1 update in 11
mostly wasted, in all runs alike (relative comparisons stand; absolute numbers may be depressed).
**Fix:** distance features are 0 when the hand scale ≤ 0.02 (detected hands: min 0.036, 0.01st pct
0.077, median 0.29 over 44k fold-0 training frames). Existing checkpoints on fold-0 val: max
|Δlogit| 3e-5, no prediction changes. With the full augmentation set the max fusion feature is 4.97
and the VQ commitment loss ~0.02. Consequence: a VQ run with the fix must be compared with a graph
run *with* the fix (`fold0_graph` had the bug), so the clean order is graph+fix, then VQ+fix.

**Graph + fix (`fold0_graph_fixed`), stopped at epoch 22 to start the VQ run:** indistinguishable from
`fold0_graph` — epochs 16–22 val 0.6353 vs 0.6359 (−0.06), clean-train gap 17.9 vs 17.8 pt, the same
per-epoch dips (same seed and order). The corrupted batches cost nothing measurable (clipping
contained them), so all earlier results stand, and the VQ + fix run is compared with `fold0_graph`
directly. VQ + fix first epochs: commitment 0.0139 / 0.0119 (was 54,745 / 15,220), codes in use
52/50/128/97 → 71/38/126/98 (dom hand / non-dom hand / body / face).

**VQ + fix (`fold0_graph_vq`, `--vq-codes 256`) — negative, stopped at epoch 17:**

| epochs | val graph → VQ | clean-train gap graph → VQ |
|---|---|---|
| 12–14 | 0.614 → 0.456 (−15.8) | 13.0 → 12.4 pt |
| 15–17 | 0.629 → 0.478 (−15.1) | 15.5 → 14.2 pt |

At matched val (~0.48; graph epochs 6–7, gap ~9 pt) the VQ gap is wider (14–15 pt): the bottleneck
discards sign-relevant information without removing signer-specific detail. Codes in use: dom hand
~118/256, body ~127, face ~108, non-dom hand collapsed to 10–25 (little information there — most
signers are one-handed — so not the cause). Commitment loss healthy (0.17). **Conclusion (planned
decision point): the VQ-VAE line stops.** The end-to-end, classification-trained bottleneck is the
most favourable case for quantisation; a frozen reconstruction-trained tokenizer (the old pipeline)
would be expected to do worse. Untested variants (bigger codebooks, several codes per part) could
narrow the loss but not rescue the hypothesis that quantisation reduces signer overfitting.


### Per-signer BatchNorm adaptation (AdaBN), offline on fold 0 (2026-11-10)
For each val signer, every BatchNorm layer's running mean/var (4 layers: the Conformer conv modules)
is replaced by the cumulative batch average over that signer's *unlabelled* clips; the model then
predicts with the adapted statistics. Fixed length-sorted batches so all models see the same order.
**Held-out protocol** (statistics from a random half of each signer's clips, accuracy on the other
half; baseline on the same half):

| | base | AdaBN | Δ | 34503 | 49445 | 62590 |
|---|---|---|---|---|---|---|
| fixed | 0.6774 | 0.6858 | +0.8 | .571→.585 | .657→.665 | .805→.809 |
| graph | 0.6855 | 0.6967 | +1.1 | .581→.597 | .662→.674 | .815→.821 |
| fs | 0.6511 | 0.6609 | +1.0 | .538→.560 | .640→.644 | .776→.780 |
| fixed+graph | 0.6990 | 0.7139 | +1.5 | .599→.623 | .680→.691 | .819→.830 |
| **fixed+graph+fs** | 0.7059 | **0.7210** | **+1.5** | .602→.626 | .696→.707 | .820→.832 |

Transductive (stats and predictions on all clips): graph 0.6861 → 0.6977, 3-model 0.7079 → 0.7210.
Same gains in both protocols → not fitting the scored clips. Every model, every signer improves,
most on the hardest (34503). Stacks with ensembling: **0.7210** best fold-0 result (needs ≥ a batch of
unlabelled clips per signer — fine for a per-user app, not comparable to Kaggle's per-clip scoring).
**Reading:** part of the cross-signer gap is a per-signer shift in feature statistics that simple
renormalisation removes — direct evidence for the signer-shift diagnosis. Next along this line:
per-signer input normalisation (train and test), and test-time adaptation of BN affine parameters
(e.g. entropy minimisation, TENT).

**Follow-up: input normalisation and TENT** (same held-out halves; clips canonicalised first):

| | base | AdaBN | input norm | input norm + AdaBN | TENT |
|---|---|---|---|---|---|
| fixed | 0.6774 | 0.6858 | 0.6511 | 0.6605 | **0.6897** |
| graph | 0.6855 | 0.6967 | 0.6720 | 0.6778 | **0.6990** |
| fixed+graph | 0.6990 | 0.7139 | 0.6840 | 0.6900 | **0.7164** |

- **Per-signer input normalisation hurts (−1.4 to −2.6)**: each position feature mapped to the
  training-population mean/std using the signer's calibration clips (hands on detected frames only,
  missing hands untouched, Δ1 rebuilt). Standardising every coordinate independently moves joints by
  different amounts — distorting hand shape — and removes the signer's typical hand locations, which
  partly carry meaning. Not pursued (a train-and-test version could differ, but the test-time form
  is clearly negative).
- **TENT** (AdaBN statistics, then BN + LayerNorm affine params minimise prediction entropy on the
  calibration half, 2 passes, Adam 1e-4): small, consistent extra over AdaBN — +0.4 / +0.2 / +0.25.
  The gain is in re-estimating normalisation statistics, not in input rescaling or affine tuning.


### 1st-place anchor on fold 0 — notebooks ready (2026-11-10)
Question: is our ~0.69 single-model fold-0 accuracy a property of these signers, or of our recipe/
pipeline? Run the published 1st-place solution on the same held-out signers. Kaggle's 0.89 is a
4-seed ensemble trained on all 21 signers and scored on a hidden test set (signer overlap unknown),
so it does not answer this.
- `Build_ISLR_TFRecords_SignerFolds.ipynb` (CPU): 1st-place record format, our 7 signer folds;
  ~18.5 GiB (3.58 M frames × ~5.5 KB; floats barely compress).
- `Train_1st_Place_Fold0.ipynb` (GPU): original cells verbatim except environment pins, data paths,
  GPU strategy (batch 64 × GPUs, lr 5e-4 × GPUs — the original's scaling of 64 × 8 TPU replicas),
  mixed_float16 on non-bf16 GPUs, `EPOCHS` + 11 h time limit, and a per-signer evaluation cell.
- Local smoke test (TF 2.15, CPU, 560 synthetic records, AWP + late dropout forced on from epoch 0):
  trains, holds out fold 0, saves best/last weights and `fold0_val_probs.npz`. Caught on the way:
  TF 2.15 needs numpy < 2, which breaks current pandas/pyarrow → pinned in the install cell.
**Reading:** ≥ 0.75 on fold 0 → our recipe/pipeline is the gap; ≈ 0.69 → the signers are; either way
its predictions are an ensemble candidate. Deviation to note: 300 epochs may not fit the 12 h
session on a T4 — the first epoch prints how many fit.

### Error review + fold-0 val breakdown (`fold0_graph`, 2026-11-11)
**Tool:** `python -m cnn_transformer.error_review` (commit `848d687`) — 100 `fold0_graph` val errors
(34/33/33 across signers 34503/49445/62590; 49 from the 20 most frequent confusion pairs), each shown as
an animated landmark clip next to two unlabelled reference sets (2 training clips each, distinct
signers) of the true and the predicted sign in random order; undetected-hand frames drawn faded.
Reviewed by a **non-signer** (no ASL knowledge), answers before reveal; 12 flagged low confidence,
1 seen with labels. Answers: `results/error_review/fold0_graph_errors_answers.tsv`.

| answer | n | |
|---|---|---|
| picked the true sign ("information present") | 35 | |
| picked the predicted sign | 39 | |
| A and B look the same | 15 | lips/mouth 7/7, say/chin 3, say/mouth 2 — mouth-region pointing signs |
| can't judge | 10 | 5 of them are **6-frame** clips; median 83 % of frames with no hand |
| neither | 1 | |

**The A/B picks are at chance:** true sign in 35/74 = 47 % (chance 50 %, exact binomial p = 0.73);
frequent pairs 22/36 = 61 % (p = 0.24); other errors 13/38 = 34 % (p = 0.07, leaning towards the model's
guess). For a non-signer the landmark animations of these errors do not separate the two signs, so the
review **does not** establish a learner-fixable share. It also cannot show the information is absent —
fine handshape/movement differences may need a fluent signer (the kit's signer mode; not done).
Patterns: consistent mismatches judged as the predicted sign, made by all three models —
glasswindow→tooth 3/3, yes→carrot 2/2, cut→scissors 2/2 (likely signer-specific / regional variants);
pairs where the reviewer beat chance — jump/stairs 3/3, cat/kitty 4/5, awake/wake 3/4, dry/dryer 2/2,
sleep/sleepy 2/2 (small handshape/movement details — relevant to the finger-dropout question).
59/100 errors are shared by all three models (fixed, graph, fs).

**Caveat — weak references (audit, same day):** the kit drew references at random with no tracking
check: 69/206 reference clips had a hand in < 50 % of frames, 15 had ≤ 2 detected frames, and in
58/100 items at least one true-sign reference was mostly hand-less. Restricted to items whose four
references all have ≥ 3 detected frames the A/B result is still at chance (23/50 = 46 %, p = 0.67);
with ≥ 30 % hand frames 11/26 (p = 0.56); with ≥ 50 % only 9 picks. So "at chance" is not refuted, but the
review is underpowered and partly confounded — treat it as inconclusive, not as evidence of ambiguous
landmarks. Fixed in `error_review.py` (references need a hand in ≥ 50 % of frames and ≥ 8 detected
frames; detection kept through display downsampling; kit id carries an item fingerprint so a
regenerated kit never inherits answers; `--n-heads` explicit). Regenerated kit
`results/error_review/fold0_graph_errors_v2.html` (same 100 errors, 0/206 weak references, median hand
fraction 0.67 → 0.83) — not yet reviewed.

**Whole fold-0 val set** (`fold0_graph`, 14,076 clips, acc 0.6861; top-2 0.7905, top-3 0.8296, top-5 0.8639;
predictions cached in `results/error_review/fold0_graph_val_probs.npz`):

| clip length | clips | acc | % of errors | | no-hand frames | clips | acc | % of errors |
|---|---|---|---|---|---|---|---|---|
| ≤ 6 | 7.6 % | **0.402** | 14.4 % | | 0 | 18.6 % | 0.787 | 12.6 % |
| 7–10 | 5.2 % | 0.634 | 6.0 % | | ≤ 25 % | 26.2 % | 0.759 | 20.1 % |
| 11–20 | 24.7 % | 0.763 | 18.6 % | | 25–50 % | 24.6 % | 0.703 | 23.2 % |
| 21–40 | 20.6 % | 0.739 | 17.1 % | | 50–75 % | 21.0 % | 0.607 | 26.2 % |
| 41–80 | 22.6 % | 0.683 | 22.8 % | | > 75 % | 9.7 % | **0.426** | 17.8 % |
| > 80 | 19.4 % | 0.660 | 21.0 % | | | | | |

- **Degraded clips** (≤ 6 frames or > 75 % no-hand frames): 14.9 % of val, acc 0.443 (not unrecognisable —
  partly recoverable); the other 85.1 % reach 0.7286. Even perfect accuracy on them would give 0.769, so
  they cost at most ~8 pts and realistically a few. Training has 13.7 % clips ≤ 6 frames too.
- **Tracking explains only part of the signer spread:** degraded-clip share 34503 / 49445 / 62590 =
  high / high / low (> 75 % no-hand: 13.5 / 12.7 / 2.8 %; median no-hand 0.35 / 0.48 / 0.08), but on
  non-degraded clips accuracy is still 0.632 / 0.706 / 0.834 (overall 0.583 / 0.664 / 0.813): the 23-pt
  spread shrinks only to 20.
- **Near-synonyms are a small ceiling:** merging the review's look-alike group {lips, mouth, say, chin}
  +0.4 pt; 13 a-priori near-synonym groups (wake/awake, sleep/sleepy/nap, cat/kitty, dog/puppy,
  dry/dryer, hear/listen, see/look, pen/pencil, cut/scissors, tooth/toothbrush, bed/bedroom,
  duck/goose, lips/mouth) +2.2 pt; merging the top-N confused pairs chosen on val (optimistic):
  +1.3 / +2.2 / +3.4 / +5.7 pt for N = 5 / 10 / 20 / 50. Errors are diffuse: 2,311 distinct
  (true, pred) pairs; the top 20 cover only 10.7 % of errors.

**Reading:** no single data defect explains the gap. Degraded clips (a few pts) and near-synonym labels
(~2 pts) are real but bounded; most errors are spread over thousands of sign pairs and remain
signer-dependent on clean clips — consistent with the signer-generalisation diagnosis. Next: the
recipe ablations (finger dropout off first — the resolvable errors hinge on handshape detail; then GRL,
mixup, depth), the 1st-place anchor, and optionally a fluent signer on the 54 unresolved items.

### Phrases 01 — synthetic continuous CTC, fold 0 (`phrases_fold0_graph`, 2026-11-11)
`train_phrases.py --val-fold 0 --arch graph --init <fold0_graph best_phase1.pth>`, all defaults
(40k train phrases/epoch redrawn, 3k fixed val phrases from the 3 held-out signers, 1–5 signs,
30 % random, holdout 10 %, lr 3e-4 OneCycle, 30 epochs). A40 bf16, ~4m40s/epoch, ~2.4 h.
Val kinds: seen 2,065 / unseen 224 / random 711. Results in `results/results_phrases01/` (gitignored).

| Epoch | Val WER | Exact | Single (n=194) | 34503 / 49445 / 62590 | random / seen / unseen |
|---|---|---|---|---|---|
| 1 | 0.842 | 0.003 | 0.036 | | 0.969 / 0.785 / 0.811 |
| 6 | 0.392 | 0.342 | 0.412 | 0.480 / 0.487 / 0.210 | 0.528 / 0.332 / 0.358 |
| 13 | 0.325 | 0.425 | 0.531 | 0.409 / 0.398 / 0.170 | 0.442 / 0.271 / 0.318 |
| **24 (best)** | **0.294** | **0.466** | **0.624** | 0.391 / 0.352 / 0.141 | 0.383 / 0.252 / 0.296 |
| 30 | 0.299 | 0.455 | 0.608 | 0.392 / 0.368 / 0.141 | 0.378 / 0.261 / 0.312 |

Train loss 0.17 at the end vs val loss flat at ~1.5 from epoch 11 (CTC loss also penalises
confidence; WER kept improving until the LR had decayed).

**Short-phrase subset** (`val_examples.jsonl` = the 300 *shortest* val phrases, 1–3 signs, WER
0.250 vs 0.294 overall): 383 correct / 116 substitutions / 7 insertions / 2 deletions over 501
signs. Substitutions dominate *in this subset*; diffuse, several are the isolated review's
look-alikes (sleep→nap, mouth→chin, say→chin, see→look). (An earlier version of this entry
concluded "segmentation is essentially solved" from this subset — not supported; see below.)

**Full 3,000-phrase breakdown** (`phrase_breakdown.py`, epoch-24 checkpoint, val rebuilt exactly as in training, CPU;
reproduces WER 0.293 / exact 0.467). Rates per reference sign:

| Signs | Phrases | WER | Exact | Sub | Del | Ins |
|---|---|---|---|---|---|---|
| 1 | 194 | 0.397 | 0.624 | 0.335 | 0.005 | 0.057 |
| 2 | 882 | 0.279 | 0.557 | 0.243 | 0.015 | 0.021 |
| 3 | 1,370 | 0.260 | 0.477 | 0.209 | 0.032 | 0.018 |
| 4 | 361 | 0.317 | 0.305 | 0.252 | 0.055 | 0.010 |
| 5 | 193 | 0.402 | 0.140 | 0.298 | 0.089 | 0.015 |

- Substitutions dominate at every length, but **deletions grow with length** (0.5 % → 8.9 %), almost
  all from the two weaker signers (5-sign del: 34503 0.170, 49445 0.100, 62590 0.016). On 4–5-sign
  phrases, those downsampled to the 384-frame cap delete more (34503 0.210 vs 0.094 full-length;
  49445 0.113 vs 0.075) — the cap contributes, but full-length phrases of these signers still delete
  more than 62590 (0.010), and downsampled phrases are also the longest/slowest (confounded).
  Candidate fix to test: `--max-frames 512`.
- Correct sign count ≠ correct boundaries. Proxy check: the frame where CTC emits each correctly
  recognised sign lies inside that sign's own clip span for 5,699/6,145 (92.7 %), substitutions
  1,781/2,006 (88.8 %); the rest fire in transitions or a neighbouring clip. Emission timing, not a
  boundary metric.
- Signer: 62590 WER 0.140 vs 34503 0.390 / 49445 0.352; the gap widens with length.

**Single sign, controlled** (same 194 trimmed clips, same input features; isolated classifier given
its 128-frame cap): classifier 131/194 = 67.5 %, CTC 121/194 = 62.4 %; both right 106, classifier
only 25, CTC only 15 (McNemar χ² 2.5, p ≈ 0.11 — not significant). Binomial 95 % CI for 62.4 % at
n = 194 ≈ 55–69 % (the earlier "±3.5 pts" was one standard error).

**Reading:**
- The run learns stitched sequences, including held-out combinations: unseen sequences cost ~4 pts
  WER over seen (0.294 vs 0.251); random sequences 0.383 (possibly also because they use signs the
  templates rarely use — not checked).
- Isolated-sign accuracy is not established as a ceiling: context could resolve ambiguous signs in
  natural phrases, and nothing here measures that.
- Synthetic stitched phrases (citation-form clips, interpolated transitions, no coarticulation or
  non-manuals) are an unvalidated proxy for natural signing — neither a demonstrated upper nor
  lower bound until tested on real continuous signing.

### Browser export + training-data conventions (2026-11-11)
`export_onnx.py` → `sign_ctc.onnx` (26.8 MiB fp32, `features` (1, T, 788), 8 ≤ T ≤ 512, all-valid mask
built in): max |Δlogit| vs PyTorch 2.3e-4, identical frame argmax on 32/32 LMDB inputs (single clips
and stitched phrases). Needed one model change: `HandDominanceModule` mirrors with `torch.where`
instead of boolean indexing (bit-identical on a 16-sequence test). `browser_fixtures.py` writes
raw-frame → feature → logit fixtures; an independent re-implementation of the written feature rules
matches them to 4e-7, 0 presence mismatches. (Review fix, same day: fixtures were computed by the
training pipeline, which re-infers presence from filled coordinates — a hand detected only at frame
77 was flagged at frame 104 — and skipped the 384-frame subsample (a 2,100-frame concat stayed
2,100). They are now built from the raw frames by an explicit contract: presence = actual
detection, subsample before Δ1; the training pipeline is reported as a cross-check.) Conventions a live camera pipeline must reproduce,
measured on the LMDB:
- **Unmirrored video:** pose 11 (left shoulder) x > pose 12 x in 600/600 clips.
- **Hand slots by body side:** `left_hand` wrist nearest pose 15 (left wrist) for 574/576 hands —
  not MediaPipe Tasks' handedness label (which assumes a mirrored selfie image).
- **Portrait ≈ 3:4 frames:** p99 |dy| / |dx| of hand bones in normalised coords 0.76–0.82 (45k hand
  frames; per signer 0.67–1.08) ≈ width/height. Mean shoulder drop below the nose / shoulder width
  = 0.40 (p10–p90 0.34–0.45, 2,000 clips); an uncropped 16:9 frame gives ≈ 0.9. Uncropped
  landscape on held-out clips: accuracy 62 % → 14 %.
- Simulated integration mistakes on 150 held-out clips (accuracy, correct pipeline 62 %): y flipped
  0 %, planar xyz layout 0 %, pose-first layout 2 %, hand slots swapped 2 %, no nose subtraction 13 %,
  presence all 1 27 %, z = 0 47 %, missing hand = 0 instead of held 51 %, Δ1 = 0 59 %, 10 fps 61 %.

### ASL Signs data has (almost) one hand per clip (2026-11-11)
Live browser test: CHOCOLATE (C hand on the back of a base hand) decoded as police. Pipeline
mistakes simulated on held-out chocolate clips (landscape framing, selfie mirror, swapped slots,
no nose subtraction, z = 0, 10–15 fps, hand shifted toward midline) give other errors, never
police; held-out chocolate/police are never confused with each other (0/57, 0/56).
Cause found in the data: **both hands are detected in the same frame in a median 2 % of clips**
over the 250 signs (40 clips each), including clearly two-handed signs — shoe 1 %, book / cloud
/ bath / car / rain 2 %, chocolate 4 %, police 6 % (200 clips each). The raw competition parquet
for a two-handed CLOUD clip has 28 left-hand frames and 0 right-hand frames, so it is the source
extraction, not `hand_presence`. The models see a second hand rarely; a correctly tracked base
hand (MediaPipe Tasks in the browser tracks both) is uncommon input for them. (An earlier version
of this entry said every model "learned signs from one hand" and made one-hand input a browser
rule — overstated: sparse detections do not show the models use no second-hand information.)
Removing the less active hand on held-out chocolate clips: 47/57 → 47/57 (those clips rarely have
a second hand, so this says little). **Targeted test** — phrase model, single-sign decode, every
held-out clip with detections of both hands (review follow-up, 2026-11-11):

| Held-out clips | n | two hands | one hand (less active removed) | +/− | McNemar p |
|---|---|---|---|---|---|
| both hands detected somewhere | 575 | 44.2 % | 49.0 % | +66 / −38 | 0.01 |
| both hands in the same frame | 321 | 43.9 % | 44.5 % | +26 / −24 | 0.89 |
| signer 34503 / 49445 / 62590 | 294 / 271 / 10 | 36.4 / 51.3 / 80.0 % | 40.1 / 57.2 / 90.0 % | | 0.14 / 0.04 / 1.0 |

The gain comes from clips whose two hands are detected at different times (possibly tracking
flicker or one hand changing slot — untested), not from true two-hand frames. One-hand mode is an
experimental browser option to compare against the two-hand path, not a rule. Clips with two
hands are hard either way (~44 % vs 62 % overall). Two-handed signs need data with both hands
(WLASL below, app recordings).

**Other datasets checked (2026-11-11):**
- **WLASL** (our own `shravnchandr/preprocess-wlasl-mediapipe` Kaggle notebook: legacy MediaPipe
  Holistic, model_complexity 1, confidence 0.5, competition parquet format; 11,980 clips, 2,000
  signs; its 233 shared glosses map onto 206 of our 250 signs — 32 via synonyms such as father → dad,
  cop → police, a few uncertain (sit → chair, suppose → if, upstairs → up) — 1,671 clips, median 7 per sign). Sample of 31 clips: **both hands
  tracked** — two-handed signs have both hands in 29–53 % of frames (bath 0.49, book 0.43, car 0.52,
  chocolate 0.53, clean 0.45, cloud 0.29; every clip has some), the one-handed APPLE 0 %. Same
  Holistic as the competition yet both hands → the 2 % is specific to the ASL Signs data.
  Framing differs: shoulder ratio 0.94 (p10–p90 0.76–1.06; ASL Signs 0.40) — landscape video, as
  the aspect estimate predicts (≈ 0.9 for 16:9) — and signers are farther away (shoulder width 0.25
  of the frame vs 0.56). Usable only after per-clip aspect correction. `data/build_wlasl_lmdb.py` (run on Kaggle by
  `notebooks/kaggle/build-wlasl-lmdb/`) writes an ASL-Signs-compatible LMDB with x and z scaled per
  clip so the shoulder ratio is 0.40 (local smoke test, 27 clips: median k 2.44, ratio 0.400 after;
  phrase model single-sign exact 7/27 uncorrected → 8/27 corrected — too few clips to read).
  Full output ≈ 19 GB; the overlapping signs ≈ 2.7 GB (`kaggle kernels output --file-pattern`,
  rate-limited — better built into an LMDB on Kaggle). Licence: WLASL is C-UDA (computational use).
- **WLASL LMDB built on Kaggle** (`shravnchandr/build-wlasl-lmdb` v1, CPU, minutes; output = kernel
  output, not yet a dataset): 1,671 clips, 206 signs (2–21 clips each, median 7.5; 193 clips via
  synonym glosses), 67 signers, WLASL splits 1,174 / 296 / 201, all 25 fps per the WLASL json,
  median 68 frames, 0 errors; aspect k 1.17–3.61 (median 2.30), never the fallback.
  **Second hand, per sign** (share of clips with both hands in some frame, 206 shared signs):
  WLASL median 12 % / mean 40 %, ASL Signs median 2 % / mean 3 %. 80 signs exceed 50 % in WLASL
  (chocolate, have, read, cut, tree: 100 %); the same signs have a median 2 % (max 12 %) in ASL Signs,
  and the per-sign rates are unrelated (Spearman −0.05) — ASL Signs' rare second-hand detections do
  not track which signs are two-handed.
- **Cross-dataset test** (fold-0 checkpoints, all 67 WLASL signers unseen, 1,671 clips):

  | | uncorrected | aspect-corrected |
  |---|---|---|
  | isolated `fold0_graph` top-1 (250-way) | 30.6 % | **64.2 %** (top-5 82.6 %; 66.7 % restricted to the 206 shared signs) |
  | phrase model `phrases01`, single-sign exact | 24.8 % | 45.4 % |

  - The aspect correction doubles accuracy: framing alone explains a 34-pt drop (the same mistake
    as an uncropped landscape webcam in the browser).
  - The isolated model transfers to new signers / cameras nearly at its fold-0 level (64.2 vs
    68.6 % in-dataset). Per WLASL signer: median 66.7 %, p10 8.6 %, p90 92.5 % (28 signers with
    ≥ 10 clips) — a few signers fail almost completely (not yet inspected). Synonym-gloss clips
    59.6 % vs identical-gloss 64.7 %.
  - The phrase model loses far more (45.4 %, vs 62.4 % single-sign in-dataset). Untested: WLASL
    clips are longer (median 68 frames at 25 fps vs 22 in ASL Signs) and the phrase model saw only
    0.8–1.25× tempo, the classifier 0.5–2.0×.
  - **Two-hand clips are much harder:** isolated 49.1 % (685 clips with both hands somewhere) vs
    74.6 % (986 without) — part may be the signs themselves. **One-hand mode on these clips:**
    isolated 49.1 → 51.4 % (+56 / −40, p = 0.13); phrase model **26.9 → 37.1 %** (+124 / −54,
    p < 0.001). On real two-handed data, removing the less active hand clearly helps the phrase
    model and is neutral-to-positive for the classifier — the second hand is input both models
    handle poorly, consistent with ASL Signs barely containing it.
- **MS-ASL** (`shravnchandr/ms-asl-nose-reference-mediapipe-pose`, private, 9.9 GB): not raw
  landmarks — `.npy` of 60 resampled frames × 1,848 standardised features (264 × 7), with flipped /
  fast / slow copies and no missing-value markers; hand presence cannot be recovered without the
  (unlocated) producer. Raw re-extraction would need the YouTube videos (C-UDA; many links dead).
  Annotations local: 5,811 of 25,513 clips are on our signs.

### Training on Kaggle GPU instead of RunPod (2026-11-11)
`notebooks/kaggle/kaggle_run.py` runs any module as a private Kaggle notebook (T4 16 GB, 4 CPUs,
Python 3.13, torch 2.11.0+cu130, 12 h per run). Private datasets: `asl-is-lmdb`, `islr-checkpoints`
(fold0_graph, phrases01), `wlasl-lmdb`. Smoke test (`islr-phrases-smoke`: phrase CTC, 1,600 train /
300 val phrases, 2 epochs) needed three fixes before it was usable:
1. **Forked DataLoader workers** inherited the parent's LMDB environment; py-lmdb refused to reopen
   it and the dataset fell back to parquets → FileNotFoundError. RunPod's Python 3.14 defaults to
   forkserver, so this never showed there. All DataLoaders now use forkserver (`worker_context`).
2. **Emulated bf16 on the T4:** `is_bf16_supported()` returns True; now bf16 only at compute
   capability ≥ 8 (A40 unchanged), fp16 + GradScaler otherwise. No non-finite batches in the smoke.
3. **Input-disk reads:** fp16 alone did not change speed (~1 batch/s; batches arrived 4 at a time
   every ~4 s = data-bound). Copying the 6 GB LMDB to local disk (33 s) → ~6 batches/s steady, epoch
   time 1m20s → 30s on the smoke set; losses identical to the uncopied run (only I/O changed).
Estimate for a default phrase run (1,250 batches + 3,000 val phrases): ~4–5 min/epoch, 30 epochs
≈ 2.5 h — comparable to the A40 (4m40s/epoch).

### phrases02: wider tempo range, 0.5–2× (2026-11-11)
Hypothesis: the phrase model transfers worse to WLASL than the isolated classifier (45.4 % vs
64.2 %) because it was trained at tempo 0.8–1.25× while WLASL signers sign at other speeds.
`islr-phrases02-tempo` (Kaggle T4, fp16): phrases01's recipe with `--tempo-min 0.5 --tempo-max 2.0`,
same init (fold0_graph), same fold-0 val phrases; WLASL eval of both checkpoints in the same notebook.
4m22s/epoch, early stop at epoch 28, best epoch 22, no non-finite batches.

| | phrases01 (0.8–1.25×, A40 bf16) | phrases02 (0.5–2×, T4 fp16) |
|---|---|---|
| Synthetic val WER (best epoch) | 0.294 | 0.287 |
| exact / single-sign (n = 194) | 0.466 / 0.624 | 0.471 / 0.588 |
| per-signer WER 34503 / 49445 / 62590 | 0.391 / 0.352 / 0.141 | 0.377 / 0.351 / 0.135 |
| WLASL single-sign exact decode | 45.4 % | 46.0 % |
| WLASL, aspect correction undone | 24.7 % | 23.3 % |
| WLASL clips with / without both hands | 26.9 % / 58.2 % | 28.0 % / 58.5 % |
| one-hand mode on two-hand clips | 26.9 → 37.1 % (+124 / −54) | 28.0 → 39.9 % (+137 / −56, p 5e-9) |
| WLASL per-signer median (p10, p90) | 43.5 % (0, 67) | 44.8 % (0, 67) |

The Kaggle run reproduced phrases01's WLASL numbers exactly (0.4536, as measured on CPU), so the
eval is deterministic across machines. **The tempo hypothesis is not supported:** WLASL +0.7 points
(≈ 11 clips of 1,671 net; the per-clip results were not saved, so no paired test — treat it as no
change), synthetic WER −0.007 (single seed and different hardware/precision — within noise). The
gap to the classifier (64.2 %) remains ~18 points. Not yet separated: the phrase score demands an
exact one-token decode (any insertion or deletion fails) while the classifier takes an argmax over
250 classes, so part of the gap may be the metric — next check is scoring the phrase model as a
classifier (best non-blank class over time) on the same clips. One-hand mode helps both models on
WLASL two-hand clips; the 0 % p10 signers are still uninspected.

### WLASL: metric vs model gap, and the "0 % signers" (2026-11-11)
`cross_dataset_eval` now also scores a phrase model like a classifier ("pooled": each sign's best
per-frame log-probability over the clip, blank excluded) and writes per-clip CSVs
(`results/wlasl_eval/`), so checkpoints can be compared clip by clip.

| WLASL, all 67 signers (1,671 clips) | exact decode | pooled top-1 | pooled top-5 | decode contains sign |
|---|---|---|---|---|
| phrases01 | 45.4 % | 55.5 % | 74.6 % | 56.1 % |
| phrases02 (tempo 0.5–2×) | 46.0 % | 57.1 % | 75.5 % | 58.1 % |
| fold0_graph classifier | — | 64.2 % (top-1) | 82.6 % | — |

- **About half the gap was the metric.** phrases01 decodes 2+ signs on 295 clips and nothing on 65;
  of the 170 clips pooled scoring recovers, 153 had decoded as two signs. The rest is real:
  pooled phrases01 vs classifier +134 / −278 clips (McNemar p 1e-12), phrases02 p 5e-10. In-domain
  the gap is ~5 points (67.0 vs 62.4 single-sign, phrase breakdown); cross-dataset ~8.
- **Paired phrases01 → phrases02:** exact +138 / −127 (p 0.54), pooled +126 / −100 (p 0.10) — the
  wider tempo range is still no clear gain.
- **The 0 % p10 signers are a small-sample artefact:** 17 of 67 signers have ≤ 3 clips (28 clips
  total); the 7 at 0 % have 1–2 clips each. Among the 22 signers with ≥ 20 clips (1,412 clips),
  classifier accuracy is 48–82 %, phrases01 pooled 34–81 %; nobody fails outright. Per-signer
  accuracy hardly tracks clip count, hand-visible fraction, aspect k or raw shoulder ratio
  (|Spearman| ≤ 0.18).
- **What does track errors is a second hand:** clips with both hands detected somewhere score 49.1 %
  vs 74.6 % (classifier) and 35.8 % vs 69.3 % (phrases01 pooled), and the phrase model loses more.
  Confounded with the sign (two-handed signs), but it matches the ASL Signs data gap (a second hand
  in a median 2 % of clips). By source, signingsavvy is lowest (54 %, 344 clips), aslbrick highest
  (80 %); clips where a hand is visible > 90 % of frames score lower (53 %), likely hands held up
  before / after the sign.

### phrases03: WLASL training signers mixed into phrase training (2026-11-11)
`build_wlasl_lmdb.signer_split`: WLASL signers sorted by clip count, dealt snake-wise → train half
33 signers / 847 clips, test half 34 / 824 clips; both have all 206 signs and 40 / 42 % two-hand
clips. `train_phrases --extra-data-dir {WLASL_DIR} --extra-frac 0.15`: 15 % of phrases come from WLASL
train-half signers (each in proportion to its clips; one signer per phrase as before, ~2.6 signs),
the rest from the 18 ASL Signs training signers as before; phrases01's recipe otherwise (tempo
0.8–1.25×). Each WLASL clip is reused ~19× per epoch — overfitting to those signers is possible and is
what the held-out test half measures. Baselines on the **test half**: classifier 66.7 %; phrases01
exact 44.2 % / pooled 57.0 % (two-hand clips exact 25.8 %); phrases02 44.2 % / 57.8 %.
Kaggle notebook `islr-phrases03-wlasl` (commit 6ede152).

**Result** (T4 fp16, 3m53s/epoch, early stop at 28, best epoch 22). Paired on the 824 test-half
clips; phrases01 rescored in the same notebook = the local CSV clip for clip.

| WLASL test half (34 unseen signers) | phrases01 | phrases03 | paired |
|---|---|---|---|
| exact decode | 44.2 % | **80.1 %** | +313 / −17, p 1e-71 |
| pooled top-1 / top-5 | 57.0 / 75.6 % | 81.2 / 94.5 % | top-1 +219 / −20 |
| two-hand clips (345), exact | 25.8 % | 76.8 % | p 2e-45 (classifier 53.0 %) |
| one-hand clips (479), exact | 57.4 % | 82.5 % | p 9e-28 (classifier 76.6 %) |
| aspect correction undone | 24.3 % | 39.2 % | |
| decodes with 0 / 1 / 2+ signs | 35 / 605 / 184 | 0 / 811 / 13 | |
| one-hand mode on two-hand clips | 25.8 → 50.5 % (helps) | 76.8 → 57.1 % (**hurts**, +13 / −81) | |
| signers with ≥ 20 clips improved | | 11 / 11 | |
| ASL Signs synthetic val WER | 0.294 | 0.296 | in-domain unchanged |

Also beats the fold0_graph classifier on the same clips (81.2 vs 66.7 %, +166 / −47).
**Leakage check:** no shared video ids, no identical clips across halves, and a test clip's nearest
same-sign train-half clip (32-frame upper-body trajectory) is no closer than its nearest other-signer
test-half clip (median 2.22 vs 2.10; none < 0.05). WLASL signer ids could still merge or split real
people; not checkable here.

What this does and does not show: ~850 clips (≈ 4 per sign) gave +36 points on new WLASL signers
without costing anything in-domain, and the model now uses a second hand (two-hand clips 26 → 77 %;
removing the base hand now hurts). But the test signers come from the **same websites / recording
setups** as the train half, so part of the gain is adaptation to WLASL's recording style (one-hand
clips also rose 57 → 83 %, single-sign decodes 73 → 98 %). It is evidence that a small amount of
two-handed, differently-recorded data matters a lot — not yet that the model works on a webcam.
Consequences: one-hand mode should be off for phrases03; exported to `exports/phrases03/`
(ONNX parity 1.06e-4, 32/32 argmax) for live testing in the browser.

---

## Pending Ideas (not yet implemented)

### High priority
| ID | Idea | Expected impact | Notes |
|----|------|----------------|-------|
| 4A | Pre-training on fingerspelling data | High | Leverage unlabeled data for better hand-shape representations; fingerspelling has 30 distinct hand configs forcing fine-grained learning — see notes below |

#### 4A — fingerspelling CTC pre-training: status and plan (2026-11-09)

**Why it is the top idea now.** Every fold-0 result points at signer generalisation, not capacity:
clean train 0.97 vs val 0.68 with 18 training signers; GRL, contrastive loss and augmentation
have not closed the gap. Google DeepMind's SL2T (Pixel 11, Aug 2026) is the external data point:
the same input (MediaPipe Holistic landmarks), but >100k h over 50+ sign languages (~¼ ASL), and
DeepMind reports joint training across languages/dialects beat single-language models (no numbers
published). The lever that works at scale is *more signers and more varied data*, even from a
related task. The ASL Fingerspelling set is the cheapest such source we already have (separate
participants from ASL Signs; signer count to be read from its `train.csv`).

**Status: implemented, never run.** `pretrain_fingerspelling.py` trains `LandmarkConformer` with
a per-frame CTC head (60 chars + blank) on `data/asl-fs-lmdb`, signer-held-out 10% val
(`GroupShuffleSplit` on `participant_id`), early stopping on val CTC loss, and saves
`backbone_best.pth`. `train.py --pretrained-backbone` loads it (`strict=False`), freezes it for
`--backbone-warmup-epochs` (5) while new params train, then unfreezes at `--backbone-lr-ratio`
(0.1). All Runs 001–005 and every fold-0 run used `--skip-pretrain`. The FS LMDB already goes
through `hand_presence()` (0/27,339 flag errors), so it shares the presence/gap convention.

**Gaps to close before spending GPU time on it** (pretraining has fallen behind `train.py`):
1. **No `--arch graph`.** It always builds `LandmarkConformer`; a `GraphConformer` fine-tune would
   load no hand-graph weights (they would train from scratch at head LR — the part pretraining is
   meant to improve). Needs the same `--arch/--graph-*` flags.
2. **fp16 + GradScaler, no non-finite guard.** The Run 004 failure mode (one NaN batch poisoning
   BatchNorm running stats) applies here too: bf16 on Ampere+ and the BN snapshot/restore from
   `train_epoch` are missing.
3. **No augmentation at all** (no noise, affine, stretch, bone scaling, finger/hand dropout) and
   **no `--seed`**. Without augmentation the backbone may learn FS-signer specifics — the very thing
   we are trying to dilute.
4. **Only CTC loss is reported.** Add greedy-decoded character error rate on the held-out signers,
   so we know the backbone learned hand shapes before paying for a fine-tune.
5. **Input conventions differ from fine-tuning:** `max_frames` 384 vs 128 (fine — sinusoidal PE),
   dropout 0.1 vs 0.2 and drop-path 0.05 vs 0.1 (fine-tune values apply after loading).
6. **Pod setup:** `setup_runpod.sh` is ISLT-only (FS handling was removed); the FS LMDB
   (`kaggle datasets download shravnchandr/asl-fs-lmdb`) is much larger than the IS LMDB — size the
   volume before downloading.

**Gaps 1–4 and 6 closed (2026-11-09):** `--arch/--graph-*` (+ `backbone_meta.json`, checked by
`train.py`), bf16 + non-finite guard, augmentation (noise/shift, bone scale, per-clip stretch
0.8–1.25×, affine; no mixup/hand/finger dropout) + `--seed`, val CER with best-by-CER selection,
`setup_runpod.sh --with-fs`; the pipeline forwards `--arch/--graph-*/--seed` to both stages.
Facts measured on the local sample: FS has **94 signers** (none sharing an ID with ASL Signs' 21;
ID schemes may differ), 67,208 phrases, median 17 chars; the LMDB is ~18 GB (263 MB per 998
phrases). The Kaggle FS build logic (script and notebook) is unchanged since May and produces
exactly `config.ALL_COLUMNS` — no rebuild needed. CPU smoke tests on 998 local phrases: loss
4.84 → 3.10 in 8 epochs, CER leaving 1.0 at epoch 5 (normal CTC blank phase); fine-tune with a
graph backbone + `--compile` + bone aug ran end to end; an arch mismatch exits with an error.

**FS LMDB rebuilt on Kaggle (2026-11-10)** with `notebooks/Build_FS_LMDB.ipynb`: 67,208 sequences,
68 files, 94 signers, **0 errors, 4 min** on 4 CPU workers (column-subset reads), 15.9 GiB, built in
`/kaggle/tmp`; published as a new version of `shravnchandr/asl-fs-lmdb`.

**Plan.** (a) ~~Close gaps 1–4.~~ (b) Pre-train for the chosen arch (whichever of `fold0_fixed` /
`fold0_graph` / `fold0_graph_bone` is best), checking held-out-signer CER. (c) Fine-tune on fold 0
with the same flags + `--pretrained-backbone`, compared against that run (one change).
**Success** = val ≥ +2 pt, or the clean-train gap narrower, with gains reaching the hard signer
(34503). A null here would be strong evidence that the 18-signer limit is not fixable with
hand-shape pretraining alone, and that only more ASL-Signs-style signers (or other isolated-sign
data, e.g. `data/WLASL_Landmarks/`) would move it.
**Cost:** two runs (pretrain + fine-tune). Estimated from measured FS lengths (median 146, mean
162, 2.4 % > 384 frames) against the graph fine-tune's 2m50s/epoch: random batches padded every
phrase to ~357 frames (2.2× the real 160) → ~18 min/epoch, 40 epochs ≈ 12 h. Now length-bucketed
(→ ~164 frames) with defaults 20 epochs / patience 5 → ~8 min/epoch, ≈ 2.5–3.5 h, plus ~4 h for the
fine-tune (estimates, ±30 %). CPU check on the local sample: same loss/CER behaviour as unbucketed.

**Note on SL2T's stated weaknesses** — rapid fingerspelling, rare signs, left-handed and
one-handed signers, low light / signer partly out of frame (i.e. tracking loss). These overlap
with what our per-signer diagnostics measured (hand-detection gaps rank per-signer accuracy,
Spearman −0.94; dominance mirroring). SL2T publishes no per-signer or per-condition numbers.

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
| 1A | GCN stem | Implemented (`--arch graph`) | Shared hand-graph encoder replacing the flat hand projections; see GraphConformer entry |
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
