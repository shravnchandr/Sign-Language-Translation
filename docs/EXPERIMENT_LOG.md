# Experiment Log — AnatomicalConformer (cnn_transformer)

Tracking every training run, the config used, and the result. Goal: 250-class ASL sign recognition on Google ASL Signs dataset (signer-independent val split).

**Baseline to beat:** 1st-place Kaggle solution — 0.8929 accuracy (1D CNN + Transformer).

---

## Run 001 — Baseline AnatomicalConformer
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

## Run 002 — Regularised AnatomicalConformer
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
- Overfitting eliminated: train acc ~0.42 vs val acc ~0.755 — inverse of Run 001's 99.9% / 74.6% pattern. Heavy aug + mixup + smaller model all contributing.
- Phase 2 didn't improve on Phase 1 best (P2 peak 0.7551 vs 0.7555); warmdown acted as polishing, not exploration. `best_final.pth` is the Phase 1 checkpoint.
- Phase 2 train loss climbed 0.40 → 0.65 over the first ~6 epochs — expected warm restart at LR=1e-4 after Phase 1 finished at LR~8e-6. Val acc held steady throughout.
- TTA gain negligible (+0.14%) — model predictions stable under augmentation; stochastic variance already low.
- Disc acc not logged — this run predates the discriminator accuracy logging commit. GRL activity cannot be confirmed from logs alone.
- Expected 0.82–0.86; achieved 0.7555. Gap vs expectation likely reflects GRL not being verified as active, and the 0.82+ target requiring additional feature improvements (multi-scale velocity, cross-part attention).

---

## Changes Applied After Run 002

*(pending — to be filled in before Run 003)*

---

## Run 003 — Pending (next RunPod run)
**Config changes under consideration:**
- Confirm GRL active (disc_acc logging now in place — verify discriminator converges to near chance)
- Multi-Scale Velocity (idea 2A): delta=1, delta=2, delta=5 velocity channels
- Cross-part attention (idea 1B): hand↔pose spatial relationship

**Expected val accuracy:** TBD

---

## Pending Ideas (not yet implemented)

### High priority
| ID | Idea | Expected impact | Notes |
|----|------|----------------|-------|
| 2A | Multi-Scale Velocity (delta=2, delta=5) | Medium | No LMDB rebuild needed; concat 3× vel channels; refactor vel_proj |

### Medium priority
| ID | Idea | Expected impact | Notes |
|----|------|----------------|-------|
| 2B | Distance features (fingertip pairs) | Medium | Invariant to translation/rotation by construction |
| 1B | Cross-part attention (hand↔pose) | Medium-high | Spatial relationship between hand position and body |

### Low priority / explored
| ID | Idea | Status | Notes |
|----|------|--------|-------|
| 1A | GCN stem | Skipped | Added complexity; conformer already captures spatial structure via conv |
| 3B | STN (Spatial Transformer) | Skipped | WristNorm covers same invariance more cheaply |

---

## Notes on Kaggle Leaderboard Context

| Rank | Score | Key techniques (public info) |
|------|-------|------------------------------|
| 1st  | 0.8929 | 1D CNN + Transformer, wrist-relative features |
| — | — | Most top solutions: small models (<10M params), strong augmentation, signer-independent splits |

Key insight: The 1st place solution's score is achievable with a relatively small model. The gap in Run 001 is almost entirely overfitting to signer identity, not model capacity.
