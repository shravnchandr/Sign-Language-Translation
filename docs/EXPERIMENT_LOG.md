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

## Run 004 — Geometry Stream + Normalization Fix (next RunPod run)
**Config changes vs Run 003:**
- Geometry stream added (joint angles + fingertip distances)
- Proper preprocessing normalization (nose→shoulder→hip fallback)
- LMDB rebuild required (CACHE_VERSION changed)
- Phase 2 dropped; single-phase 100 epochs

**Expected val accuracy:** TBD — geometry features add explicit hand-shape signal that raw XYZ buries; expect improvement over 0.7432 if the information ceiling was partly a representation issue.

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
| 1st  | 0.8929 | 1D CNN + Transformer, wrist-relative features |
| — | — | Most top solutions: small models (<10M params), strong augmentation, signer-independent splits |

Key insight: The 1st place solution's score is achievable with a relatively small model. The gap in Run 001 is almost entirely overfitting to signer identity, not model capacity.
