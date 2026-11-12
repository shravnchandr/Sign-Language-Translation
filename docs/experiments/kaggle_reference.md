# Kaggle reference

Leaderboard context and the 1st-place recipe. Entries in chronological order; paths are as they were at the time.

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
