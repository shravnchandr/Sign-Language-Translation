# Isolated Sign Language Recognition

Research into signer-independent ASL recognition from MediaPipe landmark sequences — first isolated signs (250 signs, Google ASL Signs), now continuous multi-sign phrases, with the goal of free, on-device sign recognition that runs in a browser. The core problem: models trained on one set of signers generalize poorly to new ones, because they memorize individual signing styles rather than learning sign representations.

## Results

All runs use a signer-independent val split (no val signer seen during training). The 1st-place Kaggle score (**0.8929**) is a 4-seed ensemble trained on all 21 signers and scored on a hidden test set, so it is not directly comparable.

**Signer folds (current protocol).** `--val-fold k` holds out 3 of 21 signers (GroupKFold(7)); fold 0 = signers 34503, 49445, 62590 (14,076 clips). Signer spread is large (≈ 0.57–0.82 per signer), so pooled 1-point differences are within noise.

| Fold 0 | Val accuracy | Notes |
|--------|-------------|-------|
| `fold0_fixed` — LandmarkConformer, all fixes + per-clip time stretch | 0.677 | baseline |
| `fold0_graph` — GraphConformer (`--arch graph`) | **0.686** | hand-skeleton graph encoder; McNemar p = 0.004 vs baseline (one seed) |
| ensemble of the two (softmax average) | 0.703 | 2× inference |
| `fold0_graph` + per-signer BatchNorm adaptation (AdaBN, offline) | 0.697 | BN statistics re-estimated from half of each held-out signer's unlabelled clips, scored on the other half |
| fixed + graph + FS-pretrained ensemble, each with AdaBN | **0.721** | best fold-0 result so far |

Bone-length augmentation, fingerspelling pre-training (alone) and a cross-signer contrastive loss gave no gain on fold 0 (details in `docs/experiments/`).

**Earlier runs (fixed 3-signer split, not comparable with the folds):**

| Run | Model | Val Accuracy | Key changes |
|-----|-------|-------------|-------------|
| 001 | LandmarkConformer (d_model=512, 49M) | 0.7462 | baseline |
| 002 | LandmarkConformer (d_model=256, 6.5M) | 0.7555 | smaller model, GRL, stochastic depth, dominance-aware mixup |
| 003 | + multi-scale velocity (Δ1/Δ2/Δ5), face split, depth | 0.7432 | slight regression — more features need more epochs to converge |
| 004 | + geometry stream, hand presence, normalization fix, augmentation / handedness / GRL fixes | 0.7590 (TTA 0.7610) | best at epoch 50; a NaN batch at epoch 55 broke eval for the remaining epochs |
| 005 | Run 004 + bf16 AMP, non-finite batch guard | **0.7665** (TTA 0.7669) | clean 80 epochs; plateaued — run-to-run noise ≈ 1 pt on this split |
| — | Factorized VQ-VAE + Conformer | — | dormant (approach 0) |
| — | ST-GCN | — | superseded by `--arch graph` (wrong graph indexing, by-sign split) |

### Continuous signing (synthetic phrases)

There is no openly licensed continuous ASL dataset with this vocabulary and MediaPipe landmarks, so phrases are stitched on the fly from isolated clips (`islr/phrases/`): one signer per phrase, clips trimmed to the hand activity, 3–8 interpolated transition frames, 1–5 signs per phrase (70 % ASL-style gloss templates, 30 % random sequences). A CTC head over the 250 signs + blank is trained from the fold-0 GraphConformer; validation phrases are built only from the 3 held-out signers, and 10 % of gloss sequences are never trained on.

| `phrases01` (fold 0, 3,000 val phrases) | |
|---|---|
| Word error rate | **0.294** (seen sequences 0.251, unseen 0.294, random 0.383) |
| Exact phrase | 0.467 |
| Per signer WER | 0.140 / 0.352 / 0.390 |
| By length (1 / 2 / 3 / 4 / 5 signs) | 0.40 / 0.28 / 0.26 / 0.32 / 0.40 — substitutions dominate; deletions grow with length |

These are synthetic phrases (citation-form clips, artificial transitions, no coarticulation or facial grammar): an unvalidated proxy for real signing until tested on real recordings.

**Current browser model: `phrases05`** (`exports/phrases05/`; `phrases04` before 2026-11-12). Each step changed one thing and was scored on WLASL signers never used in training (824 clips, single sign, exact decode):

| Model | Change | WLASL held-out signers | ASL Signs synthetic WER |
|---|---|---|---|
| `phrases01` | ASL Signs only | 44.2 % | 0.294 |
| `phrases02` | wider tempo range (0.5–2×) | no clear change (paired p = 0.54) | 0.287 |
| `phrases03` | + half of WLASL's signers (two-handed data, 15 % of phrases) | **80.1 %** | 0.296 |
| `phrases04` | + viewpoint augmentation (per-axis scale, shear, rotation) | 80.8 % (77.4 % with WLASL's framing correction undone; phrases03 39.2 %) | 0.291 |
| `phrases05` | + all 2,000 WLASL glosses as extra labels, WLASL 30 % of phrases — **current browser model** | 83.7 % (paired vs phrases04 p = 0.025; pooled top-1 82.6 → 84.1 %, p = 0.26; ASL Signs held-out top-1 65.1 → 64.3 %) | 0.285 |

On **ASL Citizen** (51 unseen Deaf signers, webcam, 8,557 clips; research use only — no model is trained on it) the same order holds independently: phrases01 53.7 %, phrases03 69.0 %, phrases04 68.7 % (66.9 % with the framing correction undone, phrases03 44.2 %), **phrases05 72.4 %** (paired vs phrases04 p = 2e-19; pooled top-1 75.4 vs 74.2 %, p = 0.002), isolated classifier 65.2 %.

On ASL Signs' held-out signers phrases05 ranks the right sign first 64 % of the time and in its top 3 81 % (phrases04 65 / 81 %); `sign_reliability.json` gives per-sign accuracy, common confusions (often near-synonyms: cat/kitty, pen/pencil, wake/awake) and score thresholds for a "not sure" state. Caveats: the WLASL halves share recording sources, the viewpoint augmentation was chosen on the same test half (ASL Citizen, above, is the independent check), and live webcam use is still untested beyond anecdotes.

### Running in the browser

`islr/export/export_onnx.py` exports the phrase model to ONNX for onnxruntime-web (verified against PyTorch), with `labels.json` and a `preprocess.json` describing the input features; `islr/export/browser_fixtures.py` writes golden raw-frame → feature → output test cases. The app itself is [ASL-App](https://github.com/shravnchandr/ASL-App). A live camera pipeline has to reproduce the training data's conventions, measured on the dataset:

- **Unmirrored frames**, hands assigned to slots by distance to the pose wrists (MediaPipe's handedness label assumes a mirrored selfie image).
- **Portrait ≈ 3:4 framing** — landmark x is a fraction of the image width and y of the height; an uncropped landscape webcam drops held-out accuracy from 62 % to 14 %.
- **Cross-dataset check (WLASL, 67 unseen signers, 206 shared signs):** the isolated fold-0 graph model scores 64.2 % top-1 after aspect correction (30.6 % without), close to its 68.6 % on held-out ASL Signs signers; the phrase model 45.4 % (55.5 % scored like a classifier — most of the rest of its gap is extra decoded signs). Built by `islr/core/data/build_wlasl_lmdb.py` on Kaggle.
- **Training with half of WLASL's signers (`phrases03`):** mixing the 847 clips of 33 WLASL signers into phrase training (15 % of phrases) raises single-sign accuracy on the other 34 WLASL signers from 44.2 % to **80.1 %** (two-hand clips 25.8 → 76.8 %), with ASL Signs val WER unchanged (0.296 vs 0.294) and no duplicate clips across the halves. The halves share recording sources, so part of the gain is adaptation to WLASL's recording style; live webcam use is still untested.
- **Second hand rarely present in training.** Both hands are detected in the same frame in a median **2 %** of training clips, even for two-handed signs (shoe 1 %, chocolate 4 %) — a limitation of the source data (WLASL, extracted with the same MediaPipe Holistic, tracks both). The models see a second hand rarely; how much they use it is not measured. A live base hand may push predictions off (anecdotally CHOCOLATE → police). **One-hand mode** (pass only the more active hand) is an experimental option for `phrases01`, to compare against the two-hand path — do not use it with `phrases03`/`phrases04`, which were trained with two-handed WLASL clips (it lowers two-hand clips from 76.8 % to 57.1 %). On WLASL's genuinely two-handed clips it raises the phrase model 26.9 % → 37.1 % (p < 0.001; classifier 49.1 → 51.4 %, p = 0.13). On ASL Signs held-out clips: on the 575 held-out clips with both hands detected somewhere, removing the less active hand raised phrase-model accuracy 44.2 % → 49.0 % (+66 / −38, McNemar p = 0.01), but on the 321 with both hands in the same frame it changed nothing (43.9 % → 44.5 %, +26 / −24, p = 0.89).

## Research Directions

- **Signer invariance** — adversarial GRL discriminator forces the feature extractor to discard signer identity
- **Hand dominance normalization** — always route the dominant hand to the same input channel, making the model left/right-agnostic by construction
- **Discrete motion tokenization** — factorized VQ-VAE separates what the body looks like (pose) from how it moves (motion, dynamics), producing interpretable tokens that may generalize better across signers
- **Multi-scale temporal modeling** — velocity at Δ1/Δ2/Δ5 frames and chunk encoding at 4/8/16 frames capture finger micro-motion, arm movement, and global sign structure simultaneously
- **Geometry-aware features** — joint-angle cosines and fingertip distances give hand shape representations invariant to wrist rotation and signer hand scale
- **Hand-skeleton graph encoder** — GraphConformer replaces the per-hand linear projections with a shared graph network over the hand skeleton
- **Continuous recognition** — CTC over synthetic phrases built from isolated clips, as a step toward sentence-level recognition

---

## Architectures

### LandmarkConformer backbone — approaches A–C (`research/models/islr/`)

End-to-end supervised classification. Processes raw landmarks into three parallel feature streams, fuses them, then runs Conformer blocks over the sequence.

**Three feature streams:**

```
landmarks (T, N, 3)
      │
      ├─ position stream ──────────────────────────────┐
      │   per-part Linear (lh, rh, pose,               │
      │   eyebrow, mouth) → d_model                    │
      │                                                ├─ feat_fuse → d_model → Conformer ×4 → CLS → 250 classes
      ├─ velocity stream (d1 | d2 | d5) ───────────────┤
      │   per-part Linear → d_model                    │
      │                                                │
      └─ geometry stream ──────────────────────────────┘
          joint angles + fingertip distances → d_model/4
```

| Stream | What it captures |
|--------|-----------------|
| Position (d_model) | Body-relative joint locations, split by body part |
| Velocity (d_model) | Multi-scale motion: Δ1 (raw coords), Δ2/Δ5 (body-relative) |
| Geometry (d_model/4) | 15 joint-angle cosines + 10 fingertip distances per hand |

The geometry stream is the key addition: raw XYZ obscures hand shape because it conflates location with configuration. Joint angles and fingertip distances are invariant to wrist rotation and signer hand scale.

**Signer invariance via GRL:** A `SignerDiscriminator` sits on the CLS token with a gradient reversal layer. The feature extractor is penalized for producing representations that distinguish signers — by the end of training, the discriminator performs at chance level, confirming the features have shed signer identity.

**Optional CTC pre-training:** Before supervised fine-tuning, the backbone can be pre-trained on ASL Fingerspelling via CTC loss. This initializes hand-shape representations from a much larger dataset before the model sees the 250-class labels. A warmup phase then freezes the pre-trained backbone while the new classification head trains in, then jointly fine-tunes at a lower backbone learning rate.

**Model:** ~7M parameters — d_model=256, 4 Conformer layers, 4 attention heads.

---

**GraphConformer (`--arch graph`):** the same model with the hand streams replaced by a shared hand-skeleton graph encoder (skeleton + knuckle-web adjacency plus a learned residual adjacency) over wrist-relative positions, bone vectors and multi-scale velocities; both hands share weights (the non-dominant one is mirrored). Same data, no rebuild.

**CTC mode:** with `ctc_vocab_size` set, the model skips the CLS token and returns per-frame logits — used for fingerspelling pre-training and for continuous phrases.

---

### Approach 0 — Factorized VQ-VAE + Conformer (dormant, `research/models/vqvae_seq2seq/`)

Two-phase pipeline: unsupervised discrete tokenization followed by supervised sequence-to-label translation.

**Phase 1 — Factorized VQ-VAE (unsupervised):**

Each chunk of frames is encoded into four independent discrete tokens — one per codebook:

| Codebook | Size | Captures |
|----------|------|---------|
| Pose | 256 | Body/hand configuration |
| Motion | 256 | Velocity between frames |
| Dynamics | 128 | Acceleration patterns |
| Face | 128 | Non-manual markers (expressions, mouthing) |

Separating these into independent codebooks rather than a single shared one reduces codebook collapse and produces more interpretable tokens. The encoder is trained unsupervised, so it can use all available data including fingerspelling.

The VQ-VAE runs once after training to tokenize the entire dataset; Phase 2 loads the cached token indices directly (~5–10× faster per epoch, no VQ-VAE in memory).

**Phase 2 — Conformer Translator (supervised):**

A Conformer encoder (6 layers, d_model=256, kernel=7) reads the 5–10 token sequences, and a hybrid CTC + attention decoder (3 layers) produces the 250-class label. Inference uses beam search with CTC prefix scoring.

---

### ST-GCN (legacy, `research/models/st_gcn/`)

Spatial-Temporal Graph Convolutional Network. Legacy baseline, superseded by `--arch graph` (its graph indexing was wrong and it used a by-sign split, so its numbers are not comparable).

- 209-node anatomical graph (hands, pose, selected face landmarks)
- 9 ST-GCN blocks in 3 stages: 64 → 128 → 256 channels
- Global average pooling for classification

---

## Key Design Decisions

**Signer-independent splits:** Train/val/test are split by signer identity. Without this, a model can reach high accuracy by memorizing individual signers and will fail badly on new ones.

**Hand dominance normalization:** Wrist velocity determines which hand is dominant; the dominant hand is always placed in the first input channel. Left-handed signers naturally already have their dominant hand in the left-hand slot. This makes the model hand-agnostic without requiring any data augmentation.

**Body-relative coordinates:** Nose position is subtracted at LMDB build time (fallback: shoulder center → hip center → zero) so all coordinates are relative to the signer's body rather than the camera frame.

**Factorized quantization:** A single codebook entangles pose and motion — a sign with similar handshape but different speed would get different tokens. Separate codebooks let the model attend to just the relevant factor.

**Soft diversity loss:** The VQ-VAE encoder receives a gradient from a soft entropy term computed across the distance matrix before the argmin, pushing it toward spread-out representations and away from codebook collapse.

**Dominance-aware mixup:** Mixup pairs samples with the same dominant hand. Mixing a left-dominant sample with a right-dominant one would give `HandDominanceModule` an ambiguous tensor — the paired dominant hand energies would conflict.

---

## Project Structure

```
research/models/       # model code (PYTHONPATH=research/models) — see tree below
scripts/               # tools: run_pipeline_isolated.sh / run_pipeline_vqvae_seq2seq.sh, setup_runpod.sh,
                       #   runpod_results.sh, kaggle_run.py (any module on a Kaggle GPU),
                       #   extract_aslcitizen.py (ASL Citizen landmarks, research only),
                       #   download_kernel_output.py (large Kaggle outputs; used by setup_runpod.sh)
notebooks/             # by approach — see notebooks/README.md
  data/                #   dataset builders + MediaPipe extraction (LMDBs, WLASL); legacy/ older builders
  exploration/         #   early data exploration
  early_transformers/  #   first classifier attempts
  vqvae/               #   approach 0 (VQ-VAE → translator)
  first_place/         #   1st-place Kaggle solution reproduced on our signer fold
archive/research/      # superseded prototypes, feature experiments and tools (kept for reference)
data/                  # datasets (gitignored) — see below
exports/               # browser-ready model (gitignored): phrases05/ = ONNX + labels + preprocess spec +
                       #   sign_reliability.json + fixtures/
results/               # run outputs (gitignored), by approach: isolated/, pretrain/, phrases/<run>/,
                       #   evaluation/ (wlasl, error_review, signer_diagnostics); kaggle/ = download inbox
docs/                  # local notes (gitignored); docs/experiments/ = the experiment log (tracked)

research/
├── models/
│   ├── islr/                    # Current work, organised by approach (python -m islr.<sub>.<module>)
│   │   ├── core/                       # Shared by every approach
│   │   │   ├── config.py               # Landmark layout constants, feature dimensions
│   │   │   ├── model/                  # LandmarkConformer, GraphConformer (--arch graph), VQ, GRL, SupCon
│   │   │   └── data/                   # Datasets, augmentation, LMDB builders (ASL Signs, fingerspelling, WLASL)
│   │   ├── isolated/                   # A: isolated-sign classifier — train.py, error_review, signer_diagnostics
│   │   ├── pretrain/                   # B: fingerspelling CTC pre-training — fingerspelling.py
│   │   ├── phrases/                    # C (current product model): continuous CTC on synthetic phrases —
│   │   │                               #    train.py, dataset.py, grammar.py, breakdown.py
│   │   ├── evaluation/                 # cross_dataset.py (WLASL), sign_reliability.py (per-sign, for the app)
│   │   └── export/                     # export_onnx.py (browser model), browser_fixtures.py
│   ├── vqvae_seq2seq/          # Approach 0 (dormant): factorized VQ-VAE → translator
│   │   ├── vqvae/
│   │   │   ├── vqvae_model.py          # Main VQ-VAE model
│   │   │   ├── vector_quantizer.py     # EMAVectorQuantizer, FactorizedVectorQuantizer
│   │   │   ├── multi_scale_encoder.py  # MultiScaleMotionEncoder
│   │   │   ├── face_encoder.py         # FaceNMMEncoder (5 regions)
│   │   │   ├── hand_dominance.py       # HandDominanceModule
│   │   │   └── cross_attention.py      # CrossFactorAttention
│   │   ├── translation/
│   │   │   ├── translator_model.py     # SignTranslator
│   │   │   ├── conformer.py            # Conformer encoder blocks
│   │   │   ├── decoder.py              # HybridDecoder (CTC + attention)
│   │   │   └── beam_search.py          # Beam search with CTC prefix scoring
│   │   └── scripts/
│   │       └── precompute_tokens.py    # Pre-tokenize dataset with frozen VQ-VAE
│   └── st_gcn/                  # Legacy ST-GCN baseline (superseded by --arch graph)
│       ├── graph_structure.py
│       ├── st_gcn_layers.py
│       └── st_gcn_model.py

data/
├── asl-is-lmdb/                 # ASL Signs LMDB: is.lmdb.mdb + train.csv + sign map (download below)
├── asl-fs-lmdb/                 # Fingerspelling LMDB (download below)
├── wlasl-lmdb/                  # WLASL, 206 shared signs (private; cross-dataset test)
├── wlasl-all-lmdb/              # WLASL, all 2,000 glosses (private; wider-vocabulary training)
├── Isolated_ASL_Recognition/    # Raw ASL Signs parquets (only needed to rebuild LMDB)
└── ASL_Fingerspelling_Recognition/ # Raw Fingerspelling parquets (189 GB, rarely needed)
```

---

## Setup

Requires Python 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
```

## Data

**Recommended — download pre-built LMDB datasets:**

```bash
kaggle datasets download shravnchandr/asl-is-lmdb -p data/asl-is-lmdb --unzip
kaggle datasets download shravnchandr/asl-fs-lmdb -p data/asl-fs-lmdb --unzip
kaggle datasets download shravnchandr/wlasl-lmdb -p data/wlasl-lmdb --unzip   # private; WLASL, 206 shared signs, aspect-corrected (cross-dataset test)
kaggle datasets download shravnchandr/wlasl-all-lmdb -p data/wlasl-all-lmdb --unzip   # private; WLASL, all 2,000 glosses (2,017 labels), wlasl_half split column (train_phrases --vocab)
```

**Raw competition data** (only needed to rebuild LMDBs or run the VQ-VAE pipeline):

```bash
kaggle competitions download -c asl-signs               # Google ASL Signs (~5 GB)
kaggle competitions download -c asl-fingerspelling       # Fingerspelling (~189 GB)
```

Extract to `data/Isolated_ASL_Recognition/` and `data/ASL_Fingerspelling_Recognition/` respectively.

**Other datasets.** WLASL (C-UDA) is used for cross-dataset tests and, by signer half, for training. ASL Citizen (webcam recordings by Deaf signers) is under the Microsoft Research License — **research use only**: its landmarks are extracted privately (`scripts/extract_aslcitizen.py`) and models trained on it are not exported for the app. PopSign ASL v1.0 (CC BY 4.0, the source of ASL Signs) is the permissive option for product data.

**Parquet format:** columns `frame`, `type`, `landmark_index`, `x`, `y`, `z`. Types: `pose` (33), `left_hand` (21), `right_hand` (21), `face` (468) — 543 landmarks per frame.

---

## Training

### Approach A — isolated classifier (LandmarkConformer / GraphConformer)

**On a fresh RunPod pod (no persistent volume):**

```bash
git clone https://github.com/shravnchandr/Sign-Language-Translation.git && cd Sign-Language-Translation
mkdir -p ~/.kaggle && mv ~/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
bash scripts/setup_runpod.sh        # uv + tmux, CUDA check, LMDB from the build notebook's output, key check
bash scripts/run_pipeline_isolated.sh --skip-pretrain --num-workers 8
tmux attach -t islr         # training runs in a detached tmux session; detach with Ctrl-b d
```

The pipeline relaunches itself in tmux (session `islr`) and logs to
`logs/isolated_<timestamp>.log`; pass `--no-tmux` to run in the foreground.

**Before stopping the pod** (its disk is erased on stop), pull checkpoints and logs:

```bash
bash scripts/runpod_results.sh send run005           # on the pod — prints a one-time code
bash scripts/runpod_results.sh receive <code>        # on your machine → runs/run005/
```

```bash
# Recommended: pre-built LMDBs, skip pre-training
bash scripts/run_pipeline_isolated.sh --skip-pretrain

# With CTC pre-training (uses downloaded FS LMDB — no 189 GB download needed)
bash scripts/run_pipeline_isolated.sh

# Re-use an existing backbone checkpoint
bash scripts/run_pipeline_isolated.sh \
  --pretrained-backbone checkpoints/pretrain_fs/backbone_best.pth

# Quick smoke test
bash scripts/run_pipeline_isolated.sh --skip-pretrain \
  --phase1-epochs 2 --phase2-epochs 1
```

**Key flags:**

| Flag | Default | Purpose |
|------|---------|---------|
| `--skip-pretrain` | false | Skip CTC pre-training entirely |
| `--pretrained-backbone <path>` | — | Use existing backbone checkpoint |
| `--pretrain-epochs` | 40 | CTC pre-training epochs |
| `--pretrain-patience` | 10 | Early stopping patience (val CTC loss) |
| `--phase1-epochs` | 100 | Fine-tuning epochs |
| `--build-lmdb` | — | Rebuild ASL LMDB from raw parquets |
| `--build-fs-lmdb` | — | Rebuild Fingerspelling LMDB from raw parquets |
| `--data-dir` | `data/asl-is-lmdb` | ASL Signs dataset root |
| `--fs-data-dir` | `data/asl-fs-lmdb` | Fingerspelling dataset root |

> **LMDB cache invalidation:** any change to `config.py` that affects `ALL_COLUMNS` (face landmark selection, `INCLUDE_DEPTH`, `INCLUDE_FACE`) requires rebuilding the LMDB. `CACHE_VERSION` is an MD5 of `ALL_COLUMNS` so stale archives are never silently reused.

### Approach C — continuous phrases and browser export

```bash
# CTC on synthetic phrases, initialised from a fold-0 isolated checkpoint (GPU; inside tmux)
PYTHONPATH=research/models uv run python -m islr.phrases.train --val-fold 0 --arch graph \
  --init <fold0 best_phase1.pth> --out-dir checkpoints/phrases_fold0_graph

# Full validation breakdown (CPU)
PYTHONPATH=research/models uv run python -m islr.phrases.breakdown \
  --checkpoint <best_phrases.pth> --iso-checkpoint <fold0 best_phase1.pth> --out <breakdown.txt>

# ONNX export (+ labels.json, preprocess.json) and browser test fixtures
PYTHONPATH=research/models uv run --with onnx --with onnxruntime --with onnxscript \
  python -m islr.export.export_onnx --checkpoint <best_phrases.pth> --out-dir exports/phrases05
PYTHONPATH=research/models uv run --with onnxruntime python -m islr.export.browser_fixtures \
  --parquet <competition parquets> --concat --out-dir exports/phrases05/fixtures

# Per-sign reliability for the app (held-out ASL Signs signers + WLASL test half; CPU)
PYTHONPATH=research/models uv run python -m islr.evaluation.sign_reliability \
  --checkpoint <best_phrases.pth> --wlasl-dir data/wlasl-lmdb --out exports/phrases05/sign_reliability.json

# Cross-dataset score on WLASL's held-out signer half (paired comparisons via --per-clip)
PYTHONPATH=research/models uv run python -m islr.evaluation.cross_dataset \
  --checkpoint <best_phrases.pth> --data-dir data/wlasl-lmdb --signers test --out <json> --per-clip <csv>
```

### Training on Kaggle GPUs

`scripts/kaggle_run.py` runs any repo module as a private Kaggle notebook (T4, 12 h per run) on the pushed commit, with the private datasets mounted (`{IS_LMDB_DIR}`, `{CKPT_DIR}`, `{WLASL_DIR}`, `{WLASL_ALL_DIR}`, output in `{OUT}`); `::` chains commands. phrases04 was trained this way:

```bash
python3 scripts/kaggle_run.py push phrases04-viewpoint --gpu -- \
  islr.phrases.train --data-dir {IS_LMDB_DIR} --lmdb-path {IS_LMDB_DIR}/is.lmdb.mdb --val-fold 0 --arch graph \
    --init {CKPT_DIR}/fold0_graph/best_phase1.pth --out-dir {OUT}/phrases04 --num-workers 4 \
    --extra-data-dir {WLASL_DIR} --extra-frac 0.15 --aug-rotate 20 --aug-shear 0.15 --aug-scale 0.3 --aug-affine-prob 0.8 \
  :: islr.evaluation.cross_dataset --checkpoint {OUT}/phrases04/best_phrases.pth --data-dir {WLASL_DIR} --signers test --out {OUT}/wlasl.json
python3 scripts/kaggle_run.py status phrases04-viewpoint
python3 scripts/kaggle_run.py output phrases04-viewpoint     # → results/kaggle/ (inbox), then file under results/phrases/
```

### Approach 0 — VQ-VAE pipeline (dormant)

```bash
# Full pipeline
bash scripts/run_pipeline_vqvae_seq2seq.sh

# Phase 1 — train VQ-VAE
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
  --data-dir data/Isolated_ASL_Recognition --cache-dir data/cache --epochs 100

# Tokenize dataset with trained VQ-VAE (run once after Phase 1)
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.scripts.precompute_tokens \
  --vqvae-checkpoint checkpoints/vqvae/best_model.pt \
  --data-dir data/Isolated_ASL_Recognition --token-dir data/tokens \
  --cache-dir data/cache --num-workers 4

# Phase 2 — train translator
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.translation.train_translator \
  --token-dir data/tokens --data-dir data/Isolated_ASL_Recognition --epochs 100
```
