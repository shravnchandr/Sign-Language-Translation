# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASL sign recognition from MediaPipe landmarks over the 250-sign Google ASL Signs vocabulary — first isolated signs, now continuous multi-sign phrases (CTC) — with the goal of a free, signer-independent recogniser running in the browser (the app is the separate ASL-App repo; this repo is model work only).

Approaches, all under `research/models/` (current work in the `islr` package, organised by approach):

| Approach | Location | Status |
|---|---|---|
| 0. Factorized VQ-VAE → Conformer translator | `research/models/vqvae_seq2seq/` | Dormant (earliest approach); its discrete-bottleneck idea lives on as `islr/core/model/vq.py` |
| — ST-GCN baseline | `research/models/st_gcn/` | Legacy: superseded by `--arch graph` — wrong graph indexing, by-sign split; not comparable |
| A. Isolated-sign classifier (LandmarkConformer / GraphConformer) | `research/models/islr/isolated/` (+ `core/`) | Fold 0: 0.686 graph (0.677 conformer); ensemble 0.708 |
| B. Fingerspelling CTC pre-training | `research/models/islr/pretrain/` | No single-model gain (0.650 vs 0.661); useful ensemble member |
| **C. Continuous CTC on synthetic phrases** | `research/models/islr/phrases/` | **Current product model.** `phrases01`: WER 0.294 on held-out signers' synthetic phrases; `phrases03` (+ WLASL train-half signers, 15 % of phrases): same WER, WLASL held-out signers 44.2 → 80.1 % single-sign; `phrases04` (+ `--aug-scale 0.3 --aug-shear 0.15 --aug-rotate 20 --aug-affine-prob 0.8`): 80.8 %, and 77.4 % (phrases03 39.2 %) with WLASL's aspect correction undone (2D coordinate distortion only; augmentation chosen on that same test half — not an independent test); **`phrases05`** (+ all 2,000 WLASL glosses as extra labels, `--extra-frac 0.3`) = current browser model: WLASL 83.7 %, ASL Citizen (independent: 51 unseen Deaf signers, research only) 72.4 % vs phrases04 68.7 % (p = 2e-19); browser export in `islr/export/` |

## Repository layout

- `research/models/islr/` — current code, by approach: `core/` (shared config, backbones, data) · `isolated/` (A) · `pretrain/` (B) · `phrases/` (C, current product model) · `evaluation/` · `export/`. Run as `PYTHONPATH=research/models python -m islr.<subpackage>.<module>`. `research/models/vqvae_seq2seq/` (approach 0, dormant) and `st_gcn/` (legacy) stay alongside.
- `scripts/` — shell pipelines (`run_pipeline_isolated.sh`, `run_pipeline_vqvae_seq2seq.sh`, `setup_runpod.sh`, `runpod_results.sh`; they `cd` to the repo root themselves) Kaggle runners (`kaggle_run.py`, `extract_aslcitizen.py`) and `download_kernel_output.py` (used by `setup_runpod.sh`).
- `notebooks/` — by approach: `data/` (LMDB builders, WLASL MediaPipe extraction, `build-wlasl-*` Kaggle kernel folders; `legacy/`), `exploration/`, `early_transformers/`, `vqvae/`, `first_place/`. Index in `notebooks/README.md`.
- `archive/research/` — superseded prototypes, feature experiments and tools (reference only).
- `results/` (gitignored) is organised like the code: `isolated/<run>/`, `pretrain/<run>/`, `phrases/<run>/` (checkpoint + meta + `logs/` + eval outputs), `evaluation/` (`wlasl/` per-clip CSVs, `error_review/`, `signer_diagnostics/`); `results/kaggle/` is the inbox `kaggle_run.py output` downloads into — file each run under its approach afterwards. `exports/phrases05/` is the browser model (`exports/phrases04/` = the previous one, kept until the app has switched) (ONNX, labels, preprocess spec, `sign_reliability.json`, `fixtures/`).
- Gitignored: `data/`, `exports/`, `results/`, `docs/` (except `docs/experiments/`, the experiment log). The local ASL Signs LMDB is `data/asl-is-lmdb/is.lmdb.mdb` (+ train.csv, sign map), the same layout as the Kaggle dataset and the scripts' defaults. Paths in older experiment-log entries predate this layout (reorganised 2026-11-12: `cnn_transformer` → `islr`).

## Commands

```bash
# Install dependencies (UV package manager, Python 3.14)
uv sync

# LandmarkConformer on a fresh RunPod pod: env + tmux + LMDB download/verify, then train
bash scripts/setup_runpod.sh                     # default: saved output of Kaggle notebook shravnchandr/build-islt-lmdb
bash scripts/setup_runpod.sh --with-fs           # also the fingerspelling LMDB (~16 GiB) for CTC pre-training
bash scripts/run_pipeline_isolated.sh --arch graph --val-fold 0 --fs-cleanup   # (no --compile: crashes on GPU) FS pre-train → fine-tune; the FS LMDB is downloaded only when pre-training runs (disk check: ~35 GB peak), --fs-cleanup deletes it afterwards
bash scripts/run_pipeline_isolated.sh --skip-pretrain --num-workers 8   # relaunches in tmux session "islr"
tmux attach -t islr                      # logs also in logs/isolated_<timestamp>.log; --no-tmux = foreground
bash scripts/run_pipeline_isolated.sh --skip-pretrain --val-fold 2   # validate on signer fold 2 of 7 (GroupKFold)
PYTHONPATH=research/models uv run python -m islr.isolated.error_review --checkpoint <best_phase1.pth> --also <other.pth> --data-dir data/asl-is-lmdb --lmdb-path data/asl-is-lmdb/is.lmdb.mdb --val-fold 0 --out results/evaluation/error_review/fold0_graph_errors.html   # error-review kit (CPU)
PYTHONPATH=research/models uv run python -m islr.isolated.signer_diagnostics --logs 'logs/fold*.log'   # per-signer data stats vs val acc (CPU)
PYTHONPATH=research/models uv run python -m islr.phrases.grammar   # sample synthetic gloss phrases + vocabulary coverage
PYTHONPATH=research/models uv run python -m islr.phrases.train --val-fold 0 --arch graph --init <fold0 best_phase1.pth> --out-dir checkpoints/phrases_fold0_graph 2>&1 | tee logs/phrases_fold0_graph.log   # continuous (multi-sign) CTC on synthetic phrases (GPU; run inside tmux)
bash scripts/run_pipeline_isolated.sh --skip-pretrain --val-fold 0 --stretch-mode sample --stretch-min 0.5 --stretch-max 2.0 --stretch-prob 0.8   # per-clip tempo aug
bash scripts/run_pipeline_isolated.sh --skip-pretrain --max-frames 256   # longer cap (default 128; 5.6% of clips > 128, 0.3% > 256)
bash scripts/run_pipeline_isolated.sh --skip-pretrain --val-fold 0 --hand-drop-prob 0.5   # simulate hand-tracking gaps (default off)
bash scripts/run_pipeline_isolated.sh --skip-pretrain --val-fold 0 --arch graph   # GraphConformer (hand-skeleton graph encoder)
bash scripts/run_pipeline_isolated.sh --skip-pretrain --val-fold 0 --arch graph --bone-aug-prob 0.5   # + hand-proportion augmentation (default off)
PYTHONPATH=research/models uv run python -m islr.phrases.breakdown --checkpoint <best_phrases.pth> --iso-checkpoint <fold0 best_phase1.pth> --out results/<run>/breakdown.txt   # full val breakdown: WER/ops by length/kind/signer, downsampled vs full, emission timing, controlled single-sign vs classifier (CPU, ~15 min)
PYTHONPATH=research/models uv run --with onnx --with onnxruntime --with onnxscript python -m islr.export.export_onnx --checkpoint <best_phrases.pth> --out-dir exports/phrases05   # phrase CTC model → ONNX (onnxruntime-web) + labels.json + preprocess.json; verified against PyTorch on LMDB clips (CPU)
python3 scripts/kaggle_run.py push <name> --gpu -- islr.phrases.train --data-dir {IS_LMDB_DIR} --lmdb-path {IS_LMDB_DIR}/is.lmdb.mdb --val-fold 0 --arch graph --init {CKPT_DIR}/fold0_graph/best_phase1.pth --out-dir {OUT} --num-workers 4   # any repo module as a private Kaggle notebook (T4, 12 h max, ~30 GPU h/week); clones the pushed HEAD, mounts asl-is-lmdb / islr-checkpoints / wlasl-lmdb / wlasl-all-lmdb, copies LMDBs to local disk; `::` chains commands
# phrases04 recipe: add --extra-data-dir {WLASL_DIR} --extra-frac 0.15 --aug-rotate 20 --aug-shear 0.15 --aug-scale 0.3 --aug-affine-prob 0.8
# phrases05 (current browser model) = phrases04 recipe with the wider vocabulary: --extra-data-dir {WLASL_ALL_DIR} --extra-lmdb-path {WLASL_ALL_DIR}/wlasl_all.lmdb.mdb --vocab {WLASL_ALL_DIR}/sign_to_prediction_index_map.json --extra-frac 0.3
PYTHONPATH=research/models uv run python -m islr.evaluation.cross_dataset --checkpoint <best_phrases.pth | best_phase1.pth> --data-dir data/wlasl-lmdb --out <json>   # cross-dataset score on WLASL's 67 unseen signers (phrase: single-sign exact decode; isolated: top-1/top-5), framing cost, two-hand clips, one-hand mode + McNemar
PYTHONPATH=research/models uv run python -m islr.evaluation.sign_reliability --checkpoint <best_phrases.pth> --wlasl-dir data/wlasl-lmdb --out exports/<run>/sign_reliability.json   # per-sign accuracy / confusions / score thresholds for the app (CPU, ~15 min)
python3 scripts/kaggle_run.py status <name>; python3 scripts/kaggle_run.py output <name>   # → results/kaggle/<name>/ (inbox: out/run.log, checkpoints; then file under results/<approach>/<run>/)
uvx --with six kaggle kernels push -p notebooks/data/build-wlasl-lmdb   # WLASL → ASL-Signs-compatible LMDB on Kaggle (CPU; clones this repo, runs islr/core/data/build_wlasl_lmdb.py); status / output: uvx --with six kaggle kernels status|output shravnchandr/build-wlasl-lmdb
uvx --with six kaggle kernels push -p notebooks/data/build-wlasl-all-lmdb   # same, all 2,000 glosses (--all-glosses) → dataset shravnchandr/wlasl-all-lmdb
python3 scripts/extract_aslcitizen.py push --shards 3; python3 scripts/extract_aslcitizen.py status --shards 3   # ASL Citizen → landmark parquets (Kaggle CPU, ~3 h per shard; research use only)
python3 scripts/kaggle_run.py push aslcitizen-eval --gpu --no-copy --kernel shravnchandr/extract-aslcitizen-s0 --kernel shravnchandr/extract-aslcitizen-s1 --kernel shravnchandr/extract-aslcitizen-s2 -- islr.core.data.build_aslcitizen_lmdb --input-root {INPUT} --sign-map {IS_LMDB_DIR}/sign_to_prediction_index_map.json --out-dir /tmp/aslc :: islr.evaluation.cross_dataset --checkpoint {CKPT_DIR}/phrases04/best_phrases.pth --data-dir /tmp/aslc --lmdb-path /tmp/aslc/aslcitizen.lmdb.mdb --out {OUT}/phrases04_aslc.json --per-clip {OUT}/phrases04_aslc.csv   # ASL Citizen LMDB built and scored inside one private notebook — the data never leaves Kaggle
PYTHONPATH=research/models uv run --with onnxruntime python -m islr.export.browser_fixtures --parquet <competition parquets> --concat --out-dir exports/phrases05/fixtures   # golden fixtures for the browser port (raw frames → trim / features / logits / decode)
bash scripts/runpod_results.sh send run005       # on the pod, before stopping it (disk is erased on stop)
bash scripts/runpod_results.sh receive <code>    # on your machine → runs/run005/ (gitignored)

# VQ-VAE full pipeline (cds to the repo root and sets PYTHONPATH itself)
bash scripts/run_pipeline_vqvae_seq2seq.sh
bash scripts/run_pipeline_vqvae_seq2seq.sh --vqvae-epochs 10 --translator-epochs 10

# Train VQ-VAE (Phase 1)
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.vqvae.train_vqvae \
  --data-dir data/Isolated_ASL_Recognition --cache-dir data/cache --epochs 100

# Pre-tokenize dataset with trained VQ-VAE (run once after Phase 1)
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.scripts.precompute_tokens \
  --vqvae-checkpoint checkpoints/vqvae/best_model.pt \
  --data-dir data/Isolated_ASL_Recognition --token-dir data/tokens \
  --cache-dir data/cache --num-workers 4

# Train Translator (Phase 2) — fast path using pre-tokenized data
PYTHONPATH=research/models uv run python -m vqvae_seq2seq.translation.train_translator \
  --token-dir data/tokens --data-dir data/Isolated_ASL_Recognition --epochs 100

```

## Architecture

### Approach 0 — Factorized VQ-VAE Pipeline (dormant) (`research/models/vqvae_seq2seq/`)

**Phase 1 — Factorized VQ-VAE:**
- Encodes landmark chunks into 4 discrete tokens per chunk: `(pose_id, motion_id, dynamics_id, face_id)`
- Factorized codebooks: Pose (256), Motion (256), Dynamics (128), Face (128)
- Multi-scale temporal encoding at chunk sizes `(4, 8, 16)` via `MultiScaleMotionEncoder`
- EMA vector quantization with soft diversity loss and codebook reset for dead codes
- Cross-factor attention (`CrossFactorAttention`) fuses pose/motion/dynamics representations
- `HandDominanceModule` reorders left/right hands so dominant hand is always in the first slot
- Training is unsupervised — no labels needed; uses all available datasets

**Phase 2 — Sign Translator:**
- Input: pre-tokenized indices loaded from `data/tokens/` (no VQ-VAE in memory during training)
- Encoder: Conformer (CNN + self-attention, kernel=7, 6 layers, d_model=256)
- Decoder: Hybrid CTC + Attention decoder (`HybridDecoder`, 3 layers)
- Inference: beam search with CTC prefix scoring (`BeamSearch`)
- 250-class supervised classification using Google ASL Signs labels

**Data Flow:**
```
Parquet → LandmarkProcessor → (T, N, 3)
  → RobustPreprocessor → HandDominanceModule
  → MultiScaleMotionEncoder → CrossFactorAttention
  → FactorizedVectorQuantizer → [(pose_id, motion_id, dyn_id, face_id), ...]  [saved to data/tokens/]
  → FactorizedTokenEmbedding → Conformer → HybridDecoder → 250-class output
```

### Approaches A–C — LandmarkConformer backbone (`research/models/islr/`)

End-to-end supervised classification. Optional CTC pre-training on ASL Fingerspelling initialises the backbone before fine-tuning on the 250-class task. Designed for Kaggle training.

- Input coordinates: x, y, z per landmark (`INCLUDE_DEPTH=True`)
- Per-body-part projection: separate `nn.Linear` for LH, RH, pose; face is split into `eyebrow_proj` (grammatical: questions/negation) and `mouth_proj` (phonological: mouthing), each at `d_model//8` — same total budget as a single face projection
- Multi-scale velocity stream (Δ1/Δ2/Δ5): Δ1 computed in dataset from the stored (body-relative) coords; Δ2 and Δ5 computed inside `forward()` from body-relative positions (after nose subtraction). All three scales concatenated per part and projected to `d_model//4`. Position and velocity get equal `d_model` budget before fusion.
- Geometry stream: per hand, 15 joint-angle cosines (3 per finger, at MCP/PIP/DIP joints) + 10 fingertip pairwise distances + 3 palm-normal components (3D only) = 28 features/hand. Computed in `_hand_geometry()` from wrist-relative fingers after `WristNormalization`. Projected 2 × `d_model//8` = `d_model//4`. Invariant to wrist rotation and signer hand scale — encodes fine-grained hand shape that raw XYZ obscures.
- Distance stream: dominant/non-dominant wrist-to-nose xy distance + `dom_ratio` (dominant share of wrist energy, 0.5–1) + per-frame hand presence flags (lh, rh) → `dist_proj` → `d_model//8`. Tells the model when hands are in the face region, and distinguishes a missing hand (stored as 0 = nose origin) from a hand at the face.
- Input layout: `(B, T, IN_FEAT)` = `[pos (COORD_FEAT) | Δ1 vel (COORD_FEAT) | presence (2)]` (`config.PRESENCE_START`, `config.IN_FEAT`). Presence is derived in the datasets by `hand_presence()` (see Key Patterns) — no LMDB rebuild needed.
- Feature fusion: pos (`d_model`) + vel (`d_model`) + geo (`d_model//4`) + dist (`d_model//8`) → `feat_fuse` → `d_model`
- Shoulder-width scaling in `forward()`: pos and vel divided by the per-sequence mean shoulder width (camera-distance / body-size invariance).
- Conformer blocks (depthwise conv + self-attention) + CLS token for classification
- Body-relative normalization done once at LMDB build time (`normalize_values`: nose → shoulder → hip → 0 fallback). `WristNormalization` applied in-model: landmark 0 = location (nose-relative), landmarks 1–20 = shape (wrist-relative).
- **Optional fingerspelling CTC pre-training** (`islr/pretrain/fingerspelling.py`): trains the backbone on ASL Fingerspelling (67,208 phrases, **94 signers**, none sharing an ID with ASL Signs' 21) with `nn.CTCLoss` (59-char vocab + blank). CTC mode skips the CLS token and returns per-frame logits `(B, T, vocab+1)`. `--arch {conformer,graph}` (+ `--graph-*`) builds the same backbone as the fine-tune; `backbone_meta.json` records it and `train.py` exits if `--pretrained-backbone` was pre-trained with a different arch / width / depth / heads. Signer-held-out 10% val (`GroupShuffleSplit` on `participant_id`); **best checkpoint and early stopping on greedy-decoded val CER** (`--epochs 20`, `--patience 5` by default). Length-bucketed batches (`BucketBatchSampler`, lengths from the `fs:len:` keys): FS phrases average 160 frames (cap 384), and random batches were padded to ~357 — bucketing → ~164, about half the compute (est. ~8 min/epoch for the graph model on the pod) — loss alone cannot show letters are recognised. Training-only augmentation (`--no-augment` disables): `augment_sample` noise/shift + optional `bone_scale` in `FingerspellingDataset`, per-clip stretch 0.8–1.25× (p 0.5; CTC lengths follow the mask) and spatial affine in the loop; no mixup, hand or finger dropout (they would delete letters the target contains). `--seed`, bf16 autocast on Ampere+ (fp16 + GradScaler otherwise), non-finite-loss guard restoring BatchNorm buffers. CTC loss in FP32. Saves `backbone_best.pth` (all keys except `head.`, `ctc_head.`, `signer_disc.`, `cls_token`). Loaded at fine-tuning start via `--pretrained-backbone` with `strict=False`. FS LMDB 15.9 GiB (Kaggle rebuild 2026-11-10: 0 errors, 4 min); keys `fs:{sequence_id}` carry no preprocessing version — `scripts/setup_runpod.sh --with-fs` downloads `shravnchandr/asl-fs-lmdb` and verifies keys and sample width.
- **Backbone warmup** (`--backbone-warmup-epochs`, default 5): when `--pretrained-backbone` is set, freezes all backbone params for the first N epochs and trains only the params the checkpoint did not provide (the `missing` keys from `load_state_dict` — always `head.`, `cls_token`, `signer_disc.`, plus any layer added after the backbone was saved). After warmup, backbone is unfrozen with `--backbone-lr-ratio` (default 0.1) × head LR via a separate param group. Patience counter resets at warmup end so early stopping counts from the first joint-training epoch. GRL disabled during warmup (backbone is frozen so adversarial gradients cannot update it). Set `--backbone-warmup-epochs 0` to disable.
- Loss: `FocalLoss` with per-class inverse-frequency weights (computed from train.csv, mean-normalised, registered as buffer), γ=2.0, label smoothing=0.1. Replaces the prior scalar `alpha=0.25` which was a 4× global loss scaling rather than class-imbalance handling.
- Training: Phase 1 (100 epochs default, heavy aug, mixup, OneCycleLR, early stopping on deterministic val acc). Optional Phase 2 cosine warmdown (`--phase2-epochs`, default 0 — it never beat Phase 1 in Runs 002/003).
- Test-time augmentation (5-pass TTA) at evaluation
- Stochastic depth (`drop_path_max=0.1`): linearly increasing per-block skip probability (block 0 = 0, last = drop_path_max). Controlled via `--drop-path-max` CLI arg.
- GRL signer-invariance (`--grl-lambda 0.1`): `SignerDiscriminator` on CLS token, gradient reversed so feature extractor is forced to discard signer identity. λ is applied once, inside the gradient reversal (`loss = sign_loss + adv_loss`), so the backbone receives −λ·∇ and the discriminator trains at full rate. Ganin schedule ramps λ from 0 → max over Phase 1; Phase 2 continues the ramp from Phase 1's endpoint (stays at ~max_lambda). Requires `participant_id` in train.csv; auto-disables when not present. Adversarial loss uses same mixup weighting as sign loss. Discriminator accuracy is logged each epoch alongside chance level (`1/n_signers`) to verify the feature extractor is successfully confusing the discriminator.

## Key Modules

### VQ-VAE pipeline (`research/models/vqvae_seq2seq/`)

| File | Purpose |
|------|---------|
| `vqvae/config.py` | `ImprovedVQVAEConfig` — all hyperparameters |
| `vqvae/vqvae_model.py` | Main VQ-VAE model (assembles all sub-modules) |
| `vqvae/vector_quantizer.py` | `EMAVectorQuantizer`, `FactorizedVectorQuantizer` |
| `vqvae/multi_scale_encoder.py` | Multi-scale motion encoding |
| `vqvae/face_encoder.py` | Dedicated face NMM encoder (5 regions) |
| `vqvae/hand_dominance.py` | Detects & reorders dominant/non-dominant hands |
| `vqvae/cross_attention.py` | `CrossFactorAttention` fuses pose/motion/dynamics |
| `scripts/precompute_tokens.py` | Pre-tokenize dataset with frozen VQ-VAE |
| `data/preprocessing.py` | `RobustPreprocessor`, `LandmarkProcessor` |
| `data/dataset.py` | `VQVAEDataset`, `TranslationDataset`, `TokenizedTranslationDataset` |
| `translation/translator_model.py` | `SignTranslator` (full model) |
| `translation/conformer.py` | Conformer encoder blocks |
| `translation/decoder.py` | `HybridDecoder` (CTC + attention) |
| `translation/beam_search.py` | Beam search with CTC prefix scoring |
| `translation/config.py` | `TranslationConfig` |

### `islr` package (`research/models/islr/`)

Organised by approach: `core/` (shared config, backbones, data) · `isolated/` (A: isolated-sign classifier) · `pretrain/` (B: fingerspelling CTC pre-training) · `phrases/` (C: continuous CTC on synthetic phrases — current product model) · `evaluation/` (cross-dataset, per-sign reliability) · `export/` (browser ONNX + fixtures). Run modules as `python -m islr.<subpackage>.<module>`.

| File | Purpose |
|------|---------|
| `core/config.py` | Landmark layout constants, feature dimensions |
| `core/model/landmark_conformer.py` | `LandmarkConformer` — main model |
| `core/model/conformer.py` | `ConformerBlock`, `SinusoidalPositionalEncoding` |
| `core/model/normalization.py` | `WristNormalization` |
| `core/model/grl.py` | `SignerDiscriminator`, `ganin_lambda` — GRL signer-invariance |
| `core/model/graph_conformer.py` | `GraphConformer` — `LandmarkConformer` with hand streams from a shared hand-skeleton graph encoder (`--arch graph`) |
| `core/model/vq.py` | `EMAQuantizer`, `PartVQ` — optional per-frame discrete bottleneck, one EMA codebook per body-part group (`--vq-codes`) |
| `core/model/supcon.py` | `CrossSignerSupCon` — cross-signer supervised contrastive loss with cross-batch memory |
| `core/data/dataset.py` | `ASLDataset`, `BucketBatchSampler`, `get_data_loaders` |
| `core/data/augmentation.py` | `AdvancedAugmentation` (7 types), `mixup_batch` |
| `core/data/preprocessing.py` | `frame_stacked_data` — parquet → numpy array |
| `core/data/build_wlasl_lmdb.py` + `core/data/wlasl_gloss_map.json` | WLASL landmark parquets (shared signs: 233 glosses → 206 signs, `--strict` = identical names) → ASL-Signs-compatible LMDB (same `frame_stacked_data`, same keys) with per-clip aspect correction (x, z scaled so shoulder drop / width = 0.40; WLASL is landscape) and train.csv (participant_id = 900000 + signer, fps, source, synonym, aspect_k, both_hands); `signer_split()` = fixed clip-balanced signer halves (train 33 / test 34); `--all-glosses --lmdb-name wlasl_all.lmdb.mdb` keeps all 2,000 glosses (extra labels after the 250, `wlasl_half` column = the same split) |
| `core/data/build_lmdb.py` | One-time LMDB archive builder (parallelised — `os.cpu_count()` workers by default) |
| `core/data/build_fingerspelling_lmdb.py` | Fingerspelling LMDB builder — parquet-file-level parallelism (one worker per ~1 GB file) |
| `core/data/fingerspelling_dataset.py` | `FingerspellingDataset`, `collate_ctc`, `load_char_map` — CTC pre-training data pipeline |
| `core/data/_cache_keys.py` | `CACHE_VERSION` hash, `lmdb_key`/`lmdb_length_key` helpers |
| `core/data/build_aslcitizen_lmdb.py` | ASL Citizen landmark shards (`extract_aslcitizen.py` outputs, found by `shard_*.csv` under `--input-root`) → LMDB + train.csv in the WLASL builder's format (same worker: `frame_stacked_data` + aspect correction; participant_id = 800000 + participant; `kind` from the gloss map; official `split`). **Research use only** — build in a private notebook, never publish |
| `core/data/aslcitizen_gloss_map.json` | ASL Citizen gloss → ASL Signs label (225 signs, 8,557 clips): `identical` (171 glosses), `variant` (89 numbered ASL-LEX variants, may be different sign forms), `synonym` (20, chosen by a non-signer) — score the kinds separately. **Research use only** (see Data) |
| `pretrain/fingerspelling.py` | CTC pre-training loop — saves `backbone_best.pth` for fine-tuning |
| `isolated/train.py` | Two-phase training loop with TTA evaluation |
| `isolated/error_review.py` + `isolated/error_review_template.html` | Builds a self-contained HTML kit to review a checkpoint's val errors: blind A/B comparison (query vs true-sign / predicted-sign references from other signers), tracking gaps drawn faded, answers in localStorage + CSV/JSON export |
| `isolated/signer_diagnostics.py` | Per-signer data stats (hand detection, dominance, speed, framing) from the LMDB, rank-correlated with per-signer val acc parsed from training logs |
| `phrases/grammar.py` | `PhraseGrammar` — ASL-style gloss templates over the 250 signs (+ English rendering) and random sign sequences |
| `phrases/dataset.py` | `SyntheticPhraseDataset` — one-signer phrases stitched on the fly from ASL Signs clips (trim, per-clip tempo, interpolated transitions, phrase-level hold-fill) |
| `phrases/train.py` | CTC training on synthetic phrases (gloss WER / exact / single-sign on held-out-signer phrases); `--init` from an isolated-sign or FS backbone; `--extra-data-dir <wlasl-lmdb>` mixes WLASL's train-half signers in (`--extra-frac` of phrases, `ConcatSource` + signer weights); `--vocab <wlasl-all map>` widens the label set (first 250 unchanged; `load_ctc_model` cuts a wider checkpoint back to 250 + blank, exactly); `--aug-shear` / `--aug-scale` (training-only viewpoint augmentation, default off) |
| `evaluation/cross_dataset.py` | Scores a phrase or isolated checkpoint on the WLASL or ASL Citizen LMDB (`--lmdb-path`, `--split` = official ASL Citizen split, accuracy per gloss `kind`) (67 unseen signers, 206 shared signs; `--signers test` = the held-out half of `build_wlasl_lmdb.signer_split`): phrase models exact decode + classifier-style pooled top-1/top-5 + decode-contains, accuracy aspect-corrected and with the correction undone, identical vs synonym glosses, clips with vs without both hands, one-hand mode on two-hand clips (McNemar), per-signer spread; JSON out, `--per-clip` CSV for paired comparisons. Reproduces the hand-measured phrases01 (45.4 %) and fold0_graph (64.2 %) numbers |
| `export/browser_fixtures.py` | Golden fixtures for a browser port: competition parquets (+ `--concat` phrase) → raw per-frame landmarks, `segment`, `subsample`, expected `(T, IN_FEAT)` features, ONNX logits and greedy decode. Built from the raw frames by an explicit contract (drop no-pose frames, hand-span segment, nose origin, presence = actual detection, hold-fill, subsample to `--max-frames` before Δ1); the training pipeline is a reported cross-check (it can misplace a lone detection) |
| `phrases/breakdown.py` | Rebuilds a phrase run's fixed val set and reports WER / sub / del / ins by length, kind and signer, deletions with vs without the `--max-frames` downsample, CTC emission timing vs clip spans, and a McNemar single-sign comparison against an isolated classifier on the same trimmed clips |
| `evaluation/sign_reliability.py` | Per-sign reliability of a phrase checkpoint on held-out signers (ASL Signs val fold + WLASL test half), scored as the app can (pooled per-sign max frame probability): top-1/3/5, exact, top-3 confusions, calibration table and precision thresholds → `sign_reliability.json` next to the ONNX export |
| `export/export_onnx.py` | Exports a phrase (CTC) checkpoint to ONNX for the browser: input `features` (1, T, IN_FEAT), 8 ≤ T ≤ 512, all-valid mask built in; writes `labels.json` and `preprocess.json` (landmark order, origin, hold-fill, hand slots by pose wrist, Δ1); exits if ONNX ≠ PyTorch logits |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `run_pipeline_isolated.sh` | Approach A on RunPod: optional FS LMDB + pre-training → fine-tune, in tmux |
| `run_pipeline_vqvae_seq2seq.sh` | Approach 0 pipeline (dormant) |
| `setup_runpod.sh` / `runpod_results.sh` | Fresh-pod setup (env, LMDB download/verify) / move results off the pod |
| `download_kernel_output.py` | Streams a Kaggle notebook's saved output to disk (multi-GB safe, resumable); used by `setup_runpod.sh` for the LMDBs |
| `kaggle_run.py` | `push <name> [--gpu] -- <module> <args> [:: <module> <args> ...]` / `status` / `output`: runs a repo module as a private Kaggle notebook `islr-<name>` (repo at the pushed HEAD, datasets mounted, placeholders {IS_LMDB_DIR} {CKPT_DIR} {WLASL_DIR} {WLASL_ALL_DIR} {INPUT} {OUT}, `--kernel <user>/<slug>` mounts another notebook's output, log in out/run.log) |
| `extract_aslcitizen.py` | `push --shards N [--limit k --name …]` / `status`: MediaPipe Holistic 0.10.14 landmarks from the ASL Citizen videos on Kaggle CPU, sharded private notebooks `extract-aslcitizen-s<k>` (pinned image as the WLASL extraction; only mapped glosses unless `--all`). Research use only |

## Known Bugs

| File | Line | Issue |
|------|------|-------|
| `research/models/vqvae_seq2seq/vqvae/vqvae_model.py` | 348–353 | `decode()` routes codebooks into fusion slots with mismatched names (pose_q→`dominant_hand`, motion_q→`non_dominant_hand`, dynamics_q→`pose`). Deliberate so every codebook gets reconstruction gradient, but the slot names are misleading — rename the fusion slots to the factor names. |
| `research/models/vqvae_seq2seq/vqvae/hand_dominance.py` | 173–193 | `HandMirrorAugmentation` doesn't correctly flip x-coordinates. |
| `research/models/vqvae_seq2seq/translation/train_translator.py` | — | ~~Chunk size hardcoded as `8` when computing encoder lengths.~~ **Fixed: uses `vqvae.config.base_chunk_size`.** |
| `research/models/st_gcn/st_gcn_model.py` | 106–110 | `edge_importance` parameter is allocated but never used in `forward()`. |
| `research/models/st_gcn/st_gcn_training.py` | 219–225 | Double-normalizes adjacency matrix: `LandmarkGraph.get_normalized_adjacency()` already normalizes, then chain-edges are added and it's normalized again. |
| `research/models/islr/core/data/preprocessing.py` | — | ~~BASE_PATH double-prefixing: `frame_stacked_data` prepended `BASE_PATH` to an already-absolute path built by `dataset.py`. Fixed: `pd.read_parquet(file_path)` directly.~~ **Fixed.** |
| `research/models/islr/core/model/landmark_conformer.py` | — | ~~Hand dominance swap inverted: `lh_energy > rh_energy` triggered swap when LH was already dominant.~~ **Fixed: `rh_energy > lh_energy`.** |
| `research/models/islr/core/model/normalization.py` | — | ~~`RobustNormalization` mutated input in-place, corrupting TTA source tensors.~~ **Fixed: clones output before writing.** |
| `research/models/islr/core/data/augmentation.py` | — | ~~`random_flip` double-negated velocity x-coords (two loop passes, each covering full tensor).~~ **Fixed: single pass.** |
| `research/models/islr/core/data/augmentation.py` | — | ~~`random_flip` swapped only hand blocks; pose limbs and eyebrows/lip corners stayed un-swapped, producing anatomically impossible mirrors.~~ **Removed: redundant once `HandDominanceModule` mirrors every right-dominant input.** |
| `research/models/islr/core/data/preprocessing.py` | — | ~~A never-detected hand was stored as 0 = "at the nose", indistinguishable from a hand at the face; the FS LMDB also wrote 0 for every missing frame.~~ **Fixed: `hand_presence()` flags + gap filling in both datasets.** |
| `research/models/islr/core/data/augmentation.py` | — | ~~`augment_sample` spatial shift drew an independent ±0.02 offset per feature column, distorting hand shape (≈ 0.22 mean change in joint-angle cosines vs 0.28 natural spread).~~ **Fixed: one rigid offset per axis.** |
| `research/models/islr/core/data/augmentation.py` | — | ~~`time_stretch` cropped stretched batches back to T, cutting the last 10–23% of nearly every sign (buckets make most samples ≈ T).~~ **Fixed: returns the longer batch.** |
| `research/models/islr/core/model/landmark_conformer.py` | — | ~~`HandDominanceModule` swapped LH/RH slots without mirroring, so the dominant slot held right-hand anatomy for some signers and left-hand anatomy for others.~~ **Fixed: full horizontal mirror (`MIRROR_PERM` + x negation).** |
| `research/models/islr/isolated/train.py` | — | ~~GRL λ applied twice (inside `grad_reverse` and as `grl_lam * adv_loss`): backbone got −λ², discriminator trained at λ×.~~ **Fixed: `loss = sign_loss + adv_loss`.** |
| `research/models/islr/isolated/train.py` | — | ~~Per-frame σ=0.01 noise on all features in the train loop — as large as the whole Δ1 velocity signal and ≈ 0.19 change in joint-angle cosines.~~ **Removed** (dataset-level σ=3e-3 noise before velocity remains). |
| `research/models/islr/isolated/train.py` | — | ~~Train accuracy scored against `y_a` only under mixup, roughly halving it (Runs 002/003 logged ~0.42).~~ **Fixed: `lam·acc_a + (1−lam)·acc_b`.** |
| `research/models/islr/isolated/train.py` | — | ~~Warmup "head" prefixes hardcoded `lh_geo_proj`/`rh_geo_proj`/`feat_fuse` (present in any current backbone) and omitted `dist_proj`.~~ **Fixed: head = keys missing from the checkpoint.** |
| `research/models/islr/core/data/augmentation.py` | — | ~~`mixup_batch` discarded the shuffled sample's mask.~~ **Fixed: returns `mask \| mask[index]`.** |
| `research/models/islr/isolated/train.py` | — | ~~Class weights built from `value_counts()` of present labels only — an absent class shortened the vector and shifted every later weight onto the wrong class.~~ **Fixed: reindexed over `range(NUM_CLASSES)`.** |
| `research/models/islr/isolated/train.py` | — | ~~OneCycleLR `steps_per_epoch=len(train_loader)` ignored gradient accumulation, making schedule 2–4× slower.~~ **Fixed: `total_steps` computed from actual optimizer step counts per phase.** |
| `research/models/islr/isolated/train.py` | — | ~~Phase 2 loop `range(epoch_idx+1, total_steps)` ran too many epochs after early stopping.~~ **Fixed: `range(NUM_EPOCHS_PHASE2)`.** |
| `research/models/islr/isolated/train.py` | — | ~~Validation used 5× stochastic TTA, making checkpoint selection noisy.~~ **Fixed: `evaluate_epoch` is deterministic; TTA reserved for final reporting via `evaluate_epoch_tta`.** |
| `research/models/islr/core/model/conformer.py` | — | ~~Depthwise conv ran over padded positions, leaking zeros into valid boundary frames.~~ **Fixed: conv residual zeroed at padded positions.** |
| `research/models/islr/core/data/augmentation.py` | — | ~~`mixup_batch` paired lh-dominant with rh-dominant samples, producing ambiguous hand slot assignments when `HandDominanceModule` runs inside the model.~~ **Fixed: dominance-aware pairing shuffles within same-dominance groups.** |
| `research/models/islr/core/data/preprocessing.py` + `model/landmark_conformer.py` | — | ~~Double normalization: `normalize_values` zeroed the nose before LMDB, so `RobustNormalization` in the model always fell through to the shoulder-center branch (nose appeared missing).~~ **Fixed: `normalize_values` implements the full nose→shoulder→hip fallback; `RobustNormalization` removed from model.** LMDB must be rebuilt (`_NORM_VERSION` bump auto-invalidates). |
| `research/models/islr/isolated/train.py` | — | ~~`FocalLoss` used scalar `alpha=0.25` — a 4× global loss scaling, not per-class weighting.~~ **Fixed: inverse-frequency per-class weights from train.csv, mean-normalised, registered as a buffer.** |
| `research/models/islr/isolated/train.py` | — | ~~A single NaN batch under fp16 AMP poisoned the conv-module BatchNorm running stats: GradScaler protected the weights, but eval (running stats) collapsed to chance permanently while train mode looked fine (Run 004, epoch 55).~~ **Fixed: bf16 autocast on Ampere+; non-finite-loss guard restores BN buffers and skips the backward.** |
| `research/models/islr/isolated/train.py` | — | ~~TTA held 5 full-batch logit tensors simultaneously (OOM risk on long sequences).~~ **Fixed: running sum accumulation; only one extra tensor in memory at a time.** |
| `research/models/islr/core/model/conformer.py` | — | ~~`SinusoidalPositionalEncoding` raised `IndexError` when `T > max_len=512`.~~ **Fixed: on-the-fly PE generation in `forward()` without mutating the registered buffer (thread-safe).** |
| `research/models/islr/core/config.py` + `model/landmark_conformer.py` | — | ~~`SELECTED_FACE_INDICES` built by iterating `FACE_LANDMARK_INDICES.values()` — a dict key reordering would silently corrupt the eyebrow/mouth slice in the model.~~ **Fixed: explicit key ordering in config.py + runtime assertion in `LandmarkConformer.__init__`.** |
| `research/models/islr/core/data/preprocessing.py` + `model/landmark_conformer.py` | — | ~~`normalize_values` subtracted the pose-nose z from hand/face z (different MediaPipe depth frames): stored wrist z ≈ 1.5 + jitter, 52–82% of wrist velocity energy, dominating dominance energy and hand–nose distance.~~ **Fixed in-model (`_fix_depth`, xy for dominance/distance/scale); no LMDB rebuild.** |
| `research/models/islr/core/data/augmentation.py` | — | ~~`finger_dropout_batch` zeroed fingers before wrist subtraction → finger = −wrist.~~ **Fixed: collapse onto the wrist.** |
| `research/models/islr/core/model/landmark_conformer.py` | — | ~~`_hand_geometry` divided fingertip distances by mean MCP distance + 1e-6: when finger dropout collapsed all four MCPs but not the thumb, features reached 2.6e5 (0.15% of training clips ≈ 9% of batches, each such batch's update dominated by one clip under gradient clipping; exposed by the VQ commitment loss, 5e4). Every run since Run 004 trained with it; eval unaffected (no finger dropout).~~ **Fixed: distance features are 0 when hand scale ≤ 0.02 (real hands ≥ 0.036).** |
| `research/models/islr/core/data/augmentation.py` + `train.py` | — | ~~Batch stretch and TTA tempo jitter interpolated Δ1 with positions (inconsistent velocity) and blended short clips' last frames with padding.~~ **Fixed: `_resample_prefix` (valid frames only, Δ1 rebuilt).** |
| `research/models/islr/core/data/dataset.py` | — | ~~Never-detected hands got random geometry from noise/shift in training only.~~ **Fixed: re-zeroed after `augment_sample`; TTA noise skips them.** |
| `research/models/islr/core/model/conformer.py` | — | ~~Padding leaked into the depthwise conv (LayerNorm/pointwise biases) and BatchNorm statistics (max Δlogit 0.165 on a 6-frame clip).~~ **Fixed: re-mask before conv + `MaskedBatchNorm1d`.** |
| `research/models/islr/core/model/landmark_conformer.py` | — | ~~`forward()` mutated its input in place (same tensor twice → Δlogit 7.8).~~ **Fixed: clone at the boundary.** |
| `research/models/islr/core/model/landmark_conformer.py` | — | ~~Dominance energies averaged over padded frames (100 pad frames: dom_ratio 0.998 → 0.969); zero motion → 0.~~ **Fixed: mask-averaged, neutral 0.5.** |
| `research/models/islr/core/data/augmentation.py` + `train.py` | — | ~~Mixup on raw clips could flip dominance before the in-model mirror; different-length partners left single-source tails under the union mask; TTA noise broke Δ1 and wrote into padding.~~ **Fixed: canonicalise before mixup, time-aligned partners, anchor mask; noise on valid positions + Δ1 rebuilt.** |
| `research/models/islr/core/data/dataset.py` | — | ~~`BucketBatchSampler` fixed batch membership for all epochs.~~ **Fixed: per-epoch pooled sort.** |
| `research/models/islr/isolated/train.py` + `signer_diagnostics.py` | — | ~~Clean-train line printed between best-val and its per-signer line → diagnostics parsed nothing.~~ **Fixed: order restored; parser tolerant.** |
| `research/models/islr/core/data/dataset.py` | — | ~~Length sidecar `_lengths_<CACHE_VERSION>.json` ignored `max_frames` and split membership — a run with a different cap (or an equal-size split) reused stale lengths, mis-bucketing batches.~~ **Fixed: keyed on CACHE_VERSION + max_frames + md5 of the split's paths.** |
| `research/models/islr/core/data/dataset.py` | — | ~~LMDB opened with `map_size=1<<40` (1 TiB) on flat files over NAS — caused `lmdb.open()` to hang for 45+ minutes at mmap time.~~ **Fixed: flat-file reads use `os.path.getsize() + 256 MB` as map_size.** |
| `research/models/islr/pretrain/fingerspelling.py` | — | ~~CTC loss computed inside `autocast` (FP16) — underflows with long sequences (max_frames=384).~~ **Fixed: `logits.float().log_softmax(-1)` before `CTCLoss`.** |
| `research/models/islr/pretrain/fingerspelling.py` | — | ~~`scheduler.step()` called unconditionally even when `GradScaler` skipped `optimizer.step()` on NaN/Inf gradients, desynchronising `OneCycleLR`.~~ **Fixed: compare `scaler.get_scale()` before/after to detect skips.** |
| `research/models/islr/pretrain/fingerspelling.py` | — | ~~Last 1–3 batches per epoch dropped when `len(train_loader) % accum_steps != 0` — gradients accumulated but never applied.~~ **Fixed: explicit `_optimizer_step()` flush after the training loop if `remainder > 0`.** |

## Data

### Datasets (under `data/`)

**Pre-built LMDB datasets (recommended — download from Kaggle):**
- `data/asl-is-lmdb/` — `is.lmdb.mdb` + `train.csv` + `sign_to_prediction_index_map.json`
- `data/asl-fs-lmdb/` — `fs.lmdb.mdb` + `train.csv` + `character_to_prediction_index.json`
- `data/wlasl-lmdb/` — `wlasl.lmdb.mdb` + `train.csv` + `sign_to_prediction_index_map.json` (private Kaggle dataset `shravnchandr/wlasl-lmdb`, from `notebooks/data/build-wlasl-lmdb`; same keys as `is.lmdb.mdb`; participant ids 900000 + WLASL signer; WLASL is C-UDA)
- `data/wlasl-all-lmdb/` — `wlasl_all.lmdb.mdb` + `train.csv` (with `wlasl_half`) + extended sign map (2,017 labels: the 250 first, then 1,767 WLASL glosses); private dataset `shravnchandr/wlasl-all-lmdb`, from `notebooks/data/build-wlasl-all-lmdb`. Test-half signers identical to `wlasl-lmdb`'s split
- ASL Citizen (Kaggle `abd0kamel/asl-citizen`, raw videos, 83,399 clips / 2,731 glosses / 52 Deaf signers, official signer-disjoint train/val/test): **Microsoft Research License — research use only, no redistribution**. Landmarks are extracted privately by `scripts/extract_aslcitizen.py`; models trained on it must never be exported for the app. Not mirrored, landscape webcam (aspect correction as for WLASL). PopSign ASL v1.0 (CC BY 4.0) is the permissive product-data option

```bash
kaggle datasets download shravnchandr/asl-is-lmdb -p data/asl-is-lmdb --unzip
kaggle datasets download shravnchandr/asl-fs-lmdb -p data/asl-fs-lmdb --unzip
kaggle datasets download shravnchandr/wlasl-lmdb -p data/wlasl-lmdb --unzip   # private; WLASL, 206 shared signs, aspect-corrected (cross-dataset test)
kaggle datasets download shravnchandr/wlasl-all-lmdb -p data/wlasl-all-lmdb --unzip   # private; WLASL, all 2,000 glosses (2,017 labels), wlasl_half split column (train_phrases --vocab)
```

Rebuilt on Kaggle (CPU) by `notebooks/data/build_asl_signs_lmdb.ipynb` (ASL Signs) and `notebooks/data/build_fingerspelling_lmdb.ipynb` (fingerspelling; output byte-identical to `build_fingerspelling_lmdb.py`, reads only the 393 needed columns, builds on the disk with most free space since the LMDB is 15.9 GiB — 4 min on Kaggle CPU, publishes a new version of `shravnchandr/asl-fs-lmdb` or creates it; `INCLUDE_SUPPLEMENTAL` adds the supplemental set, untested). `notebooks/data/legacy/create-*-lmdb.ipynb` are the older versions.

**1st-place anchor (Kaggle):** `notebooks/first_place/build_signer_fold_tfrecords.ipynb` (CPU) writes the competition parquets as GZIP TFRecords in the 1st-place format (raw (T, 543, 3) with NaNs, + `participant_id`/`sequence_id`), sharded `fold{k}-{shard}-{count}.tfrecords` by our 7 signer folds (hard-coded = `GroupKFold(7)` of `--val-fold`), ~18.5 GiB → dataset `shravnchandr/islr-tfrecords-signer-folds`. `notebooks/first_place/train_fold0.ipynb` (GPU) runs the published 1st-place cells verbatim on fold 0, with listed patches: TF 2.15.1 + tensorflow-addons 0.23 (+ numpy<2, pandas 2.2.3, pyarrow 17), `tf_utils` inlined (MIT), GPU strategy with the original's replica scaling, mixed_float16 on non-bf16 GPUs, `EPOCHS` + an 11 h `TimeLimit`, and a final per-signer evaluation saving `fold0_val_probs.npz`. Both generated by a scratch script and smoke-tested locally (Python 3.11 / TF 2.15, 560 synthetic records).

**Raw competition data (only needed to rebuild LMDBs or for VQ-VAE pipeline):**
- `data/Isolated_ASL_Recognition/` — Google ASL Signs (94k parquets, 250 signs)
- `data/ASL_Fingerspelling_Recognition/` — Fingerspelling parquets (189 GB)
- `data/WLASL_Landmarks/` — WLASL landmarks after MediaPipe preprocessing

### Parquet Format
Columns: `frame`, `type`, `landmark_index`, `x`, `y`, `z`
- `type`: `'pose'` (33), `'left_hand'` (21), `'right_hand'` (21), `'face'` (468) — 543 total per frame
- Coordinates normalized to [0, 1] by MediaPipe; further normalized body-relative at LMDB build time

## Key Patterns

**Normalization fallback chain** (`RobustPreprocessor`, `RobustNormalization`): nose → shoulder center → hip center. Subtracts the origin to make coordinates body-relative. Falls back when nose landmark is missing.

**Multi-scale velocity** (`LandmarkConformer`): three temporal scales are fed to the velocity projections. Δ1 is computed in `ASLDataset` as frame differences of the stored coordinates (already nose-relative from LMDB build). Δ2 and Δ5 are computed inside `LandmarkConformer.forward()` from body-relative positions after wrist normalization and shoulder-width scaling, so they capture velocity relative to body movement. All three scales are concatenated per body part before projection.

**Hand dominance** (`HandDominanceModule`): detects dominant hand from wrist xy velocity energy averaged over valid frames (stored wrist z is a depth-frame artifact; no motion → neutral `dom_ratio` 0.5). Training canonicalises clips (mirrors them) *before* mixup and calls the model with `canonical=True`, so a mixture is never re-mirrored. In `LandmarkConformer`, sequences where right-hand energy exceeds left-hand energy are horizontally mirrored (negate x, swap every bilateral landmark via `config.MIRROR_PERM`), so the dominant hand always arrives in the first (LH) slot with the same anatomy. There is therefore no mirror-flip augmentation (it would be undone on all but near-tie sequences). The presence flags are swapped with their hands. The VQ-VAE pipeline still uses its own slot-reordering implementation.

**Hand presence** (`islr/core/data/preprocessing.py:hand_presence`): neither LMDB stores NaNs. The ASL LMDB ffill/bfills gaps (a missing frame is a bit-exact copy of its neighbour); the FS LMDB writes 0 per missing frame. A real detection is never all-zero nor bit-identical to the previous frame, so `present = block ≠ 0 and block ≠ previous block` (a leading identical run is resolved as bfill). Absent frames are then filled by holding the nearest detection so both datasets share one convention. Must run on stored coords **before** `augment_sample` noise. Validated: 2/1064 flag errors on ASL-format parquets, 0/27339 on FS frames.

**Soft diversity loss** (`EMAVectorQuantizer`): computed from the distance matrix using `softmax(-distances)` before the argmin step. Gradients flow through `z_flat` to the encoder, pushing it toward spread-out representations. The codebook (EMA buffer) is detached — only the encoder receives this gradient.

**Variable-length batching**: all datasets return a `padding_mask` `(B, T)` bool tensor (`True` = valid). Pass to model alongside `landmarks` or token indices.

**Depth (z) handling** (`LandmarkConformer._fix_depth`): MediaPipe z uses different origins per part (hand: wrist, face: head, pose: hips), but the LMDB subtracts the pose nose from every z — a stored wrist z is ≈ −nose_z (~1.5) plus jitter. In-model fix (no LMDB rebuild): dominance, hand–nose distance and shoulder scale use xy; wrist z → 0 (pos and Δ1); face z re-centred per frame; finger z kept (wrist subtraction recovers MediaPipe's wrist-relative depth). Pose z kept (same frame as nose z).

**Padding safety** (`islr/core/model/conformer.py`): padding is re-masked right before the depthwise conv and `MaskedBatchNorm1d` computes batch/running statistics over valid frames only (same keys as `nn.BatchNorm1d`). `LandmarkConformer.forward` clones its input (mirroring / wrist norm / depth fix are in place).

**Mixup & batching**: `mixup_batch` expects canonicalised clips, resamples each partner to its anchor's length (Δ1 rebuilt) and keeps the anchor mask. `BucketBatchSampler` re-draws batch membership every epoch (shuffle → pools of 50 batches → sort within pool).

**Ablation flags** (defaults = Runs 001–005 behaviour): `--mixup-prob`, `--finger-drop-prob`, `--loss {focal,ce}`, `--zero-parts face,pose`, `--no-depth` (input ablations zero channels inside the model, train and eval; pass the same flags when evaluating a checkpoint; `pose` still feeds the shoulder scale).

**GraphConformer** (`--arch graph`, `--graph-hidden 64`, `--graph-layers 3`): overrides `LandmarkConformer._hand_streams` only. Per frame and hand slot, a shared graph encoder (skeleton + knuckle-web adjacency + learned zero-init residual adjacency, pre-norm residual layers, LayerNorm — no padding statistics, joint embeddings, flattened per joint) over wrist-relative positions, bone vectors and Δ1/Δ2/Δ5; the non-dominant slot is x-negated so both hands share weights with one chirality; the wrist location (position + Δ1/Δ2/Δ5) bypasses the graph. Per-slot `lh/rh_hand_out` → `d_model//2`, split into the parent's pos/vel slots. Everything else inherited. Same LMDB, no rebuild.

**VQ bottleneck** (`islr/core/model/vq.py`, `--vq-codes K`, default 0 = off; `--vq-beta 0.25`, `--vq-decay 0.99`): the fusion input of every valid frame is quantised by four EMA codebooks — dominant hand (pos + Δ-vel + geometry, 160 dims), non-dominant hand (160), body (pose pos + vel + distance stream, 160), face (eyebrow + mouth + face vel, 128) — so each frame reaches `feat_fuse` as 4 codes. Straight-through in training, the code itself in eval (output is exactly a function of the codes); commitment loss β·‖z − sg(q)‖² added in `train_epoch`; codebooks seeded from distinct first-batch frames, dead codes reset every 200 steps, all state in buffers (checkpointed, snapshot/restored by the non-finite guard, frozen in eval and in the contrastive pass). Codebooks start at zeros so a VQ model's other weights initialise identically to the same-seed non-VQ model. Works with both `--arch` values. The epoch line logs commitment loss and per-part code perplexity (codes in use).

**Bone-length augmentation** (`augmentation.bone_scale`, `--bone-aug-prob` default 0 = off, `--bone-aug-hand 0.15`, `--bone-aug-bone 0.1`): per training clip, every hand bone (joint − parent, `config.HAND_PARENT`) is scaled by U(1 ± hand) · U(1 ± bone) and the hand rebuilt outward from the fixed wrist; one draw shared by both hands and all frames (a signer trait). Joint angles are exactly preserved (geometry cosines unchanged), fingertip distances change, the wrist location is untouched. Applied in `ASLDataset` after `hand_presence()`/hand dropout and before noise and Δ1; held frames stay exact repeats (presence round-trips) and never-detected hands stay zero.

**Cross-signer contrastive loss** (`islr/core/model/supcon.py`, `--supcon-weight`, default 0 = off): supervised contrastive term on a training-only `proj_head` over the CLS embedding. Positives = same sign by a *different* known signer; same-sign/same-signer pairs excluded; a FIFO queue (`--supcon-queue`, 4096) of recent embeddings supplies cross-signer candidates without changing batching. Computed on the canonicalised, unmixed clips in a second forward pass (~1.4× epoch time) with BatchNorm running-stat updates frozen (`_frozen_bn_stats`), so enabling it changes only the objective. Non-finite embeddings are dropped and never enqueued; `--supcon-queue 0` = in-batch only. Not a new architecture; inference unchanged.

**Synthetic phrases** (`islr/phrases/dataset.py`, `islr/phrases/train.py`): pseudo-continuous signing for a CTC recogniser, built on the fly from the ASL Signs LMDB (no new data; specs redrawn per epoch, val phrases fixed and built only from held-out signers). One signer per phrase (a person change must not mark a boundary); clips trimmed to first..last hand detection, poorly tracked clips (< `--min-detected` 4 detected frames) swapped; 3–8 transition frames interpolated (hands only when detected at both ends, otherwise held with presence 0); phrase-level hold-fill so a hand missing from a clip is not at the nose. Flags are passed to the model directly; `hand_presence()` re-derived from a stitched phrase matches except when a hand is detected in only one frame of the phrase (a constant hold-filled block cannot say which frame — the same ambiguity as real stored clips; 12/400 val phrases). ~10% of multi-sign gloss sequences (`--holdout-frac`, fixed crc32 hash) are never trained on; validation WER is reported by kind: seen sequence (other signers' clips), unseen template sequence, random sequence. 70% grammar templates (`islr/phrases/grammar.py`, 236/250 signs; written by a non-signer — not reviewed), 30% random sequences (`--random-frac`), 1–5 signs. Optional extra signers (`--extra-data-dir`: WLASL train half, `--extra-frac` of phrases) and a wider label set (`--vocab`). Model = backbone in CTC mode over 250 signs + blank; `--init` loads backbone keys only and exits on any architecture mismatch (`--n-heads` must match, not checkable). Limits: citation-form signs, synthetic transitions, no grammatical non-manuals — phrase metrics need confirming on real continuous signing.

**Browser conventions** (`islr/export/export_onnx.py` → `preprocess.json`; the app lives in the separate ASL-App repo): competition frames are unmirrored (pose 11 x > pose 12 x in 600/600 LMDB clips) and the `left_hand` slot is the signer's own left hand (574/576 hands nearest the same-side pose wrist). In the browser, feed unmirrored frames and assign hands to slots by distance to pose 15/16 — MediaPipe Tasks handedness labels assume a selfie-mirrored image. Training video is portrait ≈ 3:4 (normalised x is a fraction of width, y of height): crop a landscape camera to 3:4 or rescale x and z by (W/H)/0.75; check: shoulder drop below nose / shoulder width ≈ 0.40. Per segment: trim to first..last hand detection, nose-relative coords, hold-fill missing landmarks (nose-relative values), Δ1, presence flags.

**One hand per clip** (data limitation): both hands are detected in the same frame in a median 2 % of ASL Signs clips, even for two-handed signs (shoe 1 %, book / cloud / car 2 %, chocolate 4 %; the raw parquets show it too). Models see a second hand rarely; how much they use it is unmeasured, and a live base hand may push predictions off (anecdotal CHOCOLATE → police). One-hand mode (drop the less active hand) is an experimental browser option for phrases01 only — it hurts phrases03 (trained with WLASL two-hand clips: 76.8 → 57.1 % on WLASL test-half two-hand clips): on the 575 held-out clips with both hands detected somewhere, removing the less active hand raised phrase-model accuracy 44.2 % → 49.0 % (+66 / −38, McNemar p = 0.01), but on the 321 with both hands in the same frame it changed nothing (43.9 % → 44.5 %, +26 / −24, p = 0.89) — the gain is in clips whose hands appear at different times (cause untested). WLASL (same Holistic) tracks both hands. `HandDominanceModule` mirrors with an elementwise `torch.where` (no boolean indexing) so the model exports.

**Spatial affine** (`AdvancedAugmentation.spatial_affine`): rotation × shear × anisotropic scale on x, y of positions and Δ1 (`--aug-rotate 15 --aug-shear 0 --aug-scale 0 --aug-affine-prob 0.5` = previous rotation-only behaviour). No global shift or isotropic scale (nose-relative coordinates / shoulder normalisation make them nuisance or no-ops).

**Training hygiene** (`islr/isolated/train.py`): `--seed` (default 42; seeds Python/NumPy/torch incl. DataLoader workers; CPU runs reproduce exactly, GPU not bitwise). `--train-eval-size` (default 3000) evaluates a fixed un-augmented training subset in eval mode each epoch ("Clean Train") — the loop's train acc is on augmented mixup inputs and not comparable to val. `--loss {focal,ce}` (default focal).

**torch.compile** (`--compile`) — **currently crashes on GPU** (first backward: `(*bias): last dimension must be contiguous` in the CUDA memory-efficient SDPA backward; `nn.MultiheadAttention`'s key-padding mask becomes a broadcast bias that compiled code passes through un-materialised; CPU uses another kernel, so CPU tests pass). Do not use it until attention is excluded from compilation or given a contiguous mask, verified on a GPU. Needs torch ≥ 2.10 on Python 3.14 (2.9.1 raised and `train.py` silently fell back to eager — every earlier `--compile` was a no-op). Pinned `torch>=2.10.0,<2.11`: 2.11+ PyPI wheels are CUDA 13 (driver ≥ 580); 2.10 keeps CUDA 12.8 like 2.9.1. Verified on CPU: compiled = eager to 4e-7, no recompiles across bucket lengths or the per-epoch GRL λ (floats are not specialised); stochastic depth's `.item()` causes one graph break per block. Checkpoints are saved from/loaded into the unwrapped module (`raw_model`), so they never carry the `_orig_mod.` prefix.

**Kaggle GPU runs** (`scripts/kaggle_run.py`): T4, Python 3.13, torch 2.11. Three things the first runs exposed, all fixed: (1) DataLoader workers must not be forked — a forked worker inherits the parent's LMDB handle and py-lmdb refuses to reopen the path, so the dataset fell back to non-existent parquets; `data.dataset.worker_context()` gives every DataLoader `forkserver` (Python 3.14's default, so RunPod was unaffected). (2) `torch.cuda.is_bf16_supported()` is True on a T4 (emulated, slow); the trainers use bf16 only at compute capability ≥ 8, else fp16 + GradScaler. (3) `/kaggle/input` is slow for random reads (~125 ms per stitched phrase, ~1 batch/s from 4 workers); the runner copies the LMDB folders to local disk first (6 GB in 33 s), after which phrase training runs ~6 batches/s on the T4 (A40: ~5).

**AMP training**: `torch.amp.autocast` wraps every forward pass — bf16 on Ampere+ (no `GradScaler` needed), fp16 + `GradScaler` otherwise. Disabled automatically when not on CUDA. `train_epoch` skips non-finite-loss batches and restores BatchNorm running stats from a pre-forward snapshot (a NaN forward would otherwise permanently break eval).

**Preprocessing cache**: `VQVAEDataset` accepts `cache_dir`. First access processes each parquet and saves a `.pt` tensor; subsequent accesses skip parquet parsing entirely. Default: `data/cache/`.

**Pre-tokenization cache**: `precompute_tokens.py` runs the frozen VQ-VAE once, saves per-sample token indices to `data/tokens/`. `TokenizedTranslationDataset` loads these directly — no VQ-VAE needed during Phase 2 training.

**Conformer kernel size**: must be significantly smaller than the average sequence length. VQ-VAE chunk size 8 → 40–80 frame signs produce 5–10 tokens. `encoder_kernel_size=7` fits within the sequence; larger values operate mostly on padding.

**Validation splits** (`islr/core/data/dataset.py:get_data_loaders`): default is a fixed `GroupShuffleSplit(test_size=0.1, random_state=42)` → 3 val signers (2044, 37779, 53618) — kept so runs stay comparable. `--val-fold k` uses unshuffled `GroupKFold(--n-folds, default 7)` by `participant_id` (deterministic; 3 signers ≈ 13–14k samples per fold); training every fold is k-fold CV by signer. Every epoch also prints per-signer val accuracy and the spread — signer variance is large, so pooled 1-point changes are not meaningful on their own.

**Best model selection**: Phase 1 saves checkpoints based on val reconstruction loss (not total loss). The diversity term dominates total loss magnitude and is a poor ranking signal.

**Per-sample augmentation** (`islr/isolated/train.py`): each sample in a batch gets an independent augmentation decision. Vectorized for rotation/finger dropout. All temporal resampling (batch stretch, per-clip stretch, TTA tempo jitter) goes through `AdvancedAugmentation._resample_prefix`: valid frames only, Δ1 rebuilt from positions. Finger dropout collapses a finger onto its wrist (→ exactly 0 after `WristNormalization`); hands never detected in a clip are kept exactly zero through noise/shift. Geometric transforms touch only coordinate channels (`:PRESENCE_START`), never the presence flags. Coordinate noise is added once, in `ASLDataset` (`augment_sample`, before velocity is computed). Time stretch: `--stretch-mode batch` (default, Runs 001–005: one 0.8–1.3× factor per batch, p=0.5) or `sample` (`resample_per_sample`: independent factor per clip, Δ1 recomputed from the resampled positions). Motivated by signer diagnostics: the hardest signers' signs last ~2× longer (median 45–56 frames vs ~22) at normal per-frame speed. Hand dropout (`--hand-drop-prob`, `--hand-drop-min/max`, default off; `augmentation.hand_dropout`, applied in `ASLDataset` to training data only, after `hand_presence()` and before noise/velocity): per detected hand, one span of 10–40% of the clip is frozen at the last detection with presence 0 — exactly how real MediaPipe gaps are stored (99.4% round-trip agreement with `hand_presence`). Motivated by the fraction of no-hand frames ranking per-signer accuracy (Spearman −0.94 over 6 val signers).

**Label mapping**: 250 ASL signs indexed 0–249. Mapping lives in `data/asl-is-lmdb/sign_to_prediction_index_map.json` (downloaded with the LMDB dataset).
