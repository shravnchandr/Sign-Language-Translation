# Ideas backlog

Pending ideas, by priority. Entries in chronological order; paths are as they were at the time.

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
