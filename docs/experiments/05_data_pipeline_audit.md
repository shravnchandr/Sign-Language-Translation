# 05 — Data pipeline audit (2026-11-14)

Why: three issues in two days (shared MediaPipe tracker, signer-id fallback, presence
reconstruction) each meant re-running models. Before any more training, every stage from raw
landmarks to model input was checked against data, not only read. GPU work that would have used the
current pipeline was stopped (record runs fold0_graph / phrases03–06, phrases07-v2); the v2
re-extraction continues — its raw parquets keep NaN for undetected parts, so everything below is
rebuildable from them without new video processing.

Samples used: 257 random ASL Signs competition parquets (downloaded from the competition), the 160
PopSign shard-0 clips extracted with fresh and shared trackers (`results/evaluation/tracker_reset/`),
20 ASL Citizen and 4 WLASL clips from earlier extractions, and the local v1 LMDBs. Scripts:
scratchpad `audit_presence.py`, `audit_semantics.py`, `audit_ghost.py` (to be moved into
`islr/core/data/validate.py`, below).

## Findings

| # | Stage | Finding | Evidence | Severity |
|---|---|---|---|---|
| 1 | LMDB format | Detection is lost: `frame_stacked_data` ffill/bfills gaps, `hand_presence` later guesses detection from repeats. A hand detected at frame 0 then missing looks the same as one first detected at frame r and back-filled; the guess is always "first detected at r" | ASL Signs: 56 / 257 clips (22 %) have a wrong flag, 1.05 % of hand-frames, trim boundary differs in 21 %; PopSign: 23 % / 0.24 % / 19 %. Missed = spurious exactly (59 / 59, 38 / 38): every error is that edge shift | **High** — affects training inputs, synthetic validation WER and every evaluation (presence flags and trim) |
| 2 | Extraction | **Ghost hands in PopSign**: both slots filled with the same physical hand (mean joint distance < 0.1 shoulder widths), or a single hand in the wrong slot (at the other pose wrist, far from its own) | PopSign fresh: 105 of 128 two-hand frames are ghosts (16 / 160 clips), 34 wrong-slot frames (7 clips); shared tracker similar (63 / 96). ASL Signs (Google's extraction): 0 ghosts in 27 two-hand frames, 1 wrong-slot frame. ASL Citizen: 1 ghost frame / 20 clips; WLASL: 0 / 184 | **High** for models trained on PopSign (phrases07). The 105 / 128 is from correlated frames of 160 consecutive shard-0 clips — it shows the problem exists, not its dataset-wide rate. Distances in shoulder widths of normalised x, y (not aspect-corrected) |
| 3 | Features | Evaluation and training build features differently: eval computes Δ1 on the full clip then trims (first kept frame has a non-zero Δ1); training trims, then Δ1 (first = 0). Browser fixtures drop no-pose frames (training keeps them with origin 0) and subsample with integer indices (training: float `linspace().long()`) | Code: `cross_dataset._trim` / `sign_reliability` vs `phrases/dataset.py`; `browser_fixtures.segment`. No no-pose frames in any sample (latent) | Medium — affects cross-dataset (`cross_dataset`) and reliability (`sign_reliability`) scores; synthetic validation WER uses the training path and is not affected by this item |
| 4 | Labels | WLASL: 84 source videos are stored under several video ids with bit-identical landmarks; 78 carry different glosses (agree / agreement, wind / breeze, evening→*night* / tonight) | all-gloss LMDB: ~160 of 11,980 training clips = one clip, two labels; 206-sign test half: no conflicts, 4 clips counted twice | Low–medium (label noise touching some of the 250) |
| 5 | Builders | Missing shards / mixed versions / failed clips / non-finite tensors passed silently; WLASL clips without a signer got participant −1 (= train half) | Reproduced; **fixed** (79c1355): shared validated shard loader, manifest, exits on errors, signer ids + splits from shards checked against the json | Fixed |
| 6 | Splits | WLASL halves recomputed from clip counts could move under re-extraction | **Fixed** (b04dab7): frozen file, identical to v1 | Fixed |
| 7 | Extraction scripts | `--ref` pins the clone but the notebook embeds the local `CELL_RUN`; resume reuses an existing parquet without checking its extraction version and blanks width / height / lr_dx | Code (reviewer reproduced both). Not triggered so far: each Kaggle session starts empty | Medium (future runs) |
| 8 | Fingerspelling builder | Error path returned a 5-tuple | **Fixed** (79c1355) | Fixed |

**Status of existing results**: every score so far is a *legacy-pipeline measurement*. The audit
shows the pipeline issues exist; it does not yet quantify how much any score moves when they are
fixed — that is what step 7 (re-scoring unchanged checkpoints) measures.

Checked and sound:
- Hand slot = signer's own side in Google's data (268 / 270) and ASL Citizen / WLASL; no mirrored
  clips in any sample; no frame-index gaps; out-of-frame values are extrapolated lower-body pose only.
- WLASL: no extracted clip needs frame-range trimming (all 11,980 are whole-video instances); no
  source video spans both signer halves; signer ids match the json.
- Training-loop augmentation (spatial affine) touches coordinates only and keeps Δ1 consistent;
  phrase stitching hold-fills / interpolates hands only where flagged present.
- Local v1 LMDBs: every CSV path has data + length keys, sampled tensors finite with the expected
  width (reviewer, 360 tensors).

## Plan: one contract, one rebuild, then retrain once

1. **Contract module** (`islr/core/data/contract.py`), the only code that turns a raw parquet into
   stored arrays, used by every builder and by the browser fixtures:
   drop frames without a pose → assign each detected hand to the slot of the nearest pose wrist
   (15 / 16) and drop the farther of two hands on the same slot (removes ghosts; what the browser
   does already) → nose origin → **explicit presence mask (stored)** → hold-fill from detections.
2. **LMDB v2**: coords + presence mask per clip (new `CACHE_VERSION`); datasets read the stored mask
   (old LMDBs: `hand_presence` fallback with a warning).
3. **One feature function** (`segment → subsample (integer indices) → Δ1`) used by phrase training,
   isolated training, `cross_dataset`, `sign_reliability`, `breakdown`, ONNX verification and fixtures.
4. **Builders**: drop clips whose landmarks are bit-identical to a clip with a different label
   (report them); extraction `--ref` takes cells from that commit; resume checks the version.
5. **Rebuild every LMDB once** from raw landmarks — ASL Signs from the competition parquets, WLASL /
   ASL Citizen / PopSign from the v2 shards — as new Kaggle datasets, so training notebooks stop
   rebuilding them.
6. **Validation gate** (`islr/core/data/validate.py`, run on each LMDB before any training): manifest
   and coverage, finite / shape, stored mask = raw detections on a sample, ghost and wrong-slot rate,
   two-hand rate per source, labels in vocab, signer ids resolved, split disjointness, frozen WLASL
   halves, duplicate-content check.
7. Then, in order: re-score unchanged checkpoints under the new contract (CPU) to measure what the
   contract alone changes; retrain the baseline (phrases05 recipe) and phrases07 on the v2 data —
   one run each; the record re-runs of phrases01–06 only after that, on the final pipeline.

## Status (2026-11-14, after the review of the plan)

Done and verified (commits 0e41084 … c706bd4):
- **Ghost rule calibrated on data and video**, not on wrist proximity: in two-hand frames the
  hands' 2D box IoU is ≤ 0.2 in 97–100 % of frames for ASL Citizen / WLASL / ASL Signs, while
  PopSign's are mostly > 0.5. Six PopSign videos downloaded and overlaid: goose / any / same /
  chocolate show one visible hand drawn in both slots (IoU 0.20–0.99, inter-hand distance up to
  0.40 shoulder widths — so the earlier "< 0.1" distance criterion was too narrow, and
  nearest-wrist assignment too aggressive: both pose wrists sit on the one hand). Rule: IoU ≥ 0.5
  ghost; 0.2–0.5 ghost only within 2 frames of a clear ghost, else kept as AMBIGUOUS (provenance);
  single hands never re-slotted.
- **Contract** (`contract.py`) used by every builder, dataset, evaluator and the fixtures; stored
  presence = canonicalised detections with per-hand provenance (absent / kept / ghost / ambiguous).
  On clean real clips the stored coordinates are bit-identical to the legacy path (ASL Signs
  256/256, PopSign 141/141, ASL Citizen 16/16) and presence equals the raw detections exactly.
  It also exposed one pose-less frame at the start of each WLASL clip (legacy origin 0 → a jump).
- **check_contract**: 9 synthetic cases, expected status, bit-exact features across isolated
  training, phrase training, evaluation and fixtures — passes.
- **Builders**: `--expect-shards` mandatory; duplicates excluded after final label mapping with a
  manifest; extraction notebooks embed the cells of the pinned commit (digest recorded); resume
  only reuses output of the same extraction version.
- **validate.py** gate; end to end on 257 real competition clips: build → validate (200/200
  re-prepared from raw byte-identical) → 1-epoch phrase run → `--frozen-spec` reproduces exactly,
  refuses a changed seed.

Next: when the v2 extraction finishes, rebuild every LMDB once on Kaggle (ASL Signs from the
competition parquets; WLASL / ASL Citizen / PopSign from the v2 shards), validate each with
`--raw-root`, publish as new datasets; re-score the frozen phrases05 / phrases07 checkpoints on them
(CPU); then phrases05 and phrases07 retrained once with a shared `--frozen-spec`. The app must apply
the same contract (preprocess.json now states the ghost rule, segment and integer subsample).
