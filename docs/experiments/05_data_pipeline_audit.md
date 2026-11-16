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

## Acceptance gate (2026-11-14, review of 33b6971)

The review found seven gaps *between* stages (each component correct, combinable incorrectly). All
fixed; each failure path reproduced and now refused (scratchpad `acc_tests.sh`: 257 real ASL Signs
clips + synthetic PopSign / ASL Citizen / WLASL shards):

| # | Gap | Fix | Reproduction now |
|---|---|---|---|
| 1 | Training ran on data without a validation result; a source with 0 training clips silently lost its 20 % share | `validate.require_accepted`: `phrases.train`, `cross_dataset`, `sign_reliability` need a passing `validation_report.json` made with `--raw-root` whose train.csv and content digests (recomputed over every listed clip) match what they open. `phrases.train` asserts each source's effective sampling share = requested | no report → exit 1; validated without raw → refused; NaN + status 7 in a clip with the old report → content mismatch; revalidated → 3 FAILs, training refused; 0-clip PopSign → "effective sampling share 1.0000 ≠ requested 0.8000"; PopSign 20 % → shares [0.8, 0.2] |
| 2 | Documented ASL Signs notebook wrote legacy LMDBs | It and the WLASL build / extraction notebooks moved to `notebooks/data/legacy/` (README says why). ASL Signs builds through `build_lmdb` (flat `is.lmdb.mdb` release folder) + `validate` in one `kaggle_run.py --bare --competition asl-signs` notebook | 257 clips: built, all checks pass, 200 / 200 re-prepared from raw identical |
| 3 | A video that failed to decode vanished from `shard_<k>.csv`; shard counts could not show it | Extraction writes `selected_<k>.csv` before and `failed_<k>.csv` after; for shards extracted before, `scripts/extraction_inventory.py` re-runs the pinned commit's selection cell (nothing extracted) → `--inventory`. Builders reconcile selection with listings (listed-but-unselected and missing beyond `--allow-missing` exit; shards with neither record exit); the manifest accounts for every clip (selected = listed + missing; listed = filtered + errors + duplicates + written), re-checked by `validate` | 9 selected / 8 listed → exits (recorded or not); `--allow-missing 1` → builds, validate passes and shows the count; inventory with one lost video → exits; no selection record → exits. Inventory dry run on a fake WLASL tree: paths and shard assignment = the extraction's |
| 4 | WLASL / Citizen dropped width / height → validation re-prepared at 0.75 | `prep_aspect` (the aspect the contract used) + width / height per clip in train.csv; validation uses it and fails when absent (ASL Signs: fixed 0.75 in the manifest) | 16:9 ghost tie: prepared at 1.778 keeps the right slot, at 0.75 the left; validate with the stored aspect: 9 / 9 identical |
| 5 | Signer −1 → 899999 (train half); missing Citizen / PopSign ids hashed | Missing, negative, non-integral (and Citizen colliding) ids rejected before the offset | WLASL −1, Citizen NaN, PopSign NaN → build exits |
| 6 | Duplicate `(frame, type, landmark_index)` and negative indices accepted | `contract.check_raw` before any array is built (also unknown part types, missing / non-integral frame ids); 5 refusal cases in `check_contract` | clip with a repeated record → build error → exit |
| 7 | Fixtures rounded raw coordinates to 7 decimals, so near the IoU threshold their provenance differed from their own frames | float32 values serialised exactly; every fixture is rebuilt from its serialised frames and must match bit-exactly (kept frames, hand status, segment, features) or the export exits | 160 PopSign clips (up to 38 ghost frames each) all self-verify |

Also: the reviewer's six all-gloss WLASL clips with no hand at all — `validate` lists clips with no
hand ever present by split / half in the report, `cross_dataset` reports their count and accuracy;
they are kept and scored as-is (the model sees body and face only). `kaggle_run.py --bare` mounts no
legacy dataset and every LMDB placeholder must match exactly one mounted file (two releases cannot
be mixed by search order). Manifests, reports and experiment specs record the code revision
(commit + modified-files flag); `--frozen-spec` refuses a different commit or a run from modified
files (`--allow-diff code` only by explicit choice), so the two comparison runs are pinned to one
revision and, through the release digests, to the same dataset releases.

Not covered (from the review): the full v2 Kaggle artifacts and the live browser client are checked
only when built / run; label linguistics and detector physical correctness are outside any gate.

Release procedure, per dataset: (inventory for pre-fix shards) → build + `validate --raw-root` in one
CPU notebook → the notebook's output is the release, mounted with `--kernel` in `--bare` runs. The v2
raw parquets are kept; no re-extraction is needed.

## Full audit before the rebuild (2026-11-14, round 3)

Goal: find every remaining issue *before* any LMDB is built or GPU run started. Every stage, raw
video → model input → evaluation, checked against data where data exists; full-data checks run as
Kaggle CPU notebooks (`islr.core.data.audit_raw` per dataset, `scripts/extraction_inventory.py`).

### Found and fixed so far

| Finding | Evidence | Fix |
|---|---|---|
| **Signer folds were recomputed, not frozen** — a rebuild that drops clips (exclusions, errors) or another sklearn version could move signers between ASL Signs folds, putting a signer the fold0_graph init trained on into the validation fold | Dropping 3,000 clips of 34503: `GroupKFold(7)` fold 0 becomes 27610 / 49445 / 62590 (27610 = a fold0_graph training signer) | `asl_signs_folds.json` (= GroupKFold(7) on the full 94,477-clip csv; fold 0 = 34503 / 49445 / 62590, as fold0_graph's log); `dataset.signer_fold_split` used by every loader and `sign_reliability`; unknown signers exit |
| Browser contract silent on frame shape and frame rate | `preprocess.json` had no aspect or fps field (only CLAUDE.md did); WLASL 25 fps per its JSON (measured: mostly 29.97, see the WLASL audit), PopSign 30 fps (paper), browser processing often slower | `aspect` and `frame_rate` fields in the preprocess spec; fps of every dataset measured by the inventory notebooks (300 videos each). This documents the requirement — the app's compliance (crop / rescale, processing rate) still has to be verified in the ASL-App repo |

### Checked and sound (no change needed)

- **PopSign ↔ ASL Signs, same recordings?** No matching recording detected in the 160-clip sample
  (3 PopSign signers): two extractions of one video differ by ≤ 0.095 frame-wise (upper-body pose,
  shoulder widths); the closest ASL Signs clip of the same sign and the *same frame count* is ≥ 0.123.
  This only tests untrimmed copies: matching equal-length clips cannot exclude a recording that was
  trimmed or had frames dropped. A trim / frame-drop tolerant version (offset and resampling
  alignment) was calibrated on the same 160 videos and does **not** separate: centred on posture,
  same-video pairs reach 0.83 motion disagreement vs ≥ 0.61 for different recordings — the tracker
  difference between two extractions is as large as the arm motion itself, so pose trajectories
  cannot prove absence. The PopSign paper (NeurIPS 2023 D&B) shares authors with Google / Kaggle,
  thanks the Kaggle team for dataset advice, and records on the same phones (Pixel 4A, 1944×2592,
  30 fps) over the same 250 signs: shared signers are likely and shared recordings possible.
  **Policy:** for any model trained on PopSign, the ASL Signs held-out score and the synthetic
  validation WER (also the checkpoint-selection metric) are not independent; comparisons and
  decisions use ASL Citizen, WLASL and recorded camera clips only. The exact-length test still runs
  on all PopSign clips in `audit-popsign` (it detects untrimmed copies).
- fold0_graph (the `--init` of every phrase run) held out exactly fold 0's signers.
- Phrase generator: held-out gloss sequences never drawn in training for any source; validation
  phrases only from fold-0 signers; tempo resampling keeps presence aligned; never-detected hands
  stay zero through augmentation.
- Evaluation splits: WLASL test half from the frozen file for both the extra training (train half)
  and scoring; ASL Citizen extra data = official train split, scoring = test split; PopSign used
  for training only.

### Known selection effects (documented, not leaks)

- Checkpoints are selected on fold-0 synthetic-phrase WER, and "ASL Signs held-out" scores use the
  same signers → mildly optimistic for that metric; WLASL / ASL Citizen are not selection-biased
  (except phrases04's augmentation, chosen on the WLASL test half).
- `sign_reliability` thresholds are fit and reported on the same clips → optimistic keep rates.

### Full-data audits (Kaggle CPU, queued behind the extraction)

Per dataset (WLASL v2, ASL Citizen v2, PopSign v2, ASL Signs competition): raw structure, frame
gaps, rows per frame, partial parts, off-frame values, pose-less frames, detection rates,
wrong-slot single hands, ghosts, mirroring, extraction version, exact and near-duplicate recordings
(different signer ids / across splits), labels and signers, split disjointness, aspect correction
(per-clip ratio k vs the geometric (W/H)/0.75 the app applies), WLASL instance frame ranges;
PopSign: same-recording test against all ASL Signs clips; inventories: coverage + fps probe.

**WLASL (`islr-audit-wlasl` + `inventory-wlasl`, extraction eb9fea3, 4 shards) — two checks open (below).**

| Check | Result | Action |
|---|---|---|
| Coverage | inventory 11,980 selected = 11,980 extracted, none missing or extra | — |
| Raw structure | 0 read / structure errors, no frame gaps, every frame complete, T = shard n_frames, nothing mirrored | — |
| Pose-less frames | 3,672 clips with 1, 17 with 2–3 (≤ 6 % of a clip) — the known first frame | dropped by the contract |
| Hands | left 44 % / right 63 % of frames; ambiguous 27,204 frames; single hand in the farther slot 637 of 208,675 frames (348 clips, median 1 frame) | single hands are never moved |
| Ghost rejections | a hand rejected in 23,005 frames of **2,242 clips (18.7 %)**; in 19 clips in every kept frame — all signers 10 / 11 at 288×192, all signs where the hands touch (pray ×2, help, aid, assist, thankful, grateful, sandwich, wait, …). Duplicate fits or real touching hands? The statistics cannot tell | **finding, rule to change** — see below |
| No-hand clips | 9, all signer 11 at 288×192 | reported by validate |
| Exact raw duplicates | 1,883 clips in 849 groups; 823 groups carry different glosses — WLASL files one video under synonyms (gay / lesbian, attention / pay attention / concentrate, ill / sick); 4 groups under two signer ids (15 and 59, both in the test half) | builders already drop conflicting groups and keep one copy of a repeat (`excluded_clips.csv`) |
| Near-duplicate recordings | same gloss, equal length only: 68 same-signer pairs, 1 pair under different ids (10 / 11, distance 0.084); 0 pairs across the frozen signer halves | no cross-half duplicate detected by these checks (trimmed or re-encoded copies under other glosses are not covered) |
| Signers | 78 ids; 11 (72 clips) not in `wlasl_signer_split.json` → train; none has a shared-sign gloss, so the 206-sign test half is unchanged | — |
| Official splits | 75 signers in two splits (WLASL's splits are not signer-disjoint) | we use the frozen signer halves only |
| Frame ranges | 2 instances carry a range, none longer than it | — |
| Frame rate | JSON says 25 for every clip; probe of 300 files: 29.97 (45 %), 23.98 (16 %), 30 (13 %), 25 (7 %), ~30.2–30.4 (≈ 15 %), one 50 fps | spec text and builder docstring corrected (mostly 24–30); the column is metadata only (nothing trains on it) |
| Aspect | 8 frame sizes (288×192 … 1920×1080); `ratio_raw / 0.40` vs the geometric (W/H)/0.75: median 1.05 (p10–p90 0.94–1.20). An approximation — the builder's `aspect_factor()` also accounts for shoulder tilt and falls back to 1 outside its bounds | **open**: the comparison (step 2) uses the actual `aspect_factor()` on the same clips |

**Ghost rule on WLASL (`islr-ghost-wlasl`, `-wlasl2`: 66 clips drawn on the video, per-frame evidence for all 11,980).**
In WLASL a "ghost" frame is almost always two *real* hands close together (pose wrists a median 0.36
shoulder widths apart vs 0.86 in ordinary two-hand frames). Holistic then either fits one hand into
both slots or fits two overlapping hands. The IoU rule cannot tell these apart, and fails both ways:

| Case | Evidence (of 23,005 rejected frames) | Example (drawn) |
|---|---|---|
| Near-copy candidates (one hand fitted into both slots?) | slots' mean landmark distance < 0.3 hand sizes in 54 % — candidates, not verified; overlaps the next row | PRAY 44132, HELP 27215 (left palm under the right fist not detected at all) |
| Kept slot on the **wrong side** | kept slot's wrist clearly nearer the other side's pose wrist: 3,077 frames, 535 clips (2,459 of them also near-copy candidates) | BASEBALL 05171 (right fist stored as the left hand), WASH 62358 |
| **Real hand rejected** | skeletons distinct (≥ 0.3) and each slot's wrist on its own side: 6,205 frames | SOUP 53428 (the signing right hand dropped), BEGINNING 05778, DIVE 16891 |
| Ambiguous (IoU 0.2–0.5, both kept) | 94 % have each slot on its own side | AMERICA 02353 (interlocked fingers), ATTRACT 04073 — correct |

The rule was derived from PopSign (100 of 128 two-hand frames were copies). Candidate rule (contract v4):
a ghost only when the two slots are near-copies — corresponding joints compared in aspect-corrected
image coordinates, without aligning the hands (two real hands can share a shape, not a position); the
copy goes to the slot whose pose wrist is confidently nearer; with no confident wrist evidence the
frame stays AMBIGUOUS (no forced assignment); distinct overlapping hands are kept. A hand MediaPipe
never detected stays absent. Threshold chosen on visually labelled frames, checked on separate
clips from WLASL, PopSign and ASL Signs, erring towards keeping real hands.
Before changing the contract: the same evidence on PopSign (`islr-ghost-popsign`) and ASL Signs, so
the threshold is set on all three. A contract change means v4, re-fixtures, and the browser's rule.

**Ghost rule v4 — agreed design (2026-11-14), threshold still to be set.**
Two separate decisions per two-hand frame. (1) *Copy or distinct* — from joint evidence only: the two
slots' corresponding joints compared in aspect-corrected image coordinates, unaligned. (2) *Which slot
keeps a copy* — from wrist evidence only: the slot whose pose wrist is confidently nearer; otherwise a
deterministic tie-break (the current one) and a separate `assignment_uncertain` flag (not AMBIGUOUS,
which keeps both slots present and would retain the duplicate). Confident wrist evidence does not make
a frame a copy. A hand MediaPipe never detected stays absent. Reported per dataset: fallback frequency
among copy decisions and over all frames, and clips affected.
Threshold: per candidate threshold, on visually labelled frames (`ghost_review --sample`; labels copy /
two / unclear, "copy" = both skeletons track the same physical hand however many hands are visible):
real-hand deletions (labelled two, one lost), duplicate retentions (labelled copy, both kept),
wrong-slot assignments where the side can be judged, with counts and unclear coverage; sampling is
balanced over distance bands, so rates are weighted by each band's prevalence before quoting a
dataset rate. The policy is chosen on calibration recordings, frozen, then evaluated once on holdout
(sets split by recording content). Keeping real hands breaks near-ties only. The visual labels are
reference judgements, not ground truth. Sampling covers copy-distance bands up to an open tail (PopSign: 2,200 rejected frames, 11 %, lie above 0.8) and a control band just below the overlap gate (IoU 0.1–0.2) — 8 distance bands + 1 control, 216 frames per dataset. Near-copy joints do not guarantee box overlap (a thin hand shifted sideways: distance 0.09, IoU 0), so whether v4 keeps an overlap gate is decided by measured performance, not geometry; the 8 WLASL frames with IoU < 0.1 and distance < 0.3 (6 clips) and PopSign's 2 with IoU < 0.2 are labelled separately (`ghost_review --frames`). Result (labelled blind, then the key opened): WLASL 7 two / 1 unclear / 0 copy — all palms pressed together seen edge-on (REQUEST, PRAY, BOOKSHELF, TRAFFIC, MOTIVATE), copy distance 0.19–0.30 yet two real hands; the current rule keeps both (correct); a distance-only cut near 0.3 would delete a hand in every one. Both slots' wrists are nearer their own pose wrist in 7 of 8 (#3 only by 0.141 vs 0.146, a near-tie); in #2 (BOOKSHELF, labelled two) the right slot's wrist is nearer pose wrist 15 (0.154 vs 0.182) — so wrist proximity does not separate every case either; it is a feature to test, not a result. An independent labelling (partly blinded: the summary had been seen) agreed on 8 / 10; the two disagreements (WLASL #6, #7: two vs unclear, each way) are why `unclear` stays available. PopSign 2 unclear (JACKET, two fists, one skeleton collapsed; right-slot wrist nearer the left pose wrist, 0.22 vs 0.32). 10 frames from 7 clips: evidence that distance alone is not enough, not a rate. Labels: `results/kaggle/lowiou-*/out/labels_*_claude.csv`. Each slot's projected hand size and box diagonal are logged, so a foreshortened denominator can be told from a loose fit. Rule candidates in order, on calibration only: distance alone; distance with the IoU gate (the baseline — it already keeps the low-IoU palm-to-palm pairs); distance plus the recorded hand-to-pose-wrist distances; a handedness (thumb-side) feature only if those fail. PopSign's rejected frames differ from WLASL's (fewer exact copies: 14 % vs 37 % below 0.1; pose wrists further apart, 0.55 vs 0.36 shoulder widths), so one threshold is not assumed to fit both. ASL Signs (no video) gets a statistical transfer check only —
changed decisions, fallbacks, distance distributions — and its correctness is stated as unverified.

**Ghost rule v4 — calibration result and frozen copy rule (2026-11-14, before the holdout key was opened).**
432 tiles labelled blind (`docs/experiments/ghost_labels/`, committed in 2ccbf0f before any key was
opened). Calibration: WLASL 85 labelled (41 two, 44 copy, 23 unclear), PopSign 103 (0 two, 103 copy, 5
unclear). Errors per rule (counts; w = weighted by each band's share of the dataset's IoU ≥ 0.1 frames):

| Rule (copy decision) | WLASL real hands deleted | WLASL copies kept | PopSign copies kept |
|---|---|---|---|
| current (IoU gate + window) | 12 / 41 (w 0.090) | 2 / 44 (w 0.012) | 18 / 103 (w 0.092) |
| distance < 0.3 | 3 / 41 | 13 / 44 | 56 / 103 |
| distance < 0.3, IoU ≥ 0.2 | 3 / 41 | 13 / 44 | 56 / 103 |
| distance < 0.8 | 18 / 41 | 0 / 44 | 19 / 103 |
| wrist side, margin 0.05 | 5 / 41 (w 0.051) | 5 / 44 (w 0.023) | 6 / 103 (w 0.036) |
| **wrist side, margin 0.1** | **5 / 41 (w 0.051)** | **3 / 44 (w 0.014)** | **4 / 103 (w 0.014)** |
| wrist side, margin 0.15 | 7 / 41 (w 0.102) | 1 / 44 | 2 / 103 |

Findings: joint distance does not detect copies on PopSign (labelled copies in every band, 11 / 12 in the
tail ≥ 0.8); the IoU ≥ 0.2 gate costs PopSign (11 / 11 labelled frames at IoU 0.1–0.2 are copies); on
WLASL the 12 labelled frames at IoU 0.1–0.2 and the 8 below 0.1 are all real pairs. Wrist-side evidence
is the strongest single feature; adding distance does not improve it.

**Frozen copy rule** (chosen on calibration only; margin 0.1 = the largest before deletions rise —
preservation breaks the near-tie): for a frame with both slots detected, the slots are a **copy** iff
(a) IoU of their 2D boxes ≥ 0.1 and (b) NOT both [left-slot wrist nearer pose 15 than pose 16 by more than
0.1] and [right-slot wrist nearer pose 16 than pose 15 by more than 0.1] — distances in aspect-scaled
image units / the clip's median shoulder width. Below IoU 0.1 both slots are kept (unlabelled there:
2,470 WLASL and 60 PopSign frames would otherwise turn into copies, e.g. crossed arms). Scope of the
error estimates: frames with IoU ≥ 0.1. Slot assignment of a copy is a separate decision, not evaluated
by these labels (they do not record the side); its fallback frequency is reported, and the side needs
its own labels before it is frozen. **Holdout (frozen rule, evaluated once):** WLASL 89 labelled (44 two, 45 copy, 19 unclear), PopSign 101
(101 copy, 7 unclear).

| Rule | WLASL real hands deleted | WLASL copies kept | PopSign copies kept | errors |
|---|---|---|---|---|
| current | 21 / 44 (w 0.201) | 0 / 45 | 19 / 101 (w 0.145) | 40 |
| frozen v4 | 14 / 44 (w 0.094) | 2 / 45 (w 0.014) | 5 / 101 (w 0.024) | 21 |

v4 halves the errors on both datasets, but holdout deletions (14 / 44) are well above calibration (5 / 41):
the calibration estimate was optimistic. In 12 of the 14 (and 4 of the 5 on calibration) one slot is
clearly on its own side and the other on the wrong side — hands stacked or crossed, one wrist resting by
the other pose wrist (pose wrists 0.16–0.72 apart). A copy has the same signature (both slots by one
pose wrist), so wrist side alone cannot separate stacked real pairs from copies. Not retuned on holdout.
A further feature (e.g. whether the *other* pose wrist is also at the hands) would need a fresh
calibration set and a fresh holdout, from recordings not used here.

**ASL Citizen (`islr-audit-aslcitizen` + `inventory-aslcitizen`, extraction 07f23b7, 3 shards; research use
only, data stays on Kaggle).** Coverage complete (8,557 selected = 8,557 extracted); 0 structure errors,
no frame gaps, nothing mirrored, 0 exact duplicates; 135 clips with pose-less frames, 7 no-hand clips;
frame sizes 640×480 (8,225), 960×540 (330), 480×640 (2); ~30 fps (probe). Upper-body pose coordinates outside
the expanded bounds [−0.5, 1.5]: mean per-clip fraction 6.2 % (WLASL 0.05 %) — not a measured wrist error
or arm-cropping rate; pose wrists 15 / 16 are to be checked separately, since wrist evidence drives the
ghost rule. Signers: 51, official splits disjoint (35 / 5 / 11). The 280 `seed…` clips are all P52
(the reference signer, one per gloss, train). Near-duplicate test: 8 same-gloss pairs ≤ 0.095, 5 under
different ids — val–train (P39 / P52-seed BOY, P26 / P31 CARROT), test–val (P49 / P21 LIPS), train–train;
**no test–train pair detected by these checks** (the split our evaluations depend on). As on ASL Signs,
the pose-track test probably does not separate similar movements of different people, so this is a
screen, not proof. Ghost rule fires in 1,232 clips (7,755 frames): v4 applies here too.

**Ghost rule v4 — decision round 2 (planned 2026-11-14; criteria fixed before sampling).**
One bounded round; the frozen rule above is the baseline (B0). Fresh recordings only: every recording
content already seen (main samples, low-IoU frames, contact sheets — `ghost_labels/used_content.csv`,
262 WLASL / 199 PopSign) is excluded; new calibration and holdout sets split by content. Sampling:
frames with both slots and IoU ≥ 0.1, 18 per wrist stratum per set per dataset, each recording used once
(scarcest stratum filled first, refilled from unused recordings; a first push that deduplicated after
filling — 106 / 107 tiles — was discarded unseen) — `both_own` (B0 says
two), `stacked` (not both own, both pose wrists ≤ 0.3 from a hand: stacked / crossed real pairs *and*
PRAY / HELP-style copies), `one_far` (a pose wrist away from the hands). Labels: copy / two / unclear and,
for a copy, the side of the physical hand (left / right / unclear, judged from the arm in the video, not
the pose-wrist dots). The other pose wrist being near is supporting evidence, not proof of two hands: a
candidate must improve the measured trade-off, not just keep both in the `stacked` stratum.

Procedure: candidates and the slot-assignment rule are chosen on the new calibration only, committed,
then evaluated once on the new holdout. Rates are weighted by each stratum's share of the dataset's
IoU ≥ 0.1 frames. Reported separately per dataset: real-hand deletions, copy retentions, wrong-side
assignments (copies with a side label), assignment-uncertain share, unclear share.

Adoption criteria (holdout): a candidate C replaces B0 iff (1) its weighted errors (deletions +
retentions) summed over WLASL and PopSign are lower than B0's, and (2) on neither dataset is its weighted
deletion rate or retention rate worse than B0's by more than 0.02. Otherwise B0 stays. The assignment
rule is adopted iff wrong confident assignments / confident assignments whose copy has a known reference
side (left / right label) ≤ 5 % on holdout — reported with the uncertain share (assignment fallbacks)
and the fallback tie-break's own errors on side-labelled copies; otherwise every copy is assigned by the fixed tie-break and flagged `assignment_uncertain`.
Whatever passes is v4; remaining errors are documented, with no further holdout-driven revision.

**Round 2 — calibration result and frozen decisions (before the round-2 holdout key was opened).**
Labels (`ghost_labels/round2_*`, committed in 1ff49b1 before any key): WLASL calibration 24 two / 26 copy /
4 unclear, PopSign 1 / 51 / 2. By stratum: WLASL `both_own` 18 two, `stacked` 11 copy / 4 two, `one_far`
15 copy / 2 two; PopSign `stacked` 18 / 18 copies, `one_far` 18 / 18, and `both_own` 15 copies of 16.
Weights: stratum shares of the IoU ≥ 0.1 frames (WLASL 65 / 22 / 13 %, PopSign 4 / 13 / 83 %).

| Copy rule (calibration) | WLASL deleted | WLASL kept copies | PopSign kept copies | weighted sum |
|---|---|---|---|---|
| **B0 (frozen)** | 6 / 24 (w 0.074) | 0 / 26 | 15 / 51 (w 0.037) | **0.111** |
| keep both in `stacked` | 2 / 24 | 11 / 26 (w 0.162) | 33 / 51 (w 0.167) | 0.344 |
| `stacked`: copy iff distance < 0.6 | 4 / 24 | 4 / 26 | 22 / 51 | 0.191 |
| `stacked`: copy iff hand wrists < 0.25 apart | 5 / 24 | 2 / 26 | 18 / 51 | 0.148 |

The other pose wrist being near is not evidence of two hands on PopSign (all 18 labelled `stacked` frames
are copies); every candidate that treats `stacked` differently raises PopSign retention more than it
lowers WLASL deletion. **B0 stays the copy rule** (no candidate beats it on calibration; holdout is
evaluated for the record). PopSign `both_own` copies (15 / 16, 4 % of its frames) are B0's known residual.

**Assignment (frozen):** with D_left = (|LH wrist − pose 15| + |RH wrist − pose 15|) / 2 and D_right =
(|LH wrist − pose 16| + |RH wrist − pose 16|) / 2 (the mean of the two slot wrists' *distances*, not the
distance from their midpoint, which gives PopSign holdout 1 / 24 instead of the evaluated 1 / 19), a copy
goes to the slot of the smaller D; confident iff min(D) / max(D) < 0.45 (the largest ratio with ≤ 5 % wrong
confident assignments on PopSign calibration: 1 / 25; WLASL 0 / 15), otherwise the same nearer side is
used and the frame flagged `assignment_uncertain` (calibration: 47 % of PopSign and 31 % of WLASL copies;
fallback wrong 7 / 20 and 0 / 4 among side-labelled). This replaces the clip-level "alone more often"
tie-break, which these labels cannot evaluate. Side labels follow the arm to its shoulder in the image
(image-left shoulder = `right`, the pose model's frame); 7 / 26 WLASL and 6 / 51 PopSign calibration
copies have no side label (unclear).

**Round 2 — holdout (evaluated once) and decision.** WLASL 23 two / 24 copy / 7 unclear; PopSign 0 / 53 / 1.

| Copy rule | WLASL deleted | WLASL kept copies | PopSign kept copies | weighted sum |
|---|---|---|---|---|
| **B0** | 7 / 23 (w 0.087) | 2 / 24 (w 0.072) | 18 / 53 (w 0.040) | **0.199** |
| `stacked`: copy iff hand wrists < 0.25 apart | 6 / 23 | 2 / 24 | 21 / 53 | 0.206 |
| keep both in `stacked` | 3 / 23 | 13 / 24 | 36 / 53 | 0.432 |

No candidate beats B0 → **B0 is the v4 copy rule** (criterion 1 fails for every candidate). Assignment:
wrong confident assignments 0 / 10 (WLASL) and 1 / 19 (PopSign) = 1 / 29 (3.4 %) of confident assignments
with a known side → **adopted** under the criterion as written (pooled); PopSign alone is 5.3 % (one
frame, n = 19 — the criterion did not specify per-dataset, noted rather than reinterpreted).
Uncertain share — two denominators, kept apart: among *reference-labelled copies* (including copies
the rule kept) 7 / 24 WLASL and 27 / 53 PopSign (fallback wrong 1 / 4 and 2 / 17 among side-labelled);
among the *rule's copy decisions* in the holdout sample 9 / 36 for each dataset (stratified-sample
counts, not population rates). Copies with no side label: 10 / 24 and 17 / 53.

**v4 (frozen):** per frame with both slots detected — copy iff IoU ≥ 0.1 and not both slots on their own
side by > 0.1 shoulder widths; a copy is kept in the slot of the smaller mean wrist-to-pose-wrist distance D (above; confident iff
min(D) / max(D) < 0.45, else flagged `assignment_uncertain`), the other slot rejected; otherwise both kept.
Single hands are never moved; a hand never detected stays absent. Documented residuals: WLASL real hands
lost where hands stack or cross (holdout 7 / 23 labelled pairs, ~9 % of its IoU ≥ 0.1 frames weighted);
PopSign copies kept where the copy sits between the wrists (`both_own`, 18 / 18 on holdout, ~4 % of its
frames). Implementation checklist: `contract.py` (rule, statuses, `assignment_uncertain` per frame); carried
through LMDB serialisation (`dump_clip` / `load_record`), the content digest, `validate.py` schema checks
and the `--raw-root` re-prepare comparison, `browser_fixtures` (written and verified), `check_contract`
cases (copy / two / boundary: IoU exactly 0.1, margin exactly 0.1, ratio exactly 0.45; ties);
`CONTRACT_VERSION`, `CACHE_VERSION` and `VALIDATION_POLICY` bumped; the preprocess spec *and* the
ASL-App preprocessing updated, with Python / browser parity checked on the fixtures; then the rebuild.
**Done in d9741ac** except the ASL-App port (separate repo, the user's): contract = evaluated formula on
663 real two-hand frames × 3 aspects (0 differ); every threshold tested 1e-6 either side and the exact
tie; an end-to-end build validates with copies; a flipped `assignment_uncertain` fails schema, digest
and re-prepare. `check_contract --write-fixtures DIR` writes the cases as browser fixtures plus
`rule_vectors.json` (threshold / tie inputs and expected outputs) for the app's parity test.

**Aspect: same clips, builder vs browser preparation (`islr-aspect-wlasl`, `aspect_compare`, 824
held-out-signer WLASL clips, 0 errors).** k builder / browser median 1.08 (p10–p90 0.99–1.23).

| | ASL Signs (reference) | builder (per clip) | browser ((W/H)/0.75) |
|---|---|---|---|
| shoulder drop / width, p1–p50–p99 | 0.27–0.40–0.50 | 0.40 (fixed by construction) | 0.35–0.43–0.54 |
| clips outside ASL Signs p1–p99: ratio / hand size | 2 % / 2 % | 0 % / 8.7 % | 7.5 % / 7.6 % |
| joint-angle values outside ASL Signs p1–p99 | 2 % | 1.5 % | 1.2 % (change vs builder: median 0.006, spread between ASL Signs clips 0.19) |
| phrases01 single-sign exact (trained on ASL Signs only) | | 43.3 % | **46.0 %** (+40 / −18, p = 0.005) |
| phrases05 (trained on builder-corrected WLASL) | | 83.0 % | 83.0 % (+4 / −4) |

The browser preparation stays inside what training covers for hand size and joint angles; its shoulder
ratio sits a little higher (median 0.43 vs 0.40) and outside the reference in 7.5 % of clips, concentrated
in some frame sizes (320×240 23 %, 656×370 25 %, 854×480 27 %, 640×360 4 of 9; 1280×720 and 720×400 0 %)
and signers (98, 93, 8) — possibly sources whose stored frame size is not their display shape
(letterboxed / anamorphic), unverified. On these clips the browser preparation does not score worse: the
neutral model does better with it, phrases05 the same. **No builder change is required by this WLASL comparison** (it does not establish equivalence for other datasets or live cameras; the frame sizes are those of the decoded frames, so the source / display mismatch is a hypothesis to inspect).
Open question for the rebuild (not decided): the builders could use the geometric factor too (W / H is
now recorded per clip), which matches deployment and keeps natural body-proportion variation — but the
per-clip factor is robust to a mis-stated frame size, the geometric one is not.

**Aspect, ASL Citizen (`islr-aspect-aslcitizen` at 8ba0b35, official test split, 2,085 clips, 4 without a
hand).** k builder / browser median 1.01 (p10–p90 0.88–1.16). Shoulder drop / width: browser p1–p50–p99
0.31–0.41–0.50 (ASL Signs 0.27–0.40–0.50; 0.7 % of clips outside its p1–p99), builder 0.40 by
construction; hand size outside the reference 1.5 % (builder) / 1.4 % (browser); joint-angle values
outside 1.1 % / 1.0 % (ASL Signs itself 2 %). Single-sign exact decode, builder → browser: phrases01
59.8 → 59.9 % (+54 / −52, p = 0.92), phrases05 78.8 → 78.7 % (+8 / −9, p = 1.0) — **no detectable
accuracy difference** (not a proof of equivalence). Both preparations stay inside what training covers;
the builders are kept as they are.

**ASL Signs (`islr-audit-aslsigns`, 94,477 clips).** Raw structure clean (0 errors, all frames
complete, no pose-less frames, no no-hand clip); 43 clips with frame-number gaps (the audit's counter counted
positions shifted after a gap, not missing frames — fixed to `frame_gaps` + `missing_frames`; the
missing count comes with the next audit run); 4 clips flagged by orientation (pose 11 x ≤ 12 x; median lr_dx −0.003 to −0.70 — to inspect
before calling them mirrored); 0 exact duplicates; ghost rule fires in 439 clips
(1,667 frames). The near-duplicate test flags 25,619 cross-signer pairs (all ≥ 6 frames, 4,901 ≥ 15)
— the upper-body pose track does not separate people in these short, static clips, so on ASL Signs
it **proves nothing**; with 0 exact duplicates (one extraction pipeline) a true cross-id copy is
unlikely. Face-mesh shape can triage the flagged pairs, but only after its separation is measured on
known same-recording and different-recording controls; similarity alone does not establish a copy.

**PopSign (`islr-audit-popsign` at dd3d977, 26,278 clips; `inventory-popsign` at 07f23b7).** Coverage:
26,278 selected = 26,278 extracted (2,190 / 2,189 per shard). Raw structure clean: 0 read / structure
errors, 0 frame gaps, 0 exact duplicates, 0 mirrored clips; 295 clips with some pose-less frames (dropped
by the contract), 22 no-hand clips; single hands in the other side's slot 2,862 of 1.33 M hand-frames
(0.2 %). 250 glosses, 31 signers, all `train`. Its copy counts are v3 (pushed before v4; v4 re-run
below). Three findings:

1. **Frame rate.** The probe (300 videos) has 56 at 120 fps (+ 1 at 60): 120 fps clips hold ≈ 233
   frames for a ≈ 1.9 s sign vs ≈ 63 at 30 fps — about a fifth of PopSign was 4× slow motion against
   the ≈ 30 fps training and browser rate (phrases07 trained on it). Per clip, not per signer (signers
   with 0–58 % clips over 150 frames), so no signer-level fix. ASL Citizen (17–31 fps, 267 / 300 at 30)
   and WLASL (24–30, one 50) have no such mode. **Decision (user, 2026-11-15): decimate.** Contract
   step 0: frame_step k = round(fps / 30) (120 → 4, 60 / 50 → 2, ≤ 44 → 1), positions 0, k, 2k, … of the
   raw frames, recorded per clip (train.csv `fps` = the probed video rate, `frame_step`); validate
   (policy 6) requires it for every clip and the re-prepare uses it. Fps of every PopSign video:
   `extraction_inventory.py push popsign --probe-all --name fps-popsign`; the builder requires it
   (`--fps-csv`) and exits on any clip without one. The tracker ran at 120 fps, so a decimated clip is
   close to, not identical with, a 30 fps recording. Tests: `check_contract` (step → every k-th frame
   bit-exact, bad steps refused); end to end, 120 / 119.88 fps clips 30 → 8 frames, 60 → 15, policy 6
   re-prepare = stored, an edited `frame_step` fails train.csv digest and re-prepare, a clip without
   fps stops the build.
   Review of f31f837 (policy 7): validate now requires frame_step = frame_step_for(probed fps) for
   PopSign (1 elsewhere) and no PopSign clip from the frozen match list; the infinity check runs on
   every raw frame before step 0. Consistent-but-wrong releases (fps 120 / step 1 with digests
   rewritten; a frozen path built in with an empty --exclude-csv) fail only the new checks; an infinity
   in a frame step 4 discards is refused. The fps probe reads headers only: `video_timing` decodes
   samples of each class (timestamps + duplicate-frame share).
2. **Same recordings as ASL Signs.** 257 of the 13,271 clips with an ASL Signs clip of the same sign
   and frame count lie within the same-video distance (≤ 0.095; 1 % quantile of best distances 0.077,
   median 0.190), from 21 PopSign signers, matching 17 ASL Signs signers — including fold-0
   validation signers 49445 (48) and 62590 (2) and fold-0's 53618 / 2044 elsewhere. Same recording or
   the same person in the same posture: either way not independent. **Decision (user): exclude.**
   `core/data/popsign_asl_signs_matches.csv` (the 257, with their ASL Signs match and distance) is the
   builder's default `--exclude-csv`: left out before preparation, listed in `excluded_listed.csv`,
   counted as `filtered`, the list's digest in the manifest. Copies trimmed or with dropped frames stay
   undetectable, so the policy above (PopSign-trained models: ASL Signs scores not independent) stands.
3. **9:16 clips.** 1,332 clips (5 %, signers 4a.8902 all, gtsignstudy4a.8040 348 of 810) are
   1080×1920, the rest 1944×2592 (3:4). The builder applies no aspect correction; their raw shoulder
   ratio is lower (8040: 0.355 vs 0.416 at 3:4 — a factor 1.17, not the 1.33 a pure width change would
   give, so not simply a different crop; unverified without the videos), inside ASL Signs' p1–p99
   (0.27–0.50). Recorded (width / height, prep_aspect per clip); no change.

**PopSign frame timing (`islr-timing-popsign`, cbce89a; full probe `fps-popsign`).** Every one of
the 26,278 videos probed: 6,476 at 120 fps (24.6 %), 22 at 60, the rest 29–31; the extracted landmark
frame count equals the header frame count for every clip. Decoded (30 videos per class; per-frame
timestamps in `video_timing_frames.csv`, 0 failures): timestamps are regular (120: 8.333 ms, 60: 16.667,
30: 33.4; ffprobe packet times identical) and spans alike (≈ 2.0–2.3 s), but **75 % of consecutive
frames at 120 fps and 54 % at 60 fps are near duplicates** (30 fps: 1.5 %). The duplicates come in runs
of 4 (120) / 2 (60) in every sampled video, and the number of distinct frames (median 61.5) equals a
30 fps clip's (62): these are ~30 fps recordings stored in a 120 / 60 fps container, each frame repeated.
Decimation (step 4 / 2) takes ≈ 98 % of the distinct frames exactly once (29 / 30 videos in phase;
10 runs missed and 43 taken twice of 1,890 — the 30 fps class shows the same rate of "twice", from held
poses merging runs), so the kept frames are the real 30 fps images, not a resampling of 120 fps motion.
Until now, about a quarter of PopSign clips were 4× slow in frames.

**ASL Signs copy-rule transfer (statistical; `islr-evidence-aslsigns`, `copy_evidence`).** ASL Signs has
no videos, so the 1,856 v4 copy decisions (544 clips, median 2 per clip; 1,940 two-slot frames in those
clips) were compared with the visually labelled frames of IoU ≥ 0.1. Slot landmark distance / hand size,
median (p10–p90): ASL Signs decisions 0.24 (0.04–0.78); labelled copies WLASL 0.12 (0.00–0.58), PopSign
0.41 (0.05–1.16); labelled real pairs WLASL 0.96 (0.30–1.58). Share below 0.3: decisions 59 %, labelled
copies 50 %, labelled real pairs 10.5 %. IoU median: decisions 0.69, copies 0.73, real pairs 0.26.
Consistent with the decisions being copies (a mixture of the labelled distributions does not fit
anything but nearly all copies); not a proof. The rule fires in 17 % of ASL Signs' 10,974 two-slot
frames but 0.05 % of all frames. (An earlier sample, 0 ghosts in 27 two-hand frames of 257 clips, was
too small to show this rate.)

**ASL Signs repeated-frame check — criteria, committed before the results (`repeat_frames`, bf01452).**
Why: 59 of the 257 excluded PopSign clips are 120 fps (= 30 fps frames repeated 4×) and match 59
distinct ASL Signs clips of the same frame count, so ASL Signs may contain repeated-frame recordings,
stored undecimated. ASL Signs has no videos; clip length or signer alone are not evidence, and
MediaPipe's temporal smoothing means repeated images do not give equal coordinates. Measure: per clip,
the phase share s_k of upper-body pose motion (share of the motion on the strongest of k phases; 1/k
spread evenly, → 1 when all motion falls on one phase), k = 2, 4, and its 24-frame-window median.
1. Calibration / evaluation on PopSign, split by signer (crc32(user) % 2), labels from the probed fps:
   120 → repeated ×4, 60 → ×2, 29–31 → ordinary. Threshold on s_4 chosen on the calibration half
   (balanced accuracy); reported on the evaluation half, overall and on held-pose controls (ordinary
   clips in the lowest quartile of mean motion).
2. **Separates reliably** iff on the evaluation half: recall on repeated ×4 ≥ 98 %, false positives on
   ordinary ≤ 0.5 %, and ≤ 1 % on the held-pose controls.
3. Only then ASL Signs: first the 59 candidates (ASL side of a 120 fps PopSign match) against the 198
   matches of ordinary PopSign clips (negative controls), then the whole dataset.
4. **Convincing** iff (2) holds and, on ASL Signs, the candidates are flagged at ≥ 70 % and the
   controls at ≤ 2 %, with the difference significant (Fisher exact p < 0.001). Then the affected clips
   are corrected before training (a frame step for ASL Signs clips, with the phase taken from the
   motion — a separate, reviewed change). Otherwise the accepted release stays and this is documented
   as a limitation.
5. Definitions (review of 9a59d20, committed before the results): the decision uses the **global s_4**
   only (threshold from step 1); the window median is diagnostic. **Unscorable** clips (s_4 NaN: fewer
   than 8 motion steps or no motion) are never flagged: they count as misses among repeated ×4 clips
   and as not flagged among ordinary clips and controls (every denominator includes them), their number
   is reported per group, and unscorable ASL Signs clips stay unchanged. The 60 fps (×2) clips are
   reported separately, outside the ×4-vs-ordinary evaluation. The 59 candidates and 198 controls are
   **proxy labels** (cross-dataset matches, not confirmed recordings): passing (4) supports transfer of
   the detector to ASL Signs, it does not establish each ASL Signs decision.
6. Counts (review of a25b6a0): rates are compared unrounded — candidates need ≥ 42 of 59 flagged. The
   198 control matches are **192 distinct ASL Signs clips** (6 are matched by two PopSign clips; no clip
   is in both groups), so controls pass with ≤ 3 of 192 flagged (4 / 192 = 2.08 % fails). The threshold
   is frozen by `repeat_check calibrate` (calibration signers only) into
   `core/data/repeat_frames_threshold.json`, committed before `repeat_check evaluate` runs; `evaluate`
   never refits and refuses a different calibration signer set.

**Result (repeat-frames at bf01452; `repeat_check` at 8fded24; threshold frozen in 00df7a2): gate 2 fails —
the detector does not separate, ASL Signs was not examined, the accepted ASL Signs release stays.**
Coverage complete (26,278 PopSign, 94,477 ASL Signs clips; 0 read errors; 15,676 ASL Signs clips
unscorable, never looked at further). Calibration signers (13): best threshold s_4 ≥ 0.337, recall ×4
83.9 % (3,032 clips), false positives 9.5 % (8,191 ordinary). Evaluation signers (18), frozen threshold:
recall 94.2 % (3,243 / 3,444; needed ≥ 98 %), false positives 9.9 % (1,144 / 11,589; needed ≤ 0.5 %),
held-pose 9.4 % (273 / 2,898; needed ≤ 1 %). ×2 (outside the decision): 18 of 22 flagged, s_2 median 0.69.
Why (PopSign only): s_4 median 0.44 on ×4 clips (p5–p95 0.32–0.57) against 0.29 on ordinary ones
(0.26–0.35) — exact repeats would give ≈ 1. MediaPipe's temporal smoothing spreads the step onto each
new image over the repeats, so landmarks move on nearly every frame and the period-4 signature is weak
and overlapping. **Limitation (documented, not corrected):** ASL Signs may contain recordings of ~30 fps
frames repeated 4× stored undecimated (indicated by 59 high-fps PopSign matches, proxy labels only);
their number is unknown. A different detector would need a new, separately committed round.

**v4 statistics per dataset** (raw audits at d19ba7c vs v3 at dd3d977, per clip; frame-level v3
decisions are not stored, so "frames changed" is the per-clip count difference, a lower bound).

| | WLASL | ASL Citizen | PopSign | ASL Signs |
|---|---|---|---|---|
| clips / kept frames | 11,980 / 821,099 | 8,557 / 684,002 | 26,278 / 3.03 M | 94,477 / 3.58 M |
| two-slot frames | 305,138 | 98,856 | 36,050 | 10,974 |
| v4 copy frames (share of two-slot) | 20,275 (6.6 %) | 7,031 (7.1 %) | 24,312 (67 %) | 1,856 (17 %) |
| clips with copies, v3 → v4 | 2,242 → 2,346 | 1,232 → 1,298 | 3,472 → 3,787 | 439 → 544 |
| assignment_uncertain: frames (of copies / of all frames) | 6,161 (30 % / 0.75 %) | 1,890 (27 % / 0.28 %) | 4,304 (18 % / 0.14 %) | 464 (25 % / 0.013 %) |
| clips with uncertain frames | 1,191 | 591 | 1,280 | 180 |
| clips whose copy count changed (v3 only / v4 only) | 1,680 (224 / 328) | 680 (87 / 153) | 1,137 (40 / 355) | 187 (8 / 113) |
| frames changed, lower bound | 6,628 | 1,738 | 2,101 | 265 |

PopSign's v4 audit ran at frame step 1 (it predates step 0), so its frame counts are pre-decimation.

**Releases (contract v4, built at 8ba0b35 with no local changes, re-validated under policy 7 at 0b4a608
by copying the release unchanged — content digests identical).**

| Release (Kaggle notebook) | clips | coverage | notes |
|---|---|---|---|
| ASL Signs (`accepted-aslsigns`) | 94,477 | 94,477 listed = written | 1,856 copy frames in 544 clips |
| WLASL 206 (`accepted-wlasl`) | 1,605 | 11,980 selected = 10,309 filtered (outside the 206) + 66 duplicates + 1,605 | = the old release minus exactly the 66 duplicates |
| WLASL all glosses (`accepted-wlasl-all`) | 10,162 | 11,980 = 1,818 duplicates (1,753 same content under different labels, 65 repeats) + 10,162 | the old release kept the 1,753; labels with clips 1,973 → 1,960 |
| ASL Citizen (`accepted-aslcitizen`, research only) | 8,557 | complete | |
| PopSign (`accepted-popsign`) | 26,021 | 26,278 = 257 ASL Signs matches excluded + 26,021 | frame steps 1: 19,582, 4: 6,417, 2: 22 |

Each: 200 clips re-prepared from raw = stored; canonical label map (WLASL all: canonical 250 + extras).

**Re-score of the existing checkpoints on the accepted releases (`islr-rescore-product` /
`islr-rescore-research` at 2de95c1, CPU; single-sign exact decode).** The first scores of these
checkpoints under contract v4 data (WLASL test half 808 clips / 34 signers; ASL Citizen official test
split 3,408 clips). The checkpoints themselves were trained on the legacy pipeline (phrases07: on
undecimated PopSign), so these are baselines for the frozen comparison, not new models.

| checkpoint | WLASL test exact / top-5 | ASL Citizen test exact / top-5 | ASL Citizen identical / variant / synonym |
|---|---|---|---|
| phrases01 | 42.9 % / 74.5 % | 52.8 % / 80.4 % | 59.7 / 38.3 / 56.9 % |
| phrases03 | 79.3 / 94.3 | 67.7 / 88.7 | 75.5 / 53.6 / 62.6 |
| phrases04 | 80.7 / 95.8 | 67.0 / 88.8 | 74.8 / 53.3 / 61.0 |
| phrases05 (browser model) | 82.7 / 95.9 | 71.4 / 89.8 | 78.6 / 58.6 / 66.3 |
| phrases05-long | 82.4 / 96.0 | 70.8 / 90.1 | 77.4 / 59.6 / 64.2 |
| phrases07 | 84.3 / 96.2 | 73.5 / 90.1 | 80.7 / 61.0 / 66.7 |
| phrases06 (research only; trained on ASL Citizen train) | 85.9 / 96.8 | 86.8 / 97.4 | 89.7 / 86.4 / 63.4 |
| phrases06-long (research only) | 87.5 / 97.3 | 88.0 / 97.3 | 90.6 / 87.9 / 66.3 |

Paired, phrases05 → phrases07: WLASL 82.7 → 84.3 % (+35 / −22, McNemar p = 0.11); ASL Citizen 71.4 →
73.5 % (+197 / −127, p = 1.2e-4) — the same direction as the legacy measurement (70.9 → 73.4 %).
phrases06's ASL Citizen score is in-domain (trained on other ASL Citizen signers recorded the same way; the official splits are signer-disjoint), so it is not comparable with the others there.

**Frozen comparison (`islr-frozen-<run>` at 2de95c1, Kaggle T4, `--frozen-spec` = the reference spec,
`--allow-diff spec_only`).** Each run's own `experiment_spec.json` is checked with
`check_freeze --commit 2de95c1 --profile <run>`.

| run | check_freeze | best val WER | WLASL test exact / top-5 | ASL Citizen test exact / top-5 |
|---|---|---|---|---|
| phrases05 (retrained on the accepted releases) | 15 / 15 | 0.2826 (epoch 27 / 30) | 82.5 / 95.4 % | 70.4 / 88.6 % |
| phrases07 (+ PopSign, 20 % of phrases) | 16 / 16 | 0.2326 (epoch 29 / 30) | 84.0 / 95.7 % | 73.2 / 89.9 % |
| phrases06 (research only; + ASL Citizen train) | 17 / 17 | 0.2857 (epoch 23; early stop at 29, patience 6) | 85.0 / 97.8 % | 87.1 / 97.2 % (in-domain) |

Paired against the re-scored legacy checkpoints (same clips): phrases05 legacy → frozen WLASL 82.7 →
82.5 % (+28 / −29, p = 1), ASL Citizen 71.4 → 70.4 % (+149 / −183, p = 0.07). Broadly similar, with no
statistically detectable difference — not shown equivalent (the ASL Citizen point estimate is 1.0 point
lower). The frozen phrases05 is a valid baseline because its configuration and evaluation protocol pass
the freeze checks, not because of this agreement. (legacy phrases07 → frozen phrases05: ASL Citizen 73.5 → 70.4 %, +116 / −220, p = 1.5e-8;
this compares models, not data, and is not the planned test, which is frozen 05 vs frozen 07.)

**Frozen phrases05 → frozen phrases07 (the planned test; same code, releases, init, seed, validation
phrases; only the PopSign source added).** ASL Citizen test 70.4 → 73.2 % (+207 / −114, McNemar p =
2.3e-7), higher for 9 of the 11 test signers (the two lower: −0.7 points each); WLASL test half 82.5 →
84.0 % (+39 / −27, p = 0.18) — no statistically detectable difference there. Against the legacy
phrases07 checkpoint, frozen phrases07 shows no detectable difference on either test (WLASL +24 / −26,
p = 0.89; ASL Citizen +129 / −140, p = 0.54). The val WER drop (0.2826 → 0.2326) is not independent
evidence: PopSign is likely to share signers with ASL Signs (same game), and the validation phrases are
built from ASL Signs fold-0 signers, so some of it may be the val signers' own PopSign clips; the
WLASL test half and ASL Citizen test split are signer-disjoint from their own datasets' training
signers, and no shared signers with ASL Signs or PopSign are known (only WLASL vs ASL Signs was
screened, `signer_overlap`, and a screen is not proof).

**Frozen phrases06 (research only).** Same freeze record; the profile check confirms it read the
research-only ASL Citizen release. WLASL test half: frozen phrases05 → phrases06 82.5 → 85.0 % (+56 /
−36, p = 0.047); frozen phrases07 → phrases06 84.0 → 85.0 % (+46 / −38, p = 0.45, no detectable
difference). Its ASL Citizen score (87.1 %) is in-domain and not compared with the others. Against the
legacy phrases06 (85.9 / 86.8 %) only unpaired: the research re-score kept no per-clip CSVs. It stopped
early at epoch 29 under the same patience rule the other two ran with (they never triggered it). Not
exportable (`research_only` meta).

**Browser / export review (2026-11-15; ASL-App issues are the app owner's to fix, this repo supplies the
reference).**

| Finding | Here | Reproduction now |
|---|---|---|
| Browser feature timing follows the device's processing rate (2 s at 15 / 30 / 60 fps → 30 / 60 / 120 frames, velocities halved / doubled) | `preprocess.json` `frame_rate` is now an exact rule: resample the raw stream onto a 30 fps grid by capture timestamp — nearest frame at ≥ 95 % of 30 fps, linear interpolation below (repeating frames would add zero-velocity steps training never sees), 0.5 ms snap; reference `browser_fixtures.resample_plan` / `apply_plan`, vectors `timing_vectors.json` | constant-speed 2 s motion at 15 / 30 / 60 / 120 fps → 59–60 frames, velocity 0.00333 per frame at every rate; 30 fps → identity; 120 / 60 → every 4th / 2nd; jittered 30 fps picks frames |
| Browser `buildFeatures(trim=true)` trims before hold-fill: a face seen only before the hand span becomes zeros (Python fills, then trims) | contract case + fixture `face_lost_in_span` (the trimmed span must hold the last face seen); spec `segment` says fill, then trim | trim-then-fill differs in 5,040 feature values (face all 0); the fixture fails a port that trims first |
| `export_onnx` verification passed NaN logits (`max(0, nan)` = 0) | finite PyTorch and ONNX logits required before the comparison | injected NaN: the old check reported 0 error; the new one exits |

### Review of ce3c929 (three more, fixed)

| Finding | Fix | Reproduction now |
|---|---|---|
| A report in the previous format passed `require_accepted` although today's checks fail that release | `VALIDATION_POLICY` (2) written into every report and required at preflight; bump on any new check. Also new under policy 2: every ASL Signs signer must be in the frozen folds | old report → "validated under policy None, current is 2 — run validate again"; revalidated → accepted |
| phrases05 → phrases07 could not be compared: the ASL Signs share (70 → 50 %) sat inside the strictly compared `data` | Shares moved to `sampling` (free, like `--extra-frac`); dataset entries keep only identity (release + annotations). An extra dataset present in both runs (same type + LMDB name) must be the same release | adding a dataset (0.7 → 0.5): "Spec matches"; another PopSign release: `differs in ['extra_data.popsign']`; seed: `recipe.seed`, `val_set`; uncommitted code: `code` |
| '-1', 'P-1' (Citizen) and '-1', '4a.-1' (PopSign) parsed as signer 1 | Whole-id formats: Citizen `P<digits>` / `<digits>` / a non-negative integer; PopSign `<letters/digits>.<digits>`; anything else raises | those four and 'P15_2', '1.5', '4a.6001x', '.6001', NaN, None all rejected; real ids (P15, 4a.6001, gtsignstudy4a.8031) parse |

Every release must therefore be validated with the current code after it is built; the audit
notebooks report example signer ids per dataset before any build, so an unexpected id format shows
up there first.

### Review of d9741ac (four, fixed — policy 4)

| Finding | Fix | Reproduction now |
|---|---|---|
| A sign map with two signs on one class (`boat`, `bye` → 0) passed validation and acceptance | `validate.sign_map_problems`: distinct names onto exactly 0..N-1 (integers, duplicate JSON keys detected); the map's digest in the build manifest, the validation report and the release `require_accepted` returns (so in the experiment spec) | validate: "indices not exactly 0..249 (repeated [0], missing [27, 33])" + manifest digest mismatch; acceptance: "sign map changed since validation" |
| Coordinates reinterpreted float32 → int32 kept the digest; a length key 30 → 2 passed acceptance (bucketing read 2 frames) | `contract.RECORD_DTYPES` enforced by `load_record` (every consumer); `content_digest` now hashes each field's dtype and shape and the stored length key | int32: refused at load ("stored dtypes {'coords': 'torch.int32'}"), validate + acceptance fail; length: validate "length 2 / n_frames 30 / T 30" + digest, acceptance digest mismatch |
| `browser_fixtures.verify` did not compare `subsample` | compared with the other fields | final-frame indices in `long_clip`'s subsample: "['subsample'] not reproducible" |
| Spec said "the segment's median shoulder width"; the code uses every pose frame of the whole clip before trimming | `preprocess.json` wording | — |

**Canonical label map (policy 5).** `core/data/asl_signs_vocab.json` is a byte copy of the competition's
`sign_to_prediction_index_map.json` (re-downloaded from Kaggle `asl-signs` 2026-11-14: sha256 1fe747c2…,
identical to every local copy; the app's label order matches). Its 250 assignments are immutable;
`vocab.py` pins the hash. Enforced: every builder checks its `--sign-map` (ASL Signs: the data dir's map)
is exactly canonical **before** any gloss is remapped (`wlasl_gloss_map.json` holds canonical indices, in
the shared-sign build too); `validate` requires the release map to be exactly canonical, or — WLASL all
glosses only — the canonical 250 unchanged with extra classes numbered from 250; `phrases.train` the same
for the data dir and `--vocab`; `export_onnx` exactly canonical. Every other reader too (review of 460465e:
isolated training and `cross_dataset --sign-map` read the JSON directly; a search found four more —
`get_data_loaders`, `breakdown`, `sign_reliability`, `aspect_compare`, `error_review` — all now through
`vocab.load_base`, `cross_dataset` through `load_map(extended=True)`; `config.SIGN2INDEX_JSON`, unused,
removed). Swapped map refused by each; WLASL-all extension accepted by `cross_dataset`. Tests (`check_contract`, 8 cases + a
release): two base indices swapped (still a bijection) refused by the builder, by `validate` and as an
extension; valid extras appended accepted; extras as a base vocabulary, extras not from 250 and two signs on
one class refused; the existing 2,017-label WLASL-all map passes as an extension.

The acceptance suite re-run under policies 4 and 5: every earlier refusal unchanged (checked on full messages).
None of these showed the existing data wrong; the digests change, so every release is built and
validated anew anyway.

### Order of work (agreed 2026-11-14)

1. All four full-data audit reports reviewed, every finding resolved (CPU only; times provisional
   until the jobs start).
2. Aspect correction: a gap between the per-clip shoulder factor (builders) and the geometric
   (W / H) / 0.75 (browser) is **not by itself a bug** — they normalise different things (body
   proportion vs frame shape). Before any builder change: prepare the same WLASL / ASL Citizen clips
   both ways and compare the resulting geometry (shoulder drop / width, hand size relative to the
   shoulders, joint-angle features) and a frozen checkpoint's predictions on each; change the
   builders only if the browser-style preparation lands outside what training covers.
3. Build every release, validate under the current policy (6; `--raw-root`).
4. Freeze the code revision and the shared dependencies (init checkpoint, vocabulary, validation
   phrases, releases) in the reference spec.
5. Only then the GPU runs (re-score, then phrases05 / phrases07 under one `--frozen-spec`) — even
   if that moves them past tomorrow morning.
