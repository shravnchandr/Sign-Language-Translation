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
its own labels before it is frozen. Next: this rule, unchanged, on the holdout labels.

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
per-clip factor is robust to a mis-stated frame size, the geometric one is not. ASL Citizen pending.

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

### Review of ce3c929 (three more, fixed)

| Finding | Fix | Reproduction now |
|---|---|---|
| A report in the previous format passed `require_accepted` although today's checks fail that release | `VALIDATION_POLICY` (2) written into every report and required at preflight; bump on any new check. Also new under policy 2: every ASL Signs signer must be in the frozen folds | old report → "validated under policy None, current is 2 — run validate again"; revalidated → accepted |
| phrases05 → phrases07 could not be compared: the ASL Signs share (70 → 50 %) sat inside the strictly compared `data` | Shares moved to `sampling` (free, like `--extra-frac`); dataset entries keep only identity (release + annotations). An extra dataset present in both runs (same type + LMDB name) must be the same release | adding a dataset (0.7 → 0.5): "Spec matches"; another PopSign release: `differs in ['extra_data.popsign']`; seed: `recipe.seed`, `val_set`; uncommitted code: `code` |
| '-1', 'P-1' (Citizen) and '-1', '4a.-1' (PopSign) parsed as signer 1 | Whole-id formats: Citizen `P<digits>` / `<digits>` / a non-negative integer; PopSign `<letters/digits>.<digits>`; anything else raises | those four and 'P15_2', '1.5', '4a.6001x', '.6001', NaN, None all rejected; real ids (P15, 4a.6001, gtsignstudy4a.8031) parse |

Every release must therefore be validated with the current code after it is built; the audit
notebooks report example signer ids per dataset before any build, so an unexpected id format shows
up there first.

### Order of work (agreed 2026-11-14)

1. All four full-data audit reports reviewed, every finding resolved (CPU only; times provisional
   until the jobs start).
2. Aspect correction: a gap between the per-clip shoulder factor (builders) and the geometric
   (W / H) / 0.75 (browser) is **not by itself a bug** — they normalise different things (body
   proportion vs frame shape). Before any builder change: prepare the same WLASL / ASL Citizen clips
   both ways and compare the resulting geometry (shoulder drop / width, hand size relative to the
   shoulders, joint-angle features) and a frozen checkpoint's predictions on each; change the
   builders only if the browser-style preparation lands outside what training covers.
3. Build every release, validate under policy 2 (`--raw-root`).
4. Freeze the code revision and the shared dependencies (init checkpoint, vocabulary, validation
   phrases, releases) in the reference spec.
5. Only then the GPU runs (re-score, then phrases05 / phrases07 under one `--frozen-spec`) — even
   if that moves them past tomorrow morning.
