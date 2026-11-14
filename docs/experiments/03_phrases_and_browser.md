# Continuous phrases and the browser model (approach C)

phrases01–05, ONNX export and browser conventions, Kaggle GPU runner, live-test diagnoses, per-sign reliability. Entries in chronological order; paths are as they were at the time.

---

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
| one-hand mode on two-hand clips (345) | 25.8 → 39.4 % (helps, +71 / −24, p 1e-6) | 76.8 → 57.1 % (**hurts**, +13 / −81, p 4e-13) | |
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

### phrases04: viewpoint augmentation on top of phrases03 (2026-11-12)
Live use of phrases03 needed several camera angles / distances before a sign was recognised. A
synthetic sweep on the WLASL test half (phrases03, scratch script, not committed) showed distance
alone costs nothing (shoulder-width scaling: ×0.5 / ×2 = 80.1 %), while horizontal stretch (aspect,
body turn) cost 2–14 points (×0.75 78.3, ×1.33 75.2, ×0.6 69.4, ×1.67 65.9 %), vertical squash ×0.75
75.2 %, tilt −25° 75.1 %. Phrase training only used ±15° rotation at p 0.5. New flags `--aug-shear`,
`--aug-scale` (training only, default off) pass `spatial_affine`'s existing shear / per-axis scale
through. phrases04 = phrases03 + `--aug-rotate 20 --aug-shear 0.15 --aug-scale 0.3
--aug-affine-prob 0.8`; evaluated on the clean (unaugmented) test half. Kaggle `islr-phrases04-viewpoint`
(commit 34f183e), 3m52s/epoch, ran all 30 epochs, best epoch 24.

| WLASL test half (824 clips) | phrases03 | phrases04 | paired |
|---|---|---|---|
| exact decode | 80.1 % | 80.8 % | +50 / −44, p 0.61 |
| pooled top-1 / top-5 | 81.2 / 94.5 % | 82.6 / 96.1 % | top-5 +23 / −10, p 0.035 |
| two-hand clips (345), exact | 76.8 % | 78.6 % | p 0.47 |
| **aspect correction undone** (raw landscape x) | 39.2 % | **77.4 %** | +338 / −23, p 6e-73 |
| ASL Signs synthetic val WER | 0.296 | 0.291 | |

Clean accuracy unchanged (top-5 slightly up), and the one real framing shift we have — WLASL's own
uncorrected landscape coordinates, x off by a factor ~2.3 (median aspect k) — went from 39 % to 77 %:
on WLASL's held-out signers, the model is much less sensitive to a horizontal mis-scaling of the
coordinates. That is all this tests: a 2D coordinate distortion of WLASL clips. It does not show the
crop / x-rescale can be skipped in the app, and real 3D body or head turns, camera pitch, occlusion
and tracking degradation at distance remain untested. Selection caveat: the augmentation was chosen
from a sweep on this same test half, so these numbers are development evidence, not an independent
test — fresh camera recordings are the next independent check. Exported to `exports/phrases04/`.

### Live report: PEN / PENCIL → boat (phrases04) — diagnosis (2026-11-12)
Both are in the vocabulary (pen 167, pencil 168; 399 / 392 ASL Signs clips). Offline findings:
- **Weak even in-domain.** On fold-0 held-out ASL Signs signers (all 14,076 val clips scored, trimmed
  as in the browser; overall 62.2 % exact / 65.1 % pooled): pen 38 %, pencil 40 % pooled, mostly
  confused with each other (pen → pencil 17 of 39 errors, pencil → pen 21 of 34). Boat 67 %.
- **No boat attraction offline.** 0 of 120 held-out pen/pencil clips → boat; boat is the wrong answer
  for 0.6 % of the errors on the 321 val clips with both hands in a frame, and 3 / 824 on WLASL test.
- **Second-hand data.** Both hands in a frame: pen 2.8 %, pencil 2.6 %, boat 3.8 % of ASL Signs clips.
  WLASL has no PEN, 5 PENCIL (1 in the train half; hands ~1 shoulder width apart) and 8 BOAT (5 in the
  train half, hands touching: wrist gap 0.07 sw) — boat is the most frequent hands-touching sign
  among the train half's 37 such clips.
- **Synthetic base hand does not reproduce it.** A static base hand (shape from the WLASL pencil clips)
  inserted 0.1 / 0.25 / ~0.7 shoulder widths from the active wrist of the one-hand held-out pen /
  pencil clips: boat ≤ 3 % of predictions; pencil rises 42 → 49–53 %, pen falls 39 → 23–29 % (toward
  pencil). So "a base hand → boat" is not supported, within the limits of a static synthetic hand.
Next: the live landmarks themselves (frames from the app, `browser_fixtures` format) to check what
the model actually receives (slots, mirroring, framing ratio, presence) before changing data.
MS-ASL has 50 PENCIL / 24 BOAT clips but no usable landmarks (re-extraction from YouTube).

### Per-sign reliability for the app — phrases04 (2026-11-12)
`sign_reliability.py` scores every clip by signers the checkpoint never saw (ASL Signs fold-0 val,
14,076 clips + WLASL test half, 824) the way the app can: "pooled" = per sign, max over the
segment's frames of its softmax probability (blank excluded). Output `exports/phrases04/sign_reliability.json`
(per sign: n, top-1/3/5, exact decode, top-3 confusions; calibration table; thresholds).

| | top-1 | top-3 | top-5 | exact decode |
|---|---|---|---|---|
| combined (14,900) | 66.0 % | 81.9 % | 86.0 % | 63.3 % |
| ASL Signs fold 0 | 65.1 % | 81.1 % | 85.4 % | 62.2 % |
| WLASL test half | 82.7 % | 94.3 % | 96.1 % | 80.8 % |

Calibration (top-1 accuracy by top-1 pooled probability p1): < 0.2 → 8 %, 0.5–0.6 → 36 %, 0.8–0.9 →
57 %, ≥ 0.9 → 88 % (55 % of clips). Thresholds: p1 ≥ 0.685 keeps 72 % of clips at 80 % precision;
≥ 0.928 keeps 51 % at 90 %; ≥ 0.988 keeps 34 % at 95 %. Product reading: show the top 3 (82 %),
and a "not sure" state below ~0.7. Most frequent confusions are near-synonyms (cat → kitty, goose →
duck, wake ↔ awake, mouth → lips, pencil ↔ pen, man → dad). Calibration is on held-out *dataset*
signers; live webcam calibration is unmeasured.

---

### phrases05: all 2,000 WLASL glosses as extra labels (2026-11-12)

phrases04's recipe with the full WLASL vocabulary: train-half signers' clips of every gloss (2,017
labels — the 250 first, then 1,767 WLASL-only glosses), 30 % of phrases (phrases04: 15 %, shared
signs only). The idea: more signs to tell apart → features that separate signs better, even if the
extra labels are cut off at export (`load_ctc_model` slices the head back to 250 + blank, exact).
Kaggle `islr-phrases05-vocab` (commit 5d46edc, T4, 30 epochs × ~4.2 min):

```
islr.phrases.train ... --init fold0_graph/best_phase1.pth --extra-data-dir {WLASL_ALL_DIR}
  --extra-lmdb-path {WLASL_ALL_DIR}/wlasl_all.lmdb.mdb --vocab {WLASL_ALL_DIR}/sign_to_prediction_index_map.json
  --extra-frac 0.3 --aug-rotate 20 --aug-shear 0.15 --aug-scale 0.3 --aug-affine-prob 0.8
```

| | phrases04 | phrases05 | paired (824 clips) |
|---|---|---|---|
| Synthetic-phrase val WER (held-out ASL Signs signers) | 0.291 | 0.285 | — |
| WLASL test half, single-sign exact decode | 80.8 % | **83.7 %** | +65 / −41, McNemar p = 0.025 |
| WLASL test half, pooled top-1 | 82.6 % | 84.1 % | +54 / −42, p = 0.26 |
| WLASL test half, framing correction undone | 77.4 % | 82.0 % | — |
| WLASL top-5 | 96.1 % | 95.6 % | — |
| ASL Signs fold-0 val, pooled top-1 / top-3 (14,076 clips) | 65.1 / 81.1 % | 64.3 / 80.9 % | not paired (phrases04 per-clip file not kept) |

Reading: on WLASL's unseen signers the exact decode gains ~3 points (significant), but pooled
top-1 — what the app's top-3 display uses — moves less and not significantly, so most of the gain
is the decode emitting the right single sign more cleanly. On ASL Signs' held-out signers it is
0.8 points *lower* (top-3 unchanged). The extra data is WLASL, so a WLASL gain partly reflects more
in-domain data (WLASL is now 30 % of phrases, not 15 %) rather than the wider vocabulary; the two are
confounded. Not exported — phrases04 stays the browser model until a live test or an independent set
(ASL Citizen test split, research only) separates them.

**Update (same day): the independent ASL Citizen test separates them** — phrases05 72.4 % vs
phrases04 68.7 % exact (+760 / −448, p = 2e-19), pooled top-1 75.4 vs 74.2 % (p = 0.002), identical
glosses 79.0 vs 76.2 %; see [04](04_data_cross_dataset.md). phrases05 uses no ASL Citizen data, so it
is eligible for the app; the ASL Signs held-out drop (65.1 → 64.3 % top-1) remains.

### phrases05 exported as the browser model (2026-11-12)

`exports/phrases05/` (`islr.export.export_onnx`; the 2,017-label head is cut to 250 + blank):
ONNX vs PyTorch max |Δlogit| 6.3e-5, frame argmax 32/32. `labels.json` and the preprocessing in
`preprocess.json` are identical to phrases04 (top-level `blank` 250; the nested `checkpoint_meta`
keeps the training head's 2017), so the app swaps the ONNX + `sign_reliability.json` only.
Fixtures (same two parquets): `100015657` → cloud, concat → cloud animal — as phrases04.
Reliability: Kaggle run's `phrases05_reliability.json` (same checkpoint); 0.8 precision at
pooled p1 ≥ 0.68 (phrases04 0.68), so the "not sure" cut ~0.7 is unchanged. `exports/phrases04/`
kept until the app has switched.

### phrases05b: phrases05's 30 % WLASL share, shared signs only (ablation, 2026-11-12)

Separates phrases05's two changes (WLASL 15 → 30 % of phrases, and all 2,000 glosses as labels).
phrases04 recipe with `--extra-data-dir {WLASL_DIR} --extra-frac 0.3` (847 clips, 206 signs, no
`--vocab`); Kaggle `islr-phrases05b-shared30` (commit c0cc105). Val WER 0.287.

| | phrases04 (15 %) | phrases05b (30 %, shared) | phrases05 (30 %, all glosses) |
|---|---|---|---|
| ASL Citizen test split (3,408 clips, 11 signers), exact / pooled top-1 | 67.2 / 73.2 % | 68.5 / 74.6 % | 70.9 / 74.4 % |
| WLASL test half, exact | 80.8 % | 82.8 % | 83.7 % |
| ASL Signs held-out top-1 | 65.1 % | 65.3 % | 64.3 % |

Paired on ASL Citizen test: 04 → 05b exact +222 / −177 (p = 0.03), pooled +187 / −141 (p = 0.01);
05b → 05 exact +267 / −184 (p = 1e-4), pooled +178 / −185 (p = 0.75). WLASL differences are not
significant (p ≥ 0.07). Reading: the larger WLASL share gives the ranking gain (pooled top-1, what the
app shows); the wider vocabulary adds a cleaner single-sign decode on top but no ranking gain. Both
help; phrases05 stays the browser model.

### phrases06: + ASL Citizen's train split (research only, 2026-11-12)

Does more Deaf webcam data help beyond WLASL? phrases05 recipe + ASL Citizen official train split
(3,809 clips, 35 signers, 208 signs; identical + variant glosses only — the 611 synonym clips, mapped
by a non-signer, are left out) at 15 % of phrases: `--extra-data-dir {WLASL_ALL_DIR} /tmp/aslc
--extra-frac 0.3 0.15 --extra-kinds identical variant` (LMDB built in the same notebook). Kaggle
`islr-phrases06-aslcitizen-research` (commit c0cc105); checkpoint meta `research_only: true`
(export refuses it); outputs `results/phrases/phrases06-aslcitizen-research/`. Val WER 0.292.

| ASL Citizen **test split** (3,408 clips, 11 unseen signers) | phrases05 | phrases06 |
|---|---|---|
| Exact decode / pooled top-1 | 70.9 / 74.4 % | **86.7 / 89.6 %** |
| Identical / variant / synonym glosses | 77.4 / 59.3 / 66.7 % | 89.2 / 86.9 / 64.6 % |
| Two-hand clips | 63.6 % | 84.8 % |
| Signer p10 / median / p90 | 65 / 71 / 77 % | 83 / 86 / 94 % |
| Framing correction undone | 69.5 % | 86.2 % |
| WLASL test half (not trained on) | 83.7 % | **86.9 %** |
| ASL Signs held-out top-1 | 64.3 % | 64.3 % |

Paired 05 → 06: ASL Citizen test exact +601 / −63 (p = 4e-111), pooled +568 / −51; WLASL +51 / −25
(p = 0.004).

Reading — two different effects:
- **Dataset-specific: large.** Most of the ASL Citizen gain is learning ASL Citizen's forms of these
  signs from other signers: numbered variants jump 59 → 87 % (the training clips show the variant
  forms), while the synonym glosses, left out of training, do not improve (66.7 → 64.6 %). So 86.7 %
  is not what a new data source of a *different* style would give.
- **General: real but smaller.** WLASL, which phrases06 did not train on any more than phrases05, gains
  3.2 points (significant), and ASL Signs is unchanged — more Deaf signers recorded on webcams make
  the model better on other signers too.
- For the product: data like this helps, but ASL Citizen cannot ship. PopSign ASL (CC BY 4.0, the same
  250 signs) is the candidate for the shippable version — to check next.

### phrases05-long: phrases05 with a 60-epoch schedule (2026-11-13)

Was phrases05 still learning at epoch 30? Same recipe with `--epochs 60 --patience 15` (OneCycleLR
spans all 60); Kaggle `islr-phrases05-long` (commit 2c6dc76; first run on the schandranyc account),
outputs `results/phrases/phrases05-long/`, Kaggle Models variation `schandranyc/islr/pyTorch/phrases05-long`.
Best val WER 0.271 at epoch 40 (phrases05: 0.285), early stop at 55.

| | phrases05 (30 ep) | phrases05-long (60 ep) |
|---|---|---|
| Val WER (synthetic phrases, held-out signers) | 0.285 | **0.271** |
| ASL Signs held-out (fold 0, 14,076 clips) exact / pooled top-1 | 61.9 / 64.3 % | **63.2 / 65.6 %** |
| WLASL test half exact / pooled top-1 | 83.7 / 84.1 % | 83.0 / 83.4 % |
| ASL Citizen test split (3,408 clips, 11 signers) exact / pooled top-1 | 70.9 / 74.4 % | 70.6 / 74.3 % |
| Reliability: share of clips kept at 90 % precision | 49.7 % | 51.3 % |

Paired: ASL Signs held-out exact +1042 / −861 (p = 4e-5), pooled +960 / −781 (p = 2e-5); WLASL exact
+33 / −39 (p = 0.56); ASL Citizen test exact +202 / −212 (p = 0.66), pooled +173 / −176 (p = 0.91;
scored in the phrases06-long notebook from the Kaggle Models checkpoint). Reading: longer training
helps on the training dataset's own (unseen) signers but carries over to neither independent test
(other cameras and signers). phrases05 stays the browser model.

### phrases06-long: phrases06 with a 60-epoch schedule (research only, 2026-11-13)

phrases06 recipe + `--epochs 60 --patience 15`; ASL Citizen re-extracted on the schandranyc account
(8,557 clips, 0 errors, same aspect correction as before). Kaggle `islr-phrases06-long-research`
(commit 4bf05ca), outputs `results/phrases/phrases06-long-research/`, private Kaggle Models
`schandranyc/islr-research/pyTorch/phrases06-long` (meta `research_only: true`). Best val WER 0.270
at epoch 49 (phrases06: 0.292).

| | phrases06 | phrases06-long |
|---|---|---|
| ASL Citizen test exact / pooled top-1 | 86.7 / 89.6 % | 87.3 / **90.8 %** |
| Identical / variant / synonym (exact) | 89.2 / 86.9 / 64.6 % | 89.8 / 87.3 / 66.7 % |
| WLASL test half exact / pooled | 86.9 / 87.0 % | 87.5 / 87.9 % |
| ASL Signs held-out exact / pooled | 61.9 / 64.3 % | **63.6 / 65.8 %** |

Paired: ASL Citizen pooled +107 / −65 (p = 0.002), exact +118 / −97 (p = 0.17); ASL Signs held-out
exact +1033 / −789 (p = 1e-8); WLASL +30 / −25 (p = 0.59). Reading: as for phrases05-long, the longer
schedule helps on datasets the model trains on (here ASL Signs and ASL Citizen, both with unseen
signers) and not on WLASL — it fits the training sources better rather than generalising further.

### phrases07: + PopSign ASL (CC BY 4.0, shippable) (2026-11-13)

The shippable counterpart of phrases06. PopSign ASL v1.0 game train videos (the same 250 signs, Deaf
signers, portrait phone video like ASL Signs): 4 clips per signer × sign, evenly spaced over each
pair's recordings → 26,278 clips from 31 signers, landmarks extracted on Kaggle CPU in 12 shards
(`scripts/extract_popsign.py`, downscaled to 1280 px; 24,946 clips at 1944×2592, 1,332 at
1080×1920; 1 clip mirrored), LMDB built in the training notebook without aspect correction (raw
shoulder ratio 0.41 vs ASL Signs 0.40), 0 errors. Both hands in some frame in 14 % of clips (ASL
Signs 2 %). Recipe = phrases05 (30 epochs) + `--extra-data-dir {WLASL_ALL_DIR} /tmp/popsign
--extra-frac 0.3 0.2`. Kaggle `islr-phrases07-popsign` (commit 393a255), outputs
`results/phrases/phrases07-popsign/`, Kaggle Models `schandranyc/islr/pyTorch/phrases07`. Val WER
**0.234** (phrases05 0.285).

| vs phrases05 | phrases05 | phrases07 | paired |
|---|---|---|---|
| ASL Citizen test (3,408 clips, 11 unseen signers) exact | 70.9 % | **73.4 %** | +227 / −144, p = 2e-5 |
| ASL Citizen test pooled top-1 | 74.4 % | **75.8 %** | +180 / −131, p = 0.006 |
| ASL Citizen identical / variant / synonym (exact) | 77.4 / 59.3 / 66.7 % | 80.7 / 60.6 / 67.1 % | identical p = 7e-6 |
| WLASL test half exact | 83.7 % | 84.2 % | +33 / −29, p = 0.7 |
| ASL Signs held-out exact (fold 0) | 61.9 % | 68.0 %* | +1409 / −552 |
| Reliability: clips kept at 90 % precision (offline; set 94.5 % ASL Signs) | 49.7 % | 62.7 %* | |

\* ASL Signs scores may be inflated by shared signers (below); all three val signers improve
(34503 53 → 62 %, 49445 63 → 68 %, 62590 77 → 79 % pooled).

**Signer overlap** (`islr.evaluation.signer_overlap`, in the same notebook): reliability-weighted
face / body / hand-proportion fingerprints, 400 clips per signer. Calibration — ASL Signs split
halves: same person ≤ 0.32, different people ≥ 0.46 (AUC 1.0); PopSign halves: same ≤ 0.35 (AUC
1.0). No PopSign signer falls in the same-person range of an ASL Signs signer (0 of 31 flagged), but
three are at 0.42–0.43 — closer than any two different ASL Signs signers — to ASL Signs 26734,
28656 (both training signers) and 61333 (training). None of the fold-0 val signers is within 0.57.
Reading: the screen found no match for the val signers, but it **cannot establish that they are
absent** — its perfect AUC measures consistency within one dataset, while matching a person across
datasets (other phone, session, framing) is harder and unvalidated; it also scored the 1,332 9:16
PopSign clips as 3:4 (fixed afterwards: per-clip aspect from the frame size, needs a rerun). So the
ASL Signs gain stays qualified: shared signers are possible, and PopSign is the same game and phone
set-up as the val signers in any case.

ASL Citizen per signer: 10 of 11 improve on exact decode (largest +9.0 points, one −1.6); resampling
signers gives +0.9 to +4.2 points (95 % bootstrap). The reliability figure is an offline estimate on
held-out clips (94.5 % ASL Signs, 5.5 % WLASL): it says nothing about live camera input, signs outside
the vocabulary or ordinary background movement.

Reading: the first shippable gain on the independent ASL Citizen test since phrases05 (+2.5 exact,
mostly on identical glosses; variant forms barely move, 59.3 → 60.6 % — PopSign signs the ASL Signs
forms, so that gap remains). With phrases05-long's null result, better training coverage — not more
epochs — is the lever. WLASL
does not move. Exported to `exports/phrases07/` (ONNX verified, Δlogit 3e-4; `sign_reliability.json`;
fixtures; `ATTRIBUTION.md` — **PopSign CC BY 4.0 must be credited in the app**). Candidate to replace
phrases05 as the browser model — to be confirmed on the same recorded camera clips for both models
(signers, distances, angles, out-of-vocabulary signs and no signing; first-attempt recognition,
confidently wrong outputs, latency): offline accuracy and synthetic-phrase WER cannot show whether
the product improves.

**Extraction caveat (found 2026-11-13, review):** every extractor (WLASL notebook, ASL Citizen,
PopSign) reused one MediaPipe Holistic per worker in video mode, so a clip's first frames could be
tracked from the previous, unrelated clip. Fixed in the shared cell (fresh tracker per video,
506a5ea; WLASL notebook 07f23b7). The ASL Signs competition data is not affected (not extracted by us).
Measured on the first 160 PopSign shard-0 clips (Kaggle `tracker-reset-check`), shared vs fresh
tracker, against a noise floor of production vs a shared-tracker re-run:

| | shared vs fresh | noise floor |
|---|---|---|
| Hand detection rate (left / right) | 23.3 / 16.2 % vs 23.2 / 16.4 % | — |
| Frames whose hand detection differs | 2–3 % | 0.2–0.7 % |
| Hand landmark xy difference (normalised) | 0.008 | 0.001–0.002 |
| Upper-body pose difference, shoulder widths (median): first 5 frames / rest | 0.051 / 0.030 | ~0.003 (normalised) |
| Shoulders / elbows / wrists (median, p90) | 0.067, 0.131 | |
| phrases05 prediction changes (never saw PopSign) | 13.7 % of clips; accuracy 83.1 → 86.3 % (+11 / −6, p ≈ 0.3) | |
| phrases07 prediction changes (trained on these clips) | 3.7 % | |

So the shared tracker did not lose detections but shifted landmarks through the whole clip (largest
at the start), enough to change one prediction in seven for a model that had not seen the data, with
no clear accuracy direction on 160 clips. The fresh tracker is the correct reference (each clip
independent, as in the browser where tracking starts from the person's own frames). Re-extraction
started (CPU only; v1 outputs kept): `extract-popsign-v2-s0..11`, `extract-aslcitizen-v2-s0..2`. Raw outputs (both
parquet sets, per-clip comparison, scripts): `results/evaluation/tracker_reset/`. Rebuild from v2 notebooks
only — the builders now exit if a clip path appears in more than one shard folder (d108184).
Then: rebuild, re-score phrases05 / phrases07 on ASL Citizen v2 and retrain phrases07 on PopSign v2.
WLASL (training + test half): re-extraction started too — `scripts/extract_wlasl.py` (shared cell: fresh
tracker, MediaPipe 0.10.14 pinned — the old notebook was unpinned; full-size frames; same file names, so
LMDB keys are unchanged), 4 shards `extract-wlasl-v2-s0..3`, built with `build_wlasl_lmdb --input-root`
(byte-identical to `--wlasl-dir` on the same parquets). Check after the build that the signer halves
(`signer_split`) are unchanged.

### phrases07-long: phrases07 with a 60-epoch schedule (2026-11-14)

phrases07 recipe + `--epochs 60 --patience 15`, the same v1 PopSign / ASL Citizen extractions (shared
tracker, pre-contract pipeline — a **legacy-pipeline reference**, like every score above). Kaggle
`islr-phrases07-long-popsign`, outputs `results/phrases/phrases07-long-popsign/`. Best val WER
**0.204** at epoch 54 (phrases07: 0.234).

| vs phrases07 (paired, same clips) | phrases07 | phrases07-long | paired |
|---|---|---|---|
| ASL Citizen test exact | 73.4 % | 73.9 % | +151 / −132, p = 0.28 |
| ASL Citizen test pooled top-1 | 75.8 % | 75.9 % | +125 / −121, p = 0.85 |
| ASL Citizen identical / variant / synonym (exact) | 80.7 / 60.6 / 67.1 % | 81.0 / 60.9 / 70.7 % | |
| WLASL test half exact | 84.2 % | 85.7 % | +41 / −29, p = 0.19 |
| ASL Signs held-out exact (fold 0) | 68.0 %* | 70.6 %* | +984 / −616, p = 3e-20 |
| Reliability: clips kept at 90 % precision (offline) | 62.7 %* | 66.3 %* | |

\* ASL Signs-dominated, possibly inflated by PopSign / ASL Signs shared signers (phrases07 entry).
ASL Citizen per signer: 6 of 11 up, 5 down.

Reading: the same pattern as phrases05-long and phrases06-long — the longer schedule fits the
training sources (ASL Signs and PopSign, the same game set-up) and does not carry over to either
independent test. phrases07 (30 epochs) stays the candidate; not exported.
