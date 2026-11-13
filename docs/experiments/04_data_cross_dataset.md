# Data findings and cross-dataset tests

One-hand training data, WLASL / MS-ASL checks, WLASL metric-vs-model gap. Entries in chronological order; paths are as they were at the time.

---

### ASL Signs data has (almost) one hand per clip (2026-11-11)
Live browser test: CHOCOLATE (C hand on the back of a base hand) decoded as police. Pipeline
mistakes simulated on held-out chocolate clips (landscape framing, selfie mirror, swapped slots,
no nose subtraction, z = 0, 10–15 fps, hand shifted toward midline) give other errors, never
police; held-out chocolate/police are never confused with each other (0/57, 0/56).
Cause found in the data: **both hands are detected in the same frame in a median 2 % of clips**
over the 250 signs (40 clips each), including clearly two-handed signs — shoe 1 %, book / cloud
/ bath / car / rain 2 %, chocolate 4 %, police 6 % (200 clips each). The raw competition parquet
for a two-handed CLOUD clip has 28 left-hand frames and 0 right-hand frames, so it is the source
extraction, not `hand_presence`. The models see a second hand rarely; a correctly tracked base
hand (MediaPipe Tasks in the browser tracks both) is uncommon input for them. (An earlier version
of this entry said every model "learned signs from one hand" and made one-hand input a browser
rule — overstated: sparse detections do not show the models use no second-hand information.)
Removing the less active hand on held-out chocolate clips: 47/57 → 47/57 (those clips rarely have
a second hand, so this says little). **Targeted test** — phrase model, single-sign decode, every
held-out clip with detections of both hands (review follow-up, 2026-11-11):

| Held-out clips | n | two hands | one hand (less active removed) | +/− | McNemar p |
|---|---|---|---|---|---|
| both hands detected somewhere | 575 | 44.2 % | 49.0 % | +66 / −38 | 0.01 |
| both hands in the same frame | 321 | 43.9 % | 44.5 % | +26 / −24 | 0.89 |
| signer 34503 / 49445 / 62590 | 294 / 271 / 10 | 36.4 / 51.3 / 80.0 % | 40.1 / 57.2 / 90.0 % | | 0.14 / 0.04 / 1.0 |

The gain comes from clips whose two hands are detected at different times (possibly tracking
flicker or one hand changing slot — untested), not from true two-hand frames. One-hand mode is an
experimental browser option to compare against the two-hand path, not a rule. Clips with two
hands are hard either way (~44 % vs 62 % overall). Two-handed signs need data with both hands
(WLASL below, app recordings).

**Other datasets checked (2026-11-11):**
- **WLASL** (our own `shravnchandr/preprocess-wlasl-mediapipe` Kaggle notebook: legacy MediaPipe
  Holistic, model_complexity 1, confidence 0.5, competition parquet format; 11,980 clips, 2,000
  signs; its 233 shared glosses map onto 206 of our 250 signs — 32 via synonyms such as father → dad,
  cop → police, a few uncertain (sit → chair, suppose → if, upstairs → up) — 1,671 clips, median 7 per sign). Sample of 31 clips: **both hands
  tracked** — two-handed signs have both hands in 29–53 % of frames (bath 0.49, book 0.43, car 0.52,
  chocolate 0.53, clean 0.45, cloud 0.29; every clip has some), the one-handed APPLE 0 %. Same
  Holistic as the competition yet both hands → the 2 % is specific to the ASL Signs data.
  Framing differs: shoulder ratio 0.94 (p10–p90 0.76–1.06; ASL Signs 0.40) — landscape video, as
  the aspect estimate predicts (≈ 0.9 for 16:9) — and signers are farther away (shoulder width 0.25
  of the frame vs 0.56). Usable only after per-clip aspect correction. `data/build_wlasl_lmdb.py` (run on Kaggle by
  `notebooks/kaggle/build-wlasl-lmdb/`) writes an ASL-Signs-compatible LMDB with x and z scaled per
  clip so the shoulder ratio is 0.40 (local smoke test, 27 clips: median k 2.44, ratio 0.400 after;
  phrase model single-sign exact 7/27 uncorrected → 8/27 corrected — too few clips to read).
  Full output ≈ 19 GB; the overlapping signs ≈ 2.7 GB (`kaggle kernels output --file-pattern`,
  rate-limited — better built into an LMDB on Kaggle). Licence: WLASL is C-UDA (computational use).
- **WLASL LMDB built on Kaggle** (`shravnchandr/build-wlasl-lmdb` v1, CPU, minutes; output = kernel
  output, not yet a dataset): 1,671 clips, 206 signs (2–21 clips each, median 7.5; 193 clips via
  synonym glosses), 67 signers, WLASL splits 1,174 / 296 / 201, all 25 fps per the WLASL json,
  median 68 frames, 0 errors; aspect k 1.17–3.61 (median 2.30), never the fallback.
  **Second hand, per sign** (share of clips with both hands in some frame, 206 shared signs):
  WLASL median 12 % / mean 40 %, ASL Signs median 2 % / mean 3 %. 80 signs exceed 50 % in WLASL
  (chocolate, have, read, cut, tree: 100 %); the same signs have a median 2 % (max 12 %) in ASL Signs,
  and the per-sign rates are unrelated (Spearman −0.05) — ASL Signs' rare second-hand detections do
  not track which signs are two-handed.
- **Cross-dataset test** (fold-0 checkpoints, all 67 WLASL signers unseen, 1,671 clips):

  | | uncorrected | aspect-corrected |
  |---|---|---|
  | isolated `fold0_graph` top-1 (250-way) | 30.6 % | **64.2 %** (top-5 82.6 %; 66.7 % restricted to the 206 shared signs) |
  | phrase model `phrases01`, single-sign exact | 24.8 % | 45.4 % |

  - The aspect correction doubles accuracy: framing alone explains a 34-pt drop (the same mistake
    as an uncropped landscape webcam in the browser).
  - The isolated model transfers to new signers / cameras nearly at its fold-0 level (64.2 vs
    68.6 % in-dataset). Per WLASL signer: median 66.7 %, p10 8.6 %, p90 92.5 % (28 signers with
    ≥ 10 clips) — a few signers fail almost completely (not yet inspected). Synonym-gloss clips
    59.6 % vs identical-gloss 64.7 %.
  - The phrase model loses far more (45.4 %, vs 62.4 % single-sign in-dataset). Untested: WLASL
    clips are longer (median 68 frames at 25 fps vs 22 in ASL Signs) and the phrase model saw only
    0.8–1.25× tempo, the classifier 0.5–2.0×.
  - **Two-hand clips are much harder:** isolated 49.1 % (685 clips with both hands somewhere) vs
    74.6 % (986 without) — part may be the signs themselves. **One-hand mode on these clips:**
    isolated 49.1 → 51.4 % (+56 / −40, p = 0.13); phrase model **26.9 → 37.1 %** (+124 / −54,
    p < 0.001). On real two-handed data, removing the less active hand clearly helps the phrase
    model and is neutral-to-positive for the classifier — the second hand is input both models
    handle poorly, consistent with ASL Signs barely containing it.
- **MS-ASL** (`shravnchandr/ms-asl-nose-reference-mediapipe-pose`, private, 9.9 GB): not raw
  landmarks — `.npy` of 60 resampled frames × 1,848 standardised features (264 × 7), with flipped /
  fast / slow copies and no missing-value markers; hand presence cannot be recovered without the
  (unlocated) producer. Raw re-extraction would need the YouTube videos (C-UDA; many links dead).
  Annotations local: 5,811 of 25,513 clips are on our signs.

### WLASL: metric vs model gap, and the "0 % signers" (2026-11-11)
`cross_dataset_eval` now also scores a phrase model like a classifier ("pooled": each sign's best
per-frame log-probability over the clip, blank excluded) and writes per-clip CSVs
(`results/wlasl_eval/`), so checkpoints can be compared clip by clip.

| WLASL, all 67 signers (1,671 clips) | exact decode | pooled top-1 | pooled top-5 | decode contains sign |
|---|---|---|---|---|
| phrases01 | 45.4 % | 55.5 % | 74.6 % | 56.1 % |
| phrases02 (tempo 0.5–2×) | 46.0 % | 57.1 % | 75.5 % | 58.1 % |
| fold0_graph classifier | — | 64.2 % (top-1) | 82.6 % | — |

- **About half the gap was the metric.** phrases01 decodes 2+ signs on 295 clips and nothing on 65;
  of the 170 clips pooled scoring recovers, 153 had decoded as two signs. The rest is real:
  pooled phrases01 vs classifier +134 / −278 clips (McNemar p 1e-12), phrases02 p 5e-10. In-domain
  the gap is ~5 points (67.0 vs 62.4 single-sign, phrase breakdown); cross-dataset ~8.
- **Paired phrases01 → phrases02:** exact +138 / −127 (p 0.54), pooled +126 / −100 (p 0.10) — the
  wider tempo range is still no clear gain.
- **The 0 % p10 signers are a small-sample artefact:** 17 of 67 signers have ≤ 3 clips (28 clips
  total); the 7 at 0 % have 1–2 clips each. Among the 22 signers with ≥ 20 clips (1,412 clips),
  classifier accuracy is 48–82 %, phrases01 pooled 34–81 %; nobody fails outright. Per-signer
  accuracy hardly tracks clip count, hand-visible fraction, aspect k or raw shoulder ratio
  (|Spearman| ≤ 0.18).
- **What does track errors is a second hand:** clips with both hands detected somewhere score 49.1 %
  vs 74.6 % (classifier) and 35.8 % vs 69.3 % (phrases01 pooled), and the phrase model loses more.
  Confounded with the sign (two-handed signs), but it matches the ASL Signs data gap (a second hand
  in a median 2 % of clips). By source, signingsavvy is lowest (54 %, 344 clips), aslbrick highest
  (80 %); clips where a hand is visible > 90 % of frames score lower (53 %), likely hands held up
  before / after the sign.

### ASL Citizen: independent test on 51 unseen Deaf signers (2026-11-12, research use only)

The first test set that played no part in any choice so far (WLASL's test half guided the
viewpoint augmentation). ASL Citizen (Microsoft Research License — research only; landmarks and
LMDB never leave private Kaggle notebooks; no model trained on it) — webcam recordings by Deaf
signers, landscape, unmirrored. Landmarks: `scripts/extract_aslcitizen.py` (3 CPU shards, same
MediaPipe Holistic 0.10.14 / settings / image as WLASL). LMDB + scoring in one private GPU notebook
(`islr-aslcitizen-eval`, commit b7d6f8b): `islr.core.data.build_aslcitizen_lmdb` (same worker as
WLASL: aspect correction to 0.40) then `islr.evaluation.cross_dataset` per checkpoint; outputs
`results/evaluation/aslcitizen/`.

Data: 8,557 clips with a gloss mapped onto the 250 (225 signs; kinds: 5,241 identical name, 2,705
numbered ASL-LEX variant, 611 synonym chosen by a non-signer), 51 signers, 0 errors. Raw shoulder
ratio median 0.72 (portrait ASL Signs 0.40) → aspect k median 1.81 (WLASL: smaller). Both hands in
some frame: 47 % of clips (ASL Signs median 2 %). All splits scored (no model saw any of these
signers); official test split alone is within ~2.5 points of the total for every model.

| Model | Exact decode / top-1 | Framing correction undone | Pooled top-1 | Top-5 | Identical / variant / synonym | Two-hand clips (3,987) | Signer p10 / median / p90 |
|---|---|---|---|---|---|---|---|
| fold0_graph (isolated classifier) | 65.2 % | 39.0 % | — | 81.9 % | 74.6 / 47.8 / 60.9 | 51.8 % | 58 / 67 / 80 |
| phrases01 (ASL Signs only) | 53.7 % | 34.7 % | 63.0 % | 81.1 % | 60.9 / 39.1 / 55.8 | 34.3 % | 40 / 55 / 69 |
| phrases03 (+ WLASL train half) | 69.0 % | 44.2 % | 73.5 % | 88.7 % | 76.6 / 55.9 / 61.2 | 60.4 % | 56 / 70 / 80 |
| phrases04 (+ viewpoint aug) — app model | 68.7 % | 66.9 % | 74.2 % | 88.8 % | 76.2 / 56.0 / 61.5 | 59.3 % | 56 / 69 / 81 |
| **phrases05 (+ all WLASL glosses, 30 %)** | **72.4 %** | **71.8 %** | **75.4 %** | **89.8 %** | 79.0 / 61.1 / 65.6 | 65.6 % | 64 / 72 / 84 |

Paired (exact decode, same 8,557 clips): phrases01 → 03 +1,640 / −331 (p ≈ 1e-207); 03 → 04
+503 / −522 (p = 0.57); **04 → 05 +760 / −448 (p = 2e-19)**; pooled 04 → 05 +542 / −443
(p = 0.002); fold0_graph → phrases04 +1,088 / −781 (p = 1e-12).

What it shows:
- The WLASL findings replicate on independent signers and recording conditions: WLASL training
  signers (two-handed data) are the big step (+15 points; two-hand clips 34 → 60 %), and
  phrases05's gain over phrases04 — not significant for pooled top-1 on WLASL (p = 0.26) — is
  clear here, including pooled top-1, the score the app's top-3 uses. The bottom-tenth signer
  improves most (56 → 64 %).
- Viewpoint augmentation (phrases04) adds nothing when framing is corrected but makes the models
  robust when it is not: 44.2 → 66.9 % with the correction undone (phrases05 71.8 %). ASL Citizen's
  raw framing is far from ASL Signs' (k 1.81), so this is the strongest evidence yet that the app's
  framing step matters less with these models — still 2D coordinate distortion, not body turns.
- Variant glosses score ~20 points below identical ones for every model — plausibly a different
  sign form under the same English word, so the identical-name number (79 % for phrases05) is the
  fairer accuracy estimate. Synonym mappings (unreviewed) are in between.
- Still isolated citation signs from a dataset, not live continuous use.

**Follow-up (2026-11-12):** training on ASL Citizen's train split (phrases06, research only) raises
its test split 70.9 → 86.7 %, mostly by learning its sign forms (variants 59 → 87 %; untrained synonym
glosses unchanged), and WLASL 83.7 → 86.9 % (p = 0.004). Details in
[03](03_phrases_and_browser.md#phrases06--asl-citizens-train-split-research-only-2026-11-12).
