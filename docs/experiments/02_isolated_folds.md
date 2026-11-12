# Isolated classifier on signer folds (approaches A–B)

Signer-fold validation, review fixes, GraphConformer, bone augmentation, fingerspelling pre-training, VQ, AdaBN, ensembles, 1st-place anchor, error review. Entries in chronological order; paths are as they were at the time.

---

## Signer-fold validation & per-signer diagnostics (2026-10-07)
**Setup:** Run 005 recipe, `--val-fold k` (GroupKFold by signer, 7 folds × 3 signers), per-signer val
accuracy logged every epoch. Folds 0–1 run; fold 1 was interrupted at epoch 33/80.

| Split | Val signers | Best val | Per-signer |
|---|---|---|---|
| default (Run 005) | 2044, 37779, 53618 | 0.7665 | (not logged) |
| fold 0 | 34503, 49445, 62590 | **0.6680** (TTA 0.6674) | 0.557 / 0.654 / 0.794 |
| fold 1 (ep 33, partial) | 27610, 37055, 61333 | ~0.70 | 0.659 / 0.687 / 0.789 |

- **Signer variance dominates:** ~24 pt spread inside fold 0; the default split scores ~10 pt above fold 0 with
  the same recipe — it happens to hold the shortest-clip signers (median 9–16 frames). Fold 0 also shows a
  14 pt train–val gap (0.81 vs 0.67) that the default split did not.
- **`signer_diagnostics`** (500 clips/signer): every signer effectively uses one hand (the other is almost never
  detected; `dom_ratio` ≈ 1.0). Google clips end exactly at the last hand frame (`idle_trail` = 0 for all).
- **Spearman ρ with val acc over the 6 signers:** `no_hand` (frames with no hand detected) **−0.94**,
  `shoulder_w` −0.94, clip length −0.83, speed +0.20. `no_hand` ranks the six almost exactly and explains
  61333 (long clips but well tracked → 0.789), which broke the clip-length-only hypothesis (predicted to be
  fold 1's weakest; it was its strongest). n=6 and fold 1 partial — leads, not conclusions.
- Clip lengths over all 94,477 samples: median 22, 95th pct 135, 5.6% > 128 (subsampled), 0.3% > 256;
  highest over-128 share for 49445 (17.6%).

### Fold 0 + per-clip time stretch (`--stretch-mode sample --stretch-min 0.5 --stretch-max 2.0 --stretch-prob 0.8`)
| | Best | TTA | avg ep 61–70 | avg ep 71–80 |
|---|---|---|---|---|
| pooled | **0.6802** (+1.2) | 0.6786 (+1.1) | +1.3 | +1.1 |
| 34503 (slowest signer) | 0.5773 (+2.0) | +2.0 | +2.3 | +1.9 |
| 49445 | 0.6614 (+0.7) | +0.6 | +1.1 | +1.1 |
| 62590 | 0.8032 (+0.9) | +0.7 | +0.5 | +0.5 |

Consistent small gain on every signer and every summary, largest for the slowest signer; single seed vs
single seed (~1 pt run-to-run noise) so modest but likely real. Cost: 1m45s vs 1m21s per epoch (+29%).
Duration is a minor factor — it does not close 34503's ~22 pt gap to 62590. Next: hand dropout on top
(`--hand-drop-prob 0.5`), compared against this run.

### Fold 0 + stretch + hand dropout (`--hand-drop-prob 0.5`, span 0.1–0.4)
| | Baseline | Stretch | Stretch + drop | Δ drop vs stretch |
|---|---|---|---|---|
| best, pooled | 0.6680 | **0.6802** | 0.6774 | −0.3 |
| TTA, pooled | 0.6674 | 0.6786 | 0.6777 | −0.1 |
| avg ep 71–80, pooled | 0.6629 | 0.6743 | 0.6743 | ±0.0 |
| 34503 (avg 71–80) | 0.5521 | 0.5708 | 0.5674 | −0.3 |
| 49445 (avg 71–80) | 0.6458 | 0.6567 | 0.6658 | +0.9 |
| 62590 (avg 71–80) | 0.7919 | 0.7964 | 0.7901 | −0.6 |

**Null result.** Simulated tracking gaps do not help the signers with the most real gaps (34503 −0.3);
only 49445 (highest `no_hand`) moves, within noise. Train acc drops slightly (0.795 → 0.784). The
`no_hand` correlation is real but not addressable by robustness training: when MediaPipe loses the hand
(per prior work, mostly hand–face and hand–hand contact) the discriminative information is absent from
the input. *(Correction: this is a hypothesis, not a conclusion — six signers and one augmentation
run cannot separate tracking quality from duration, framing, signing style or the depth artifact
found afterwards.)* Hand dropout is
dropped from the recipe; current best = Run 005 recipe + per-clip stretch (fold 0: 0.6802).

---

## Review corrections & fixes (2026-10-08)
An external code review raised five pipeline issues and five over-strong conclusions. All
measurable claims reproduced locally (Run 005 checkpoint, local parquets):

| Issue | Measured | Fix (commit) |
|---|---|---|
| Nose-z subtracted from hand/face z (different MediaPipe depth frames) | raw wrist z 0.000 → stored 1.46–1.53; **52–82% of wrist velocity energy** is this artifact; hand–nose distance dominated by it | xy for dominance / hand–nose distance / shoulder scale; wrist z → 0; face z re-centred per frame; finger z kept (= raw wrist-relative z to 1e-6) (`70e651a`) |
| Finger dropout zeroed fingers before wrist subtraction → finger = −wrist | code | dropped finger collapses onto its wrist → exactly 0 in wrist frame (`5cf402c`) |
| Never-detected hands got noise/shift geometry in training only | code | re-zeroed after `augment_sample`; TTA noise skips them (`5cf402c`) |
| Batch stretch & TTA interpolated Δ1 with positions, and blended short clips' last frames with padding | 2× stretch kept Δ1 at 1.0 while positions moved 0.5 | all resampling via `_resample_prefix`: valid frames only, Δ1 rebuilt (`5cf402c`) |
| Padding leaked into depthwise conv; BatchNorm counted padded frames | +10 masked frames changed logits by up to **0.165** (6-frame clip) | re-mask before depthwise conv + `MaskedBatchNorm1d`; now ≤ 1e-3 (`b14f75e`) |
| `forward()` mutated its input | same tensor twice → max Δlogit **7.8** | clone at the model boundary (`b14f75e`) |
| No seeding; focal modulation from weighted smoothed CE; only augmented train acc | code; class counts 299–415 | `--seed` (default 42), `--loss ce`, `--train-eval-size` "Clean Train" metric (`b62a3e3`) |

**Corrected conclusions** (marked inline above):
- "No overfitting" / "underfitting" claims compared augmented, mixup-weighted train acc with clean val —
  unsupported. Use Clean Train.
- "GRL working" (low disc acc) — unsupported; needs on/off ablation + clean signer probe.
- Hand-dropout null result ≠ proof of an extraction ceiling — a hypothesis.
- Signer diagnostics "hard signers sign at normal per-frame speed" — that `speed` used wrist xyz, mostly
  the z artifact; recomputed in xy from now on. The duration/no-hand correlations used other columns
  and stand, but remain n=6.
- **All runs up to and including fold0_stretch_handdrop carry the depth artifact and the other bugs.**
  Their relative comparisons (same bugs on both sides) are informative, but the next seeded run with all
  fixes is a new baseline, not directly comparable to the 0.668 / 0.680 fold-0 numbers.

### Second review round (2026-10-08)
Reproduced and fixed: padding-dependent `dom_ratio` (0.998 → 0.969 with 100 pad frames; 0 for no
motion), per-signer summary parser regression (introduced by the Clean Train line), TTA noise breaking
Δ1, batch membership fixed across epochs, mixup on raw clips able to flip dominance, single-source
mixup tails, discriminator accuracy ignoring mixup weights, class weights from the full CSV.
Ablation flags added so the open questions can be **measured**: `--loss ce`, `--mixup-prob`,
`--finger-drop-prob`, `--zero-parts face,pose`, `--no-depth`; GRL off = `--grl-lambda 0`.
Open (needs runs, not code): focal vs CE, GRL on/off + frozen-feature signer probe, mixup / finger
dropout value, face / pose / depth value, presence-heuristic accuracy on sparse clips (raw masks were
not kept in the LMDB).

### Fold 0, all review fixes + stretch, seed 42 (`fold0_fixed`) — new baseline
| | Stretch (pre-fix) | Fixed | Δ |
|---|---|---|---|
| best, pooled | 0.6802 | **0.6771** | −0.3 |
| TTA | 0.6786 | 0.6775 | −0.1 |
| avg ep 71–80 | 0.6743 | 0.6738 | ±0.0 |
| 34503 / 49445 / 62590 (best) | 0.577 / 0.661 / 0.803 | 0.574 / 0.656 / 0.803 | ≤ 0.5 |

Same final accuracy: the fixes sped up early learning (+9 pt at epochs 8–10) but not the endpoint.
0 non-finite batches; 1m54s/epoch.

**Clean train (3,000 un-augmented training clips, eval mode) vs val:**

| epoch | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|---|
| clean train | 0.684 | 0.813 | 0.884 | 0.924 | 0.950 | 0.963 | 0.970 | 0.970 |
| val | 0.550 | 0.612 | 0.647 | 0.662 | 0.669 | 0.671 | 0.670 | 0.675 |
| gap (pt) | 13.4 | 20.1 | 23.7 | 26.2 | 28.1 | 29.2 | 30.0 | 29.5 |

Best checkpoint (epoch 72): clean train **0.971** vs val 0.677 → **29.4 pt gap**. The earlier
"underfitting" reading (augmented train acc ≈ 0.80) was wrong: the model nearly memorises its 18
training signers while val plateaus from ~epoch 50. The bottleneck is signer generalisation — not
capacity, not training length (longer runs would widen the gap). Disc acc 0.142 → 0.109 (chance
0.056): the CLS sheds some signer identity, yet the gap still widens.

Next (budget ≈ 2 runs): `--grl-lambda 0` (does the GRL help at all?) and stronger signer-variation
augmentation (1st place: scale 0.8–1.2, shear 0.15, shift 0.1, rotate 30° vs our 15° / small shift).
Deprioritised: 160-epoch run, `--loss ce`.

Implemented for the next two runs (both off by default, compared against `fold0_fixed`):
- `--supcon-weight 0.2` — cross-signer supervised contrastive loss (same sign, different signer =
  positive; cross-batch memory queue; unmixed second pass). Success = val ≥ +2 pt, clean-train gap
  narrower, gains on the hard signers.
- `--aug-rotate 30 --aug-shear 0.15 --aug-scale 0.2 --aug-affine-prob 0.75` — stronger signer-variation
  spatial augmentation (closer to the 1st-place recipe).
GRL-off deferred: if the contrastive term works, the GRL may be redundant and that ablation becomes
more informative.

### Fold 0 + cross-signer contrastive loss (`fold0_supcon`, `--supcon-weight 0.2`, τ 0.1, queue 4096) — stopped at epoch 37
Same config as `fold0_fixed` (seed 42) plus the contrastive term (BN running stats frozen in its
pass). Behaved as designed: 100% of anchors had a cross-signer positive from epoch 1, Con Loss
8.23 → 3.98 by epoch 37, no non-finite batches, ~2m25s/epoch (1.3×).

| avg | ep 8–12 fixed / supcon | ep 22–27 fixed / supcon | ep 28–37 fixed / supcon |
|---|---|---|---|
| val | 0.552 / 0.538 (−1.3) | 0.633 / 0.638 (+0.5) | 0.649 / 0.647 (−0.2) |
| clean train | — | 0.850 / 0.860 (+1.0) | 0.895 / 0.903 (+0.8) |
| gap (pt) | 11.5 / 13.6 (+2.1) | 21.7 / 22.2 (+0.5) | 24.6 / 25.6 (+1.0) |
| 34503 / 49445 / 62590 | −3.1 / −0.3 / −0.6 | +1.1 / +0.9 / −0.6 | −0.7 / +0.4 / −0.5 |

Best val by epoch 37: 0.660 (fixed: 0.658). **Null at this setting**, stopped by the pre-agreed
epoch-35 rule (val not above fixed, gap not narrower) to fund the spatial-affine run. The
objective is learned — the 18 training signers' executions of each sign are pulled together — but
that alignment is itself a fit to the training signers and does not transfer: clean train rises
slightly faster and the gap is slightly wider. One setting only (w 0.2, τ 0.1, single seed);
other weights untested.

### Fold 0 + stronger spatial affine (`--aug-rotate 30 --aug-shear 0.15 --aug-scale 0.2 --aug-affine-prob 0.75`) — incomplete, no result
Same config and seed as `fold0_fixed` plus the stronger rotation/shear/anisotropic-scale augmentation.
The pod ran out of RunPod credit around epoch 30–40, before the log or checkpoint was retrieved, so
there is **no final or best score**. Last numbers seen (vs `fold0_fixed` at the same epochs):
neutral through epoch 12; epochs 13–16 val +0.4, clean train −1.0, **clean-train gap 1.4 pt
narrower**, hard signer 34503 +1.4. That is the pattern the augmentation was meant to produce, but
four early epochs are within noise (the gap is still small then). **Inconclusive — not a null.**
Worth re-running on the current base (`--arch graph`) if budget allows; it targets the same
signer-variation gap as bone-length augmentation (camera angle / body proportions vs hand
proportions), so the two could be combined after each is measured alone.

### GraphConformer (`--arch graph`) — design (2026-11-09); result in the `fold0_graph` entry below
The old `research/models/st_gcn/` pipeline was not salvageable as a comparison: its skeleton graph
was indexed for a face → LH → pose → RH layout while its data was LH → pose → RH → face (almost every
edge on the wrong joints), the mask downsampling crashed on most odd batch lengths, it used a
random by-sign split (signers shared with val), 2D positions only, and none of the input fixes above.

Instead the graph is a drop-in backbone on this pipeline (same LMDB, folds, augmentation, GRL,
logging). `GraphConformer` subclasses `LandmarkConformer` and replaces **only** the four flat hand
projections (`lh/rh_proj`, `lh/rh_vel_proj`) with:
- a **shared hand graph encoder** (21 joints, skeleton + knuckle-web adjacency plus a learned
  zero-initialised residual adjacency per layer, 3 pre-norm residual layers × 64, joint embeddings,
  flattened per joint — not pooled) over wrist-relative positions, bone vectors and Δ1/Δ2/Δ5;
  the non-dominant slot is x-negated first so both hands share weights with the same chirality;
- the **wrist location** (body-relative position + Δ1/Δ2/Δ5) kept outside the graph;
- per-slot output projections to the same `d_model//2` per hand.
Pose, face, geometry and distance streams, fusion, Conformer blocks and heads are unchanged, so the
comparison with `fold0_fixed` isolates the graph encoder. 6.68 M params (vs 6.61 M; graph 18 k);
~1.1× step time on CPU.

Checked: default `LandmarkConformer` logits bit-identical to before the refactor; padding
(+50 frames), input mutation, mirror canonicalisation all exact (Δlogit 0); mirrored hands reach
the shared encoder identically.

Run: `--arch graph --val-fold 0 --stretch-mode sample --stretch-min 0.5 --stretch-max 2.0
--stretch-prob 0.8 --checkpoint-dir checkpoints/fold0_graph` (fold0_fixed's flags + `--arch graph`).
Success = val ≥ +2 pt over 0.6771 or a narrower clean-train gap; otherwise the per-part
projections already capture what the skeleton prior adds (the 1A "GCN stem" hypothesis).
Either way, compare the per-signer ranking (34503 < 49445 < 62590) across architectures.

On the pod (bf16) the graph costs **~2m58s/epoch vs 1m54s** (1.55×, not the 1.1×
measured on CPU), so a full 80-epoch run is ~4 h. First 10 epochs: val 0.5865 vs 0.550 at epoch 10,
clean-train gap 10.8 vs 13.4 pt — early, but ahead on both.

**Next (implemented, off by default): bone-length augmentation** (`--bone-aug-prob 0.5`,
hand ±0.15, per-bone ±0.1). Hand size and finger proportions are a per-signer trait that shoulder
scaling does not remove; scaling bones along their own direction changes fingertip distances and
absolute joint positions but keeps every joint angle, so hand shape (the sign) is preserved.
Intended run: the `fold0_graph` command + `--bone-aug-prob 0.5`, compared against `fold0_graph`
(one change). Success = clean-train gap narrower with val ≥ `fold0_graph`.

**Speed: `--compile`.** It had never worked: torch 2.9.1 refuses `torch.compile` on Python 3.14 and
`train.py` fell back to eager silently. Upgraded to torch 2.10.0 (same CUDA 12.8 wheels; default
model logits within 3e-7 of 2.9.1 on CPU, so runs stay comparable). On CPU, compiled = eager to
4e-7 and nothing recompiles across lengths 19–128 or the per-epoch GRL λ. GPU speed-up unmeasured:
the bone-length run uses `--compile` and its it/s from epoch 2 on is compared with `fold0_graph`'s
~7.2 it/s (epoch 1 includes compilation). Speed only — no effect on the comparison beyond float
rounding.

**Update (2026-11-10): `--compile` crashes on the GPU.** First backward pass of `fold0_graph_bone`
(torch 2.10, bf16): `RuntimeError: (*bias): last dimension must be contiguous` in
`_scaled_dot_product_efficient_attention_backward`. `nn.MultiheadAttention` turns the key padding
mask into a broadcast (stride-0) bias; eager handles it, but the compiled graph passes it as is to
the CUDA memory-efficient attention backward, which needs a contiguous last dimension. CPU uses a
different SDPA kernel, so the local tests could not catch it. Not pursued: a fix (exclude the
attention call from compilation, or build a contiguous mask) can only be verified on a GPU, for an
unmeasured speed-up. Runs continue without `--compile`. torch stays at 2.10 (needed for compile,
harmless otherwise).

### Fold 0 GraphConformer (`fold0_graph`, `fold0_fixed` flags + `--arch graph`) — stopped at epoch 65, final extrapolated
Same config and seed as `fold0_fixed`; only the hand streams differ. ~2m57s/epoch (1.55×).
Stopped after epoch 65 (val 0.6768; log and `best_phase1.pth` = epoch 56 kept locally) to fund the bone-length run: LR had decayed to 5.9e-5 and val had been on
a plateau since ~epoch 54.

| | `fold0_fixed` | graph | Δ |
|---|---|---|---|
| val ep 10 / 20 | 0.550 / 0.612 | 0.587 / 0.635 | +3.7 / +2.3 |
| val ep 28–35 avg | ~0.645 | 0.660 | ~+1.5 |
| val ep 49–64 avg | ~0.670 | 0.679 | ~+0.9 |
| best (measured) | 0.6771 (ep 72) | **0.6859** (ep 56) | +0.9 |
| clean-train gap ep 50 / 60 | 28.1 / 29.2 pt | 27.4 / 28.9 pt | ~0.5 narrower |
| per-signer at best: 34503 / 49445 / 62590 | 0.574 / 0.656 / 0.803 | 0.583 / 0.664 / 0.813 | +0.9 / +0.8 / +1.0 |

`fold0_fixed` values between its logged epochs are interpolated. **Extrapolated final (not
measured):** `fold0_fixed` gained ~+0.6 pt of best from epoch 64 to its peak at 72 and its
epoch-71–80 average sat ~0.3 below its best; applied here → best ≈ 0.686–0.690, avg ep 71–80
≈ 0.680. No TTA number (the run was stopped before the final evaluation; TTA has matched the
deterministic score within ±0.3 in every run so far).

**Reading.** A small, steady gain (~+1 pt from epoch 35 on, single seed), below the +2 pt success
bar. It is uniform across the three val signers, does not target the hard signer, and the
clean-train gap is essentially unchanged (clean train still → 0.97): the explicit hand graph
models hand shape slightly better but does not reduce overfitting to the 18 training signers —
consistent with every other result pointing at signer variety, not architecture. Kept as the
base for the next runs (bone-length augmentation, then fingerspelling pre-training), which target
the gap directly. Comparisons with it use matching epochs (curves), since its endpoint is
extrapolated.

**What the learned adjacency added** (`best_phase1.pth`, epoch 56; symmetrised zero-initialised
residual `adj_residual` per graph layer, links not in the fixed skeleton). Mean |residual| 0.05–0.06,
max 0.27–0.33 — the same order as the fixed normalised skeleton weights (~0.31), so the layers were
substantially rewired, not left near zero.
- **Layer 0: a PIP–tip link on every finger** (two hops apart): index −0.28, middle −0.25, thumb
  MCP–tip −0.24, ring −0.23, pinky −0.22. A negative link acts as a difference (tip − PIP), i.e. a
  per-finger **curl** measure — learned, not given.
- **Layers 1–2: same-level links between neighbouring fingers** (index–middle and middle–ring PIP,
  index–middle, middle–ring and ring–pinky DIP; ±0.13–0.20) — finger **spread / adjacency**.
- The thumb tip ↔ index tip link anticipated in the design (pinch shapes) did not emerge strongly.
Reading: the residual adjacency extracts explicit hand-shape structure (curl, spread), consistent with
attributing the ~+1 pt to better hand-shape modelling — and it overlaps the hand-crafted geometry
stream (joint-angle cosines), which may now be partly redundant (untested).

**Offline re-evaluation on fold 0** (Mac CPU, fp32, both `best_phase1.pth`, 14,076 val clips, one pass
with both models per batch):

| | logged (pod, bf16) | re-run | TTA |
|---|---|---|---|
| `fold0_fixed` | 0.6771 | 0.6769 | 0.6773 |
| `fold0_graph` | 0.6859 | 0.6861 | **0.6858** |

Reproduces within 2–3 clips. TTA again adds nothing.
- **Paired test:** graph right / fixed wrong 1,030 clips, the reverse 901 → net +129; exact McNemar
  **p = 0.0036**. The two trained models differ significantly on this fold — but the test does not
  include training randomness, so "the architecture is better" still needs a second seed or fold.
  The models disagree on **26 %** of clips: a lot of churn for a net +0.9.
- **Per sign:** 129 up, 92 down, 29 unchanged (mean +1.0, sd 6.6 pt). Largest gains: thankyou
  0.56→0.78, grandma 0.55→0.75, like 0.72→0.91, child 0.43→0.59; largest losses: think 0.86→0.67,
  dryer 0.50→0.33, that 0.69→0.53, car 0.67→0.53.
- **Much of the churn is within near-identical pairs**, resolved one way and broken the other:
  duck↔goose (8 resolved / 14 introduced), wake↔awake (7 / 11), home↔kiss (7 / 6), plus dryer→dry,
  grandpa→dad. Some resolved confusions differ mainly in hand shape (thankyou vs taste, hen vs pen),
  but new ones (think→penny) cut the other way — **only weak support** for "the gain is hand shape".
- Side finding: part of the remaining error is concentrated in near-identical sign pairs —
  possibly a dataset-level ambiguity ceiling (not yet quantified).

**Ensemble `fold0_fixed` + `fold0_graph`** (average of softmax probabilities, equal weights, same
14,076 clips; offline, no retraining):

| | pooled | 34503 | 49445 | 62590 | top-5 |
|---|---|---|---|---|---|
| `fold0_fixed` | 0.6769 | 0.574 | 0.656 | 0.803 | 0.856 |
| `fold0_graph` | 0.6861 | 0.583 | 0.664 | 0.813 | 0.864 |
| **ensemble** | **0.7025** | **0.605** | **0.684** | **0.820** | **0.874** |

**First fold-0 result above 0.70**: +1.6 pt over the graph alone (net +231 clips, McNemar
p = 4e-14), +2.6 over `fold0_fixed`, and the largest gain on the hardest signer (34503 +2.2 over
graph). Averaging logits instead: 0.7041. Equal weights are also the optimum of a weight sweep
(w_graph 0.4 / 0.5 / 0.6 → 0.699 / 0.703 / 0.701), so the number is not inflated by tuning on val.
Cost: 2× inference. **Open question:** whether the gain comes from *architectural* diversity or
just from averaging two independently trained models — a 2-seed ensemble of one architecture would
separate the two (it would also give the missing seed-noise estimate).


### Fold 0 GraphConformer + bone-length augmentation (`fold0_graph_bone`, `--bone-aug-prob 0.5`) — null, stopped at epoch 32
`fold0_graph` + `--bone-aug-prob 0.5` (hand ±0.15, bone ±0.1); same seed and batch order, so the two
curves are tightly coupled and small effects would show.

| epochs | val Δ vs graph | clean-train gap graph → bone | 34503 Δ |
|---|---|---|---|
| 10–14 | +0.3 | 12.5 → 12.1 | +0.8 |
| 15–20 | +0.1 | 16.4 → 16.3 | +0.5 |
| 21–26 | +0.3 | 19.7 → 19.3 | +1.1 |
| 27–32 | −0.2 | 22.1 → 22.2 | +0.1 |

Best by epoch 32: 0.6609 (graph: 0.6665). **Null**: the gap grows at the same rate and the early
hard-signer edge vanished; stopped by the epoch-35 rule to fund fingerspelling pre-training.
Likely reason (untested): the model is already largely invariant to hand proportions — joint-angle
cosines are scale-free and fingertip distances are divided by each hand's own size — so simulated
proportions add no signer variety that matters. Low value as an ensemble member: it tracks
`fold0_graph` epoch by epoch (same seed and order), unlike `fold0_fixed` (26 % disagreement).


### Fold 0 GraphConformer + fingerspelling pre-training (`fold0_graph_fs`) — no gain, stopped at epoch 35
**Pre-training** (graph arch, seed 42, bucketed, augmented): 59,342 phrases / 84 signers, val 7,866 /
10 held-out signers; 5m30s/epoch (est. was ~8 min); CER 0.997 → 0.485 (ep 2) → 0.379 (5) → 0.331 (10)
→ **0.3098** (ep 20, best; ran all 20). Val loss flat from ~ep 12 (mild FS overfit; selection is by CER).

**Fine-tune** = `fold0_graph` flags + `--pretrained-backbone` (the standard recipe: 5 warmup epochs with
the backbone frozen, then 75 joint epochs with **backbone LR = 0.1 × head LR**). Warmup alone (frozen
FS features, new head) reached val 0.26 — real transfer. Then:

| | FS-pretrained | graph, same joint epochs | graph, same epochs |
|---|---|---|---|
| val ep 31–35 avg | **0.650** (best 0.652, ep 34) | 0.661 (ep 26–30) | 0.659 |
| clean-train gap | 16.0 pt | 20.9 pt | 24.0 pt |
| 34503 | 0.538 | 0.559 | 0.559 |
| signer-disc acc | ~0.19 | ~0.12 | ~0.12 |

At matched val (~0.65; graph ep 19/25/31) the FS gap is a little narrower (16.0 vs 17.4–23.9) and
34503 equal (0.538 vs 0.537–0.548); at ep 22 the FS gap had been the wider one. **Reading:** ~1 pt
behind at matched training, no hard-signer gain, features carry more signer identity — no support
for "84 extra FS signers fix signer generalisation" *with this recipe*. **Confound:** the 0.1× backbone
LR weakens both the backbone's adaptation and the GRL's reversed gradient on it (≈10× weaker
signer-invariance pressure), which fits the higher discriminator accuracy. Clean test still open:
re-fine-tune from the same `pretrain_fs/backbone_best.pth` with `--backbone-lr-ratio 1.0`
(fine-tune only, ~4 h). Checkpoint (0.652) kept as a candidate ensemble member — a backbone
pre-trained on other data may err differently from `fold0_fixed` and `fold0_graph`.

**Ensemble check (offline, Mac CPU, same 14,076 fold-0 val clips, equal-weight probability average):**
the FS model reproduces 0.6520 and is the *most different* member — prediction disagreement 32 % with
`fold0_fixed` and 30 % with `fold0_graph` (fixed/graph: 26 %); error overlap (both wrong / either
wrong) 0.58–0.60 vs 0.65 for fixed/graph.

| members | pooled | 34503 | 49445 | 62590 |
|---|---|---|---|---|
| graph + fs | 0.6971 | 0.592 | 0.682 | 0.819 |
| fixed + fs | 0.6920 | 0.583 | 0.682 | 0.811 |
| fixed + graph | 0.7025 | 0.605 | 0.684 | 0.820 |
| **fixed + graph + fs** | **0.7079** | **0.608** | **0.694** | **0.822** |

graph + fs beats graph alone by +1.1 (net +156 clips, McNemar p = 2e-11) although fs is 3.4 pt worse
alone; adding fs to fixed + graph: +0.5 (net +76, p = 6e-5). **Best fold-0 result so far: 0.7079.**
Revised reading: FS pre-training does not make a better single model (with this recipe) but yields a
genuinely different one — useful as an ensemble member. The fs member was stopped at epoch 34 of 80.

### VQ bottleneck (`--vq-codes`) — implemented, not yet run (2026-11-10)
Test (c) of the VQ-VAE plan: does a discrete bottleneck, trained end-to-end with classification,
help signer generalisation? The hypothesis that would make VQ useful here: snapping each frame to a
small codebook is an information bottleneck that may drop signer-specific detail — the 29-pt gap.
Kept from the old `vqvae_seq2seq/`: factorised codebooks per body part, EMA codebook updates,
dead-code reset. Dropped: its data pipeline and the seq2seq/CTC/beam-search decoder.

Design: the fusion input of each valid frame is quantised by four EMA codebooks (dominant hand,
non-dominant hand, body, face; 160/160/160/128 dims) → 4 codes per frame (≈32 bits at 256 codes),
straight-through gradient, commitment β 0.25, per-part code perplexity logged each epoch. Same
initial weights as the matching non-VQ run. Checked: default model unchanged; groups tile the fusion
input exactly; codebooks seeded from data, frozen in eval; eval output is a function of the codes
only (residual 1e-7 differences traced to two near-duplicate seed codes — now seeded from distinct
frames); `train.py` end-to-end on a synthetic fold.

Run: `fold0_graph` flags + `--vq-codes 256` (no pre-training, so it compares with `fold0_graph`).
**Reading:** gap narrower with val holding (≥ 0.6859 − 1) → quantisation helps; then the
reconstruction-trained tokenizer (b), possibly pre-trained on FS without labels, is worth building.
Val clearly lower (> 2 pt) → the bottleneck discards needed information; (b) — frozen tokens learned
for reconstruction — would be worse, so the VQ-VAE line stops there. Code perplexity far below 256
for a part means that codebook collapsed and the test is not fair for it (raise β or lower K).


### VQ run (`fold0_graph_vq`) aborted at epoch 5 — exposed a geometry-stream bug (2026-11-10)
The VQ commitment loss logged 54,745 / 15,220 / 19,133 / 11,837 for epochs 1–4 (expected ~0.02), val
0.160 at epoch 4 vs 0.290 for `fold0_graph`: the β-weighted squared error dominated every clipped
update. Not the quantiser: on real fold-0 clips the fusion features are ≤ 4.9 without augmentation.
**Cause:** `_hand_geometry` divided fingertip distances by the mean MCP distance + 1e-6. Finger
dropout (default p 0.5, in every run since Run 004) collapses dropped fingers onto the wrist; with all
four MCPs collapsed and the thumb intact the divisor is ~1e-6 → features up to **2.6e5 in 0.15 % of
training clips ≈ 9 % of batches**. The fusion LayerNorm hid it in the forward pass, but each such
batch's gradient was dominated by one corrupted clip and then clipped — roughly 1 update in 11
mostly wasted, in all runs alike (relative comparisons stand; absolute numbers may be depressed).
**Fix:** distance features are 0 when the hand scale ≤ 0.02 (detected hands: min 0.036, 0.01st pct
0.077, median 0.29 over 44k fold-0 training frames). Existing checkpoints on fold-0 val: max
|Δlogit| 3e-5, no prediction changes. With the full augmentation set the max fusion feature is 4.97
and the VQ commitment loss ~0.02. Consequence: a VQ run with the fix must be compared with a graph
run *with* the fix (`fold0_graph` had the bug), so the clean order is graph+fix, then VQ+fix.

**Graph + fix (`fold0_graph_fixed`), stopped at epoch 22 to start the VQ run:** indistinguishable from
`fold0_graph` — epochs 16–22 val 0.6353 vs 0.6359 (−0.06), clean-train gap 17.9 vs 17.8 pt, the same
per-epoch dips (same seed and order). The corrupted batches cost nothing measurable (clipping
contained them), so all earlier results stand, and the VQ + fix run is compared with `fold0_graph`
directly. VQ + fix first epochs: commitment 0.0139 / 0.0119 (was 54,745 / 15,220), codes in use
52/50/128/97 → 71/38/126/98 (dom hand / non-dom hand / body / face).

**VQ + fix (`fold0_graph_vq`, `--vq-codes 256`) — negative, stopped at epoch 17:**

| epochs | val graph → VQ | clean-train gap graph → VQ |
|---|---|---|
| 12–14 | 0.614 → 0.456 (−15.8) | 13.0 → 12.4 pt |
| 15–17 | 0.629 → 0.478 (−15.1) | 15.5 → 14.2 pt |

At matched val (~0.48; graph epochs 6–7, gap ~9 pt) the VQ gap is wider (14–15 pt): the bottleneck
discards sign-relevant information without removing signer-specific detail. Codes in use: dom hand
~118/256, body ~127, face ~108, non-dom hand collapsed to 10–25 (little information there — most
signers are one-handed — so not the cause). Commitment loss healthy (0.17). **Conclusion (planned
decision point): the VQ-VAE line stops.** The end-to-end, classification-trained bottleneck is the
most favourable case for quantisation; a frozen reconstruction-trained tokenizer (the old pipeline)
would be expected to do worse. Untested variants (bigger codebooks, several codes per part) could
narrow the loss but not rescue the hypothesis that quantisation reduces signer overfitting.


### Per-signer BatchNorm adaptation (AdaBN), offline on fold 0 (2026-11-10)
For each val signer, every BatchNorm layer's running mean/var (4 layers: the Conformer conv modules)
is replaced by the cumulative batch average over that signer's *unlabelled* clips; the model then
predicts with the adapted statistics. Fixed length-sorted batches so all models see the same order.
**Held-out protocol** (statistics from a random half of each signer's clips, accuracy on the other
half; baseline on the same half):

| | base | AdaBN | Δ | 34503 | 49445 | 62590 |
|---|---|---|---|---|---|---|
| fixed | 0.6774 | 0.6858 | +0.8 | .571→.585 | .657→.665 | .805→.809 |
| graph | 0.6855 | 0.6967 | +1.1 | .581→.597 | .662→.674 | .815→.821 |
| fs | 0.6511 | 0.6609 | +1.0 | .538→.560 | .640→.644 | .776→.780 |
| fixed+graph | 0.6990 | 0.7139 | +1.5 | .599→.623 | .680→.691 | .819→.830 |
| **fixed+graph+fs** | 0.7059 | **0.7210** | **+1.5** | .602→.626 | .696→.707 | .820→.832 |

Transductive (stats and predictions on all clips): graph 0.6861 → 0.6977, 3-model 0.7079 → 0.7210.
Same gains in both protocols → not fitting the scored clips. Every model, every signer improves,
most on the hardest (34503). Stacks with ensembling: **0.7210** best fold-0 result (needs ≥ a batch of
unlabelled clips per signer — fine for a per-user app, not comparable to Kaggle's per-clip scoring).
**Reading:** part of the cross-signer gap is a per-signer shift in feature statistics that simple
renormalisation removes — direct evidence for the signer-shift diagnosis. Next along this line:
per-signer input normalisation (train and test), and test-time adaptation of BN affine parameters
(e.g. entropy minimisation, TENT).

**Follow-up: input normalisation and TENT** (same held-out halves; clips canonicalised first):

| | base | AdaBN | input norm | input norm + AdaBN | TENT |
|---|---|---|---|---|---|
| fixed | 0.6774 | 0.6858 | 0.6511 | 0.6605 | **0.6897** |
| graph | 0.6855 | 0.6967 | 0.6720 | 0.6778 | **0.6990** |
| fixed+graph | 0.6990 | 0.7139 | 0.6840 | 0.6900 | **0.7164** |

- **Per-signer input normalisation hurts (−1.4 to −2.6)**: each position feature mapped to the
  training-population mean/std using the signer's calibration clips (hands on detected frames only,
  missing hands untouched, Δ1 rebuilt). Standardising every coordinate independently moves joints by
  different amounts — distorting hand shape — and removes the signer's typical hand locations, which
  partly carry meaning. Not pursued (a train-and-test version could differ, but the test-time form
  is clearly negative).
- **TENT** (AdaBN statistics, then BN + LayerNorm affine params minimise prediction entropy on the
  calibration half, 2 passes, Adam 1e-4): small, consistent extra over AdaBN — +0.4 / +0.2 / +0.25.
  The gain is in re-estimating normalisation statistics, not in input rescaling or affine tuning.


### 1st-place anchor on fold 0 — notebooks ready (2026-11-10)
Question: is our ~0.69 single-model fold-0 accuracy a property of these signers, or of our recipe/
pipeline? Run the published 1st-place solution on the same held-out signers. Kaggle's 0.89 is a
4-seed ensemble trained on all 21 signers and scored on a hidden test set (signer overlap unknown),
so it does not answer this.
- `Build_ISLR_TFRecords_SignerFolds.ipynb` (CPU): 1st-place record format, our 7 signer folds;
  ~18.5 GiB (3.58 M frames × ~5.5 KB; floats barely compress).
- `Train_1st_Place_Fold0.ipynb` (GPU): original cells verbatim except environment pins, data paths,
  GPU strategy (batch 64 × GPUs, lr 5e-4 × GPUs — the original's scaling of 64 × 8 TPU replicas),
  mixed_float16 on non-bf16 GPUs, `EPOCHS` + 11 h time limit, and a per-signer evaluation cell.
- Local smoke test (TF 2.15, CPU, 560 synthetic records, AWP + late dropout forced on from epoch 0):
  trains, holds out fold 0, saves best/last weights and `fold0_val_probs.npz`. Caught on the way:
  TF 2.15 needs numpy < 2, which breaks current pandas/pyarrow → pinned in the install cell.
**Reading:** ≥ 0.75 on fold 0 → our recipe/pipeline is the gap; ≈ 0.69 → the signers are; either way
its predictions are an ensemble candidate. Deviation to note: 300 epochs may not fit the 12 h
session on a T4 — the first epoch prints how many fit.

### Error review + fold-0 val breakdown (`fold0_graph`, 2026-11-11)
**Tool:** `python -m cnn_transformer.error_review` (commit `848d687`) — 100 `fold0_graph` val errors
(34/33/33 across signers 34503/49445/62590; 49 from the 20 most frequent confusion pairs), each shown as
an animated landmark clip next to two unlabelled reference sets (2 training clips each, distinct
signers) of the true and the predicted sign in random order; undetected-hand frames drawn faded.
Reviewed by a **non-signer** (no ASL knowledge), answers before reveal; 12 flagged low confidence,
1 seen with labels. Answers: `results/error_review/fold0_graph_errors_answers.tsv`.

| answer | n | |
|---|---|---|
| picked the true sign ("information present") | 35 | |
| picked the predicted sign | 39 | |
| A and B look the same | 15 | lips/mouth 7/7, say/chin 3, say/mouth 2 — mouth-region pointing signs |
| can't judge | 10 | 5 of them are **6-frame** clips; median 83 % of frames with no hand |
| neither | 1 | |

**The A/B picks are at chance:** true sign in 35/74 = 47 % (chance 50 %, exact binomial p = 0.73);
frequent pairs 22/36 = 61 % (p = 0.24); other errors 13/38 = 34 % (p = 0.07, leaning towards the model's
guess). For a non-signer the landmark animations of these errors do not separate the two signs, so the
review **does not** establish a learner-fixable share. It also cannot show the information is absent —
fine handshape/movement differences may need a fluent signer (the kit's signer mode; not done).
Patterns: consistent mismatches judged as the predicted sign, made by all three models —
glasswindow→tooth 3/3, yes→carrot 2/2, cut→scissors 2/2 (likely signer-specific / regional variants);
pairs where the reviewer beat chance — jump/stairs 3/3, cat/kitty 4/5, awake/wake 3/4, dry/dryer 2/2,
sleep/sleepy 2/2 (small handshape/movement details — relevant to the finger-dropout question).
59/100 errors are shared by all three models (fixed, graph, fs).

**Caveat — weak references (audit, same day):** the kit drew references at random with no tracking
check: 69/206 reference clips had a hand in < 50 % of frames, 15 had ≤ 2 detected frames, and in
58/100 items at least one true-sign reference was mostly hand-less. Restricted to items whose four
references all have ≥ 3 detected frames the A/B result is still at chance (23/50 = 46 %, p = 0.67);
with ≥ 30 % hand frames 11/26 (p = 0.56); with ≥ 50 % only 9 picks. So "at chance" is not refuted, but the
review is underpowered and partly confounded — treat it as inconclusive, not as evidence of ambiguous
landmarks. Fixed in `error_review.py` (references need a hand in ≥ 50 % of frames and ≥ 8 detected
frames; detection kept through display downsampling; kit id carries an item fingerprint so a
regenerated kit never inherits answers; `--n-heads` explicit). Regenerated kit
`results/error_review/fold0_graph_errors_v2.html` (same 100 errors, 0/206 weak references, median hand
fraction 0.67 → 0.83) — not yet reviewed.

**Whole fold-0 val set** (`fold0_graph`, 14,076 clips, acc 0.6861; top-2 0.7905, top-3 0.8296, top-5 0.8639;
predictions cached in `results/error_review/fold0_graph_val_probs.npz`):

| clip length | clips | acc | % of errors | | no-hand frames | clips | acc | % of errors |
|---|---|---|---|---|---|---|---|---|
| ≤ 6 | 7.6 % | **0.402** | 14.4 % | | 0 | 18.6 % | 0.787 | 12.6 % |
| 7–10 | 5.2 % | 0.634 | 6.0 % | | ≤ 25 % | 26.2 % | 0.759 | 20.1 % |
| 11–20 | 24.7 % | 0.763 | 18.6 % | | 25–50 % | 24.6 % | 0.703 | 23.2 % |
| 21–40 | 20.6 % | 0.739 | 17.1 % | | 50–75 % | 21.0 % | 0.607 | 26.2 % |
| 41–80 | 22.6 % | 0.683 | 22.8 % | | > 75 % | 9.7 % | **0.426** | 17.8 % |
| > 80 | 19.4 % | 0.660 | 21.0 % | | | | | |

- **Degraded clips** (≤ 6 frames or > 75 % no-hand frames): 14.9 % of val, acc 0.443 (not unrecognisable —
  partly recoverable); the other 85.1 % reach 0.7286. Even perfect accuracy on them would give 0.769, so
  they cost at most ~8 pts and realistically a few. Training has 13.7 % clips ≤ 6 frames too.
- **Tracking explains only part of the signer spread:** degraded-clip share 34503 / 49445 / 62590 =
  high / high / low (> 75 % no-hand: 13.5 / 12.7 / 2.8 %; median no-hand 0.35 / 0.48 / 0.08), but on
  non-degraded clips accuracy is still 0.632 / 0.706 / 0.834 (overall 0.583 / 0.664 / 0.813): the 23-pt
  spread shrinks only to 20.
- **Near-synonyms are a small ceiling:** merging the review's look-alike group {lips, mouth, say, chin}
  +0.4 pt; 13 a-priori near-synonym groups (wake/awake, sleep/sleepy/nap, cat/kitty, dog/puppy,
  dry/dryer, hear/listen, see/look, pen/pencil, cut/scissors, tooth/toothbrush, bed/bedroom,
  duck/goose, lips/mouth) +2.2 pt; merging the top-N confused pairs chosen on val (optimistic):
  +1.3 / +2.2 / +3.4 / +5.7 pt for N = 5 / 10 / 20 / 50. Errors are diffuse: 2,311 distinct
  (true, pred) pairs; the top 20 cover only 10.7 % of errors.

**Reading:** no single data defect explains the gap. Degraded clips (a few pts) and near-synonym labels
(~2 pts) are real but bounded; most errors are spread over thousands of sign pairs and remain
signer-dependent on clean clips — consistent with the signer-generalisation diagnosis. Next: the
recipe ablations (finger dropout off first — the resolvable errors hinge on handshape detail; then GRL,
mixup, depth), the 1st-place anchor, and optionally a fluent signer on the 54 unresolved items.
