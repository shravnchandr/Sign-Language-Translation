# Experiment log

Every run, its config and its result, split by topic. Goal: an accessible ASL sign recogniser
(250 signs from MediaPipe landmarks, the Google ASL Signs vocabulary) that works for signers it
has never seen — in the browser, from a webcam.

## Where things stand

| Approach | Code | Best result so far | Log |
|---|---|---|---|
| 0. VQ-VAE → translator | `research/models/vqvae_seq2seq/` | dormant | — |
| A. Isolated classifier | `islr/isolated/` | fold 0: 0.686 single (GraphConformer), 0.708 ensemble | [01](01_isolated_runs.md), [02](02_isolated_folds.md) |
| B. Fingerspelling pre-training | `islr/pretrain/` | no single-model gain; useful ensemble member | [02](02_isolated_folds.md) |
| **C. Continuous phrases (CTC)** — product model | `islr/phrases/` | phrases04: WLASL unseen signers 80.8 % single-sign, ASL Signs held-out 65 % top-1 / 81 % top-3; ASL Citizen (independent, research only) phrases04 68.7 %, phrases05 72.4 % | [03](03_phrases_and_browser.md) |

Data findings (one-hand training data, WLASL / MS-ASL / ASL Citizen) are in [04](04_data_cross_dataset.md);
open ideas in [ideas](ideas.md); Kaggle leaderboard context and the 1st-place recipe in
[kaggle_reference](kaggle_reference.md).

## Files

| File | Contents |
|---|---|
| [01_isolated_runs.md](01_isolated_runs.md) | Runs 001–005 on the default 3-signer split and the fixes between them |
| [02_isolated_folds.md](02_isolated_folds.md) | Signer folds, review fixes, GraphConformer, bone aug, FS pre-training, VQ, AdaBN, ensembles, 1st-place anchor, error review |
| [03_phrases_and_browser.md](03_phrases_and_browser.md) | phrases01–05, browser export, Kaggle GPU runner, live-test diagnoses, per-sign reliability |
| [04_data_cross_dataset.md](04_data_cross_dataset.md) | Training-data limits and cross-dataset tests (WLASL, MS-ASL, ASL Citizen) |
| [ideas.md](ideas.md) | Pending ideas, by priority |
| [kaggle_reference.md](kaggle_reference.md) | Leaderboard context, 1st-place recipe |

## Adding an entry

Append to the file for its topic (newest last) as `### <title> (YYYY-MM-DD)`: what changed and why,
the exact command or config, the result with the comparison it needs (paired / same split), and
what it does and does not show. Update the table above when a best result changes.

## Repository history

Until 2026-11-12 this was one file, `docs/EXPERIMENT_LOG.md` (split here unchanged). The same day
the code moved from `research/models/cnn_transformer/` to `research/models/islr/` (by approach),
shell pipelines and Kaggle runners to `scripts/`, notebooks to `notebooks/<approach>/`, and the
local ASL Signs LMDB `IS_LMDB_Database.mdb` to `data/asl-is-lmdb/is.lmdb.mdb`. Paths in older
entries are as they were at the time.
