# Ghost-rule reference labels

Visual reference judgements (not ground truth) of two-hand frames, made on the blind tiles of
`ghost_review --sample` / `--frames` (shuffled numbers, no distance or rule decision shown), before
any key was opened. `copy` = both skeletons track the same physical hand (however many hands are
visible); `two` = different physical hands; `unclear` = cannot tell (blur, occlusion, resolution, or
a skeleton on no hand at all).

| File | Kaggle run (key: `label_key.csv`) |
|---|---|
| `wlasl_{calibration,holdout}_claude.csv` | `islr-label-wlasl3` (seed 0, 12 per band per set) |
| `popsign_{calibration,holdout}_claude.csv` | `islr-label-popsign2` (PopSign shards 0–9) |
| `lowiou_*` | `islr-lowiou-wlasl`, `islr-lowiou-popsign` (IoU < 0.1 / < 0.2, distance < 0.3) |

Committed before the calibration analysis. An independent labelling of the same tiles is welcome:
the blank `labels_<set>.csv` are in each run's output.
