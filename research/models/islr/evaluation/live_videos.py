"""Score phrase checkpoints on your own recordings (camera acceptance, before the app does it live).

Input: landmarks written by `scripts/extract_live.py` for videos laid out as
<videos>/<condition>/<block>[_<take>].<ext>, one video per block of `live_script.json` (the signs of
the block in order, hands out of frame between signs). Each video goes through the browser's path
(`preprocess.json`), not the builders': capture timestamps resampled to 30 fps
(`browser_fixtures.resample_plan`), contract v4 `prepare` over the whole video (hold-fill before any
cut), x and z scaled by (W / H) / 0.75 (the browser's frame-shape rule, not the builders' per-clip
shoulder correction), then cut into signs at hand-free gaps of at least --gap frames. A video whose
segment count differs from its block is not scored: its `segments.json` is written for a contact sheet
(`extract_live.py --sheet`) and `--assign` can then name each segment (null = discard).

Per segment and checkpoint: greedy CTC decode (exact = the one sign; phrases: gloss WER), pooled
top-1 / top-5 over the 250 signs. Segments shorter than 8 frames are not run (the browser model's
minimum) and count as wrong. Paired McNemar between the first two checkpoints on single-sign
segments. One signer, a learner copying reference videos: this tests the pipeline and recording
conditions, not accuracy for Deaf signers.

  PYTHONPATH=research/models uv run python -m islr.evaluation.live_videos \
      --landmarks data/live-recordings/landmarks \
      --checkpoint results/phrases/frozen-phrases05/phrases05/best_phrases.pth \
                   results/phrases/frozen-phrases07/phrases07/best_phrases.pth \
      --out results/evaluation/live/<date>
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.data import contract
from ..core.data.vocab import canonical
from ..export.browser_fixtures import apply_plan, frames_to_rows, raw_frames, resample_plan
from ..export.export_onnx import load_ctc_model
from .aspect_compare import scaled
from .cross_dataset import mcnemar

SCRIPT = Path(__file__).with_name("live_script.json")
MIN_T = 8  # the browser model's shortest input


def resampled_rows(df: pd.DataFrame, ts: list) -> tuple[pd.DataFrame, list]:
    """Raw rows on the 30 fps grid of the capture timestamps, and the raw frame of each tick."""
    plan = resample_plan(ts)
    if all(w == 0.0 for _, _, w in plan):  # nearest-frame mode: a selection, done on the rows directly
        src = [i0 for i0, _, _ in plan]
        parts = [df[df["frame"] == i].assign(frame=k) for k, i in enumerate(src)]
        return pd.concat(parts, ignore_index=True), src
    return frames_to_rows(apply_plan(raw_frames(df), plan)), [i0 if w < 0.5 else i1 for i0, i1, w in plan]


def hand_runs(hand: np.ndarray, ticks: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """[start, end) runs of hand frames (indices into the kept frames), joined across hand-free gaps
    shorter than gap ticks of the 30 fps grid (dropped pose-less frames count as hand-free)."""
    idx = np.flatnonzero(hand)
    if not len(idx):
        return []
    runs, s = [], idx[0]
    for prev, i in zip(idx[:-1], idx[1:]):
        if ticks[i] - ticks[prev] > gap:
            runs.append((s, prev + 1))
            s = i
    runs.append((s, idx[-1] + 1))
    return runs


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--landmarks", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--script", type=Path, default=SCRIPT)
    p.add_argument("--gap", type=int, default=15, help="hand-free frames (30 fps) that end a sign")
    p.add_argument("--min-run", type=int, default=6, help="shorter hand runs are flickers, not signs")
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--assign", type=Path, help='JSON {"<condition>/<video>": [label | null, ...]} per segment')
    a = p.parse_args()
    blocks = json.loads(a.script.read_text())["blocks"]
    assign = json.loads(a.assign.read_text()) if a.assign else {}
    sign2idx = canonical()
    lab = sorted(sign2idx, key=sign2idx.get)
    blank = len(sign2idx)
    models = {}
    for ck in a.checkpoint:
        name = ck.parent.name if ck.parent.name not in models else f"{ck.parent.parent.name}/{ck.parent.name}"
        models[name] = load_ctc_model(ck, json.loads(ck.with_name("phrases_meta.json").read_text()), blank).eval()

    @torch.no_grad()
    def decode(model, x: torch.Tensor) -> dict:
        out = model(x[None].float(), torch.ones(1, len(x), dtype=torch.bool))[0]
        hyp, prev = [], None
        for c in out.argmax(-1).tolist():
            if c != prev and c != blank:
                hyp.append(lab[c])
            prev = c
        pooled = out.float().log_softmax(-1)[:, :blank].max(0).values
        return {"hyp": hyp, "top5": [lab[i] for i in pooled.topk(5).indices.tolist()]}

    a.out.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for meta_path in sorted(a.landmarks.rglob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        cond = meta_path.parent.relative_to(a.landmarks).as_posix()
        stem = meta_path.name[: -len(".meta.json")]
        key = f"{cond}/{stem}"
        block = re.match(r"[A-Za-z]+", stem).group(0).upper()
        if block not in blocks:
            problems.append(f"{key}: no block {block!r} in {a.script.name}")
            continue
        df = pd.read_parquet(meta_path.with_name(stem + ".parquet"))
        rs, src = resampled_rows(df, meta["timestamps_ms"])
        W, H = meta["width"], meta["height"]
        clip = contract.prepare(rs, aspect=W / H)
        coords = scaled(clip["coords"], (W / H) / 0.75)
        presence = clip["presence"]
        kept = clip["frames"]  # tick ids after dropping pose-less frames
        runs = [r for r in hand_runs(presence.any(1), np.asarray(kept), a.gap) if r[1] - r[0] >= a.min_run]
        labels = assign.get(key, blocks[block])
        segs = [{"index": i, "video": key, "start": int(s), "end": int(e), "frames": int(e - s),
                 "mid_raw_frame": int(src[int(kept[(s + e) // 2])]),
                 "label": (labels[i] if i < len(labels) else None)} for i, (s, e) in enumerate(runs)]
        (a.out / "segments").mkdir(exist_ok=True)
        (a.out / "segments" / f"{key.replace('/', '__')}.json").write_text(json.dumps(
            {"videos": {key: meta["video"]}, "mirrored": {key: meta["mirrored_in_file"]}, "segments": segs,
             "expected": blocks[block]}, indent=1))
        info = (f"{key}: {meta['width']}x{meta['height']}, {meta.get('fps_measured') or 0:.1f} fps, "
                f"{rs["frame"].nunique()} frames at 30 fps ({clip['stats']['frames_no_pose']} without pose), "
                f"{len(runs)} segments for {len(labels)} labels")
        if len(runs) != len(labels):
            problems.append(info + " — not scored; check the contact sheet, then --assign")
            continue
        print(info)
        for sg in segs:
            if sg["label"] is None:
                continue
            truth = sg["label"] if isinstance(sg["label"], list) else [sg["label"]]
            sl = slice(sg["start"], sg["end"])
            x = contract.clip_input(torch.from_numpy(coords[sl]), torch.from_numpy(presence[sl]).float(),
                                    a.max_frames, trim=True)
            base = {"condition": cond, "video": stem, "block": block, "segment": sg["index"],
                    "label": " ".join(truth), "n_signs": len(truth), "frames": len(x),
                    "two_hands": bool(presence[sl].all(1).any()), "hand_frac": float(presence[sl].any(1).mean())}
            for name, model in models.items():
                r = decode(model, x) if len(x) >= MIN_T else {"hyp": [], "top5": []}
                rows.append({**base, "model": name, "decoded": " ".join(r["hyp"]), "exact": r["hyp"] == truth,
                             "errors": edit_distance(truth, r["hyp"]), "pooled_top1": r["top5"][:1] == truth[:1] and len(truth) == 1,
                             "top5": len(truth) == 1 and truth[0] in r["top5"], "top5_list": " ".join(r["top5"])})
    for m in problems:
        print("!!", m)
    if not rows:
        raise SystemExit("nothing scored")
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "per_segment.csv", index=False)
    single, phr = res[res.n_signs == 1], res[res.n_signs > 1]
    summary = {"checkpoints": [str(c) for c in a.checkpoint], "problems": problems,
               "single_sign": {m: {"n": len(g), "exact": float(g.exact.mean()), "top5": float(g.top5.mean()),
                                   "by_condition": g.groupby("condition").exact.mean().round(4).to_dict(),
                                   "by_block": g.groupby("block").exact.mean().round(4).to_dict()}
                               for m, g in single.groupby("model")},
               "phrases": {m: {"n": len(g), "wer": float(g.errors.sum() / g.n_signs.sum()), "exact": float(g.exact.mean())}
                           for m, g in phr.groupby("model")}}
    names = list(models)
    if len(names) > 1 and len(single):
        piv = single.pivot_table(index=["condition", "video", "segment"], columns="model", values="exact", aggfunc="first")
        summary["mcnemar_single_sign"] = {f"{names[0]} → {names[1]}": mcnemar(piv[names[0]].astype(bool), piv[names[1]].astype(bool))}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print("\nsingle-sign exact decode by condition:")
    print(single.pivot_table(index="condition", columns="model", values="exact", aggfunc="mean").round(3).to_string())
    print("\nper sign (exact, all conditions):")
    print(single.pivot_table(index="label", columns="model", values="exact", aggfunc="mean").round(2).to_string())
    if len(phr):
        print("\nphrases:")
        print(phr[["condition", "label", "model", "decoded", "errors"]].to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoints"}, indent=1))


if __name__ == "__main__":
    main()
