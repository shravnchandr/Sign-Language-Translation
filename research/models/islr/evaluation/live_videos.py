"""Score phrase checkpoints on your own recordings (camera acceptance, before the app does it live).

Two inputs, both scored the way the app's phrase mode (ASL-App `src/utils/signPhrase.ts`,
`PhraseSegmenter` + `handleSegment`) treats a live stream:

  --landmarks  landmarks written by `scripts/extract_live.py` (frames centre-cropped to 3:4 as the
               app's `portraitCrop` does before detection) for videos <videos>/<condition>/<block>[_<take>].<ext>,
               one per block of `live_script.json` (the block's signs in order, hands out of frame for
               at least 1 s between signs). The stream is cut into utterances exactly as the app does
               (a segment starts at the first frame with a hand, with up to 1 s of pre-roll, and ends
               once no hand has been seen for 1 s; the frames after the last hand are the next segment's
               lead-in; 1,800-frame cap); each segment is then prepared on its own: its timestamps
               resampled to 30 fps, contract v4 `prepare` (hold-fill inside the segment only), hand
               span, subsample, Δ1. A video whose segment count differs from its block is not scored:
               its `segments.json` is written for a contact sheet (`extract_live.py --sheet`) and
               `--assign` can then name each segment (null = discard).
  --recordings segments exported by the app itself (`toRecording`: {label, frames, timestamps_ms}),
               <dir>/<condition>/*.json — the app's own landmarker, so the only input that tests it.

What --landmarks does not reproduce: the app detects with MediaPipe Tasks (Pose / Hand / Face
Landmarker) and assigns hands to slots by pose-wrist distance; extract_live uses Holistic 0.10.14
(the training data's tracker, whose slots are the body side already). Those videos test signing and
recording conditions with training-like landmarks; the app's tracker is tested by --recordings.

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
import hashlib
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
from .cross_dataset import mcnemar

SCRIPT = Path(__file__).with_name("live_script.json")
MIN_T = 8  # the browser model's shortest input
# ASL-App signPhrase.ts PhraseSegmenter
SEGMENT_END_GAP_MS = 1000
SEGMENT_PRE_ROLL_MS = 1000
SEGMENT_MAX_FRAMES = 1800


def app_segments(posed: np.ndarray, hand: np.ndarray, ts: list) -> list[list[int]]:
    """PhraseSegmenter over a recorded stream: the raw frame indices of each utterance. A frame
    without a pose is the app's NO_POSE frame (no hand either); the end of the recording flushes."""
    buf, last, out = [], -1, []
    for i, t in enumerate(ts):
        buf.append(i)
        if posed[i] and hand[i]:
            last = len(buf) - 1
        if last < 0:  # waiting for a hand: only the last second is kept
            stale = next(k for k, j in enumerate(buf) if ts[j] >= t - SEGMENT_PRE_ROLL_MS)
            buf = buf[stale:]
            continue
        if t - ts[buf[last]] >= SEGMENT_END_GAP_MS or len(buf) >= SEGMENT_MAX_FRAMES:
            out.append(buf)
            buf, last = buf[last + 1:], -1  # the frames after the last hand: the next lead-in
    if last >= 0:
        out.append(buf)
    return out


def segment_input(rows: pd.DataFrame, times: list, aspect: float, max_frames: int) -> tuple[torch.Tensor, dict]:
    """handleSegment: one utterance's raw rows (frame = position, 0..n-1) and capture times →
    model input. Times relative to the first; a repeated timestamp is dropped (resampleFrames)."""
    t = np.asarray(times, float) - times[0]
    keep = [0] + [i for i in range(1, len(t)) if t[i] > t[: i].max()]
    rows = rows[rows["frame"].isin(keep)].assign(frame=lambda d: d["frame"].map({k: i for i, k in enumerate(keep)}))
    plan = resample_plan(t[keep].tolist())
    if all(w == 0.0 for _, _, w in plan):  # nearest-frame mode: a selection, done on the rows directly
        rs = pd.concat([rows[rows["frame"] == i0].assign(frame=k) for k, (i0, _, _) in enumerate(plan)], ignore_index=True)
    else:
        rs = frames_to_rows(apply_plan(raw_frames(rows), plan))
    clip = contract.prepare(rs, aspect=aspect)
    coords, presence = torch.from_numpy(clip["coords"]), torch.from_numpy(clip["presence"]).float()
    x = contract.clip_input(coords, presence, max_frames, trim=True)
    s = contract.segment(presence)
    return x, {"frames_30fps": int(rs["frame"].nunique()), "no_pose": clip["stats"]["frames_no_pose"],
               "two_hands": bool((presence[s] > 0).all(1).any()), "hand_frac": float((presence[s] > 0).any(1).float().mean())}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--landmarks", type=Path)
    p.add_argument("--recordings", type=Path, help="app-exported segments, <dir>/<condition>/*.json")
    p.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--script", type=Path, default=SCRIPT)
    p.add_argument("--max-frames", type=int, default=384)
    p.add_argument("--assign", type=Path, help='JSON {"<condition>/<video>": [label | null, ...]} per segment')
    a = p.parse_args()
    if not (a.landmarks or a.recordings):
        p.error("give --landmarks and / or --recordings")
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

    rows, problems = [], []

    def score(base: dict, truth: list, x: torch.Tensor) -> None:
        for name, model in models.items():
            r = decode(model, x) if len(x) >= MIN_T else {"hyp": [], "top5": []}
            rows.append({**base, "label": " ".join(truth), "n_signs": len(truth), "frames": len(x), "model": name,
                         "decoded": " ".join(r["hyp"]), "exact": r["hyp"] == truth, "errors": edit_distance(truth, r["hyp"]),
                         "pooled_top1": len(truth) == 1 and r["top5"][:1] == truth,
                         "top5": len(truth) == 1 and truth[0] in r["top5"], "top5_list": " ".join(r["top5"])})

    a.out.mkdir(parents=True, exist_ok=True)
    for meta_path in sorted(a.landmarks.rglob("*.meta.json")) if a.landmarks else []:
        meta = json.loads(meta_path.read_text())
        cond = meta_path.parent.relative_to(a.landmarks).as_posix()
        stem = meta_path.name[: -len(".meta.json")]
        key = f"{cond}/{stem}"
        block = re.match(r"[A-Za-z]+", stem).group(0).upper()
        if block not in blocks:
            problems.append(f"{key}: no block {block!r} in {a.script.name}")
            continue
        if not Path(meta["video"]).exists() or sha256(meta["video"]) != meta.get("source_sha256"):
            problems.append(f"{key}: the video is missing or changed since extraction — re-run extract_live.py")
            continue
        df = pd.read_parquet(meta_path.with_name(stem + ".parquet"))
        ts = meta["timestamps_ms"]
        n = len(ts)
        first = df[df["landmark_index"] == 0].pivot_table(index="frame", columns="type", values="x", aggfunc="first").reindex(range(n))
        posed = first["pose"].notna().to_numpy()
        hand = (first.get("left_hand", pd.Series(np.nan, first.index)).notna()
                | first.get("right_hand", pd.Series(np.nan, first.index)).notna()).to_numpy()
        segs_idx = app_segments(posed, hand, ts)
        # segments whose hand span is under MIN_T frames at 30 fps (flickers; the app shows them as
        # "too short to read") are reported, not matched to labels
        span = [round((ts[h[-1]] - ts[h[0]]) * 30 / 1000) + 1 for h in ([j for j in i if posed[j] and hand[j]] for i in segs_idx)]
        short = [k for k, n_ in enumerate(span) if n_ < MIN_T]
        segs_idx = [i for k, i in enumerate(segs_idx) if k not in short]
        labels = assign.get(key, blocks[block])
        segs = []
        for i, idx in enumerate(segs_idx):
            hs = [j for j in idx if posed[j] and hand[j]]
            segs.append({"index": i, "video": key, "raw_frames": [idx[0], idx[-1] + 1], "hand_frames": [hs[0], hs[-1] + 1],
                         "mid_raw_frame": hs[len(hs) // 2], "label": labels[i] if i < len(labels) else None})
        (a.out / "segments").mkdir(exist_ok=True)
        (a.out / "segments" / f"{key.replace('/', '__')}.json").write_text(json.dumps(
            {"videos": {key: meta["video"]}, "mirrored": {key: meta["mirrored_in_file"]}, "segments": segs,
             "expected": blocks[block]}, indent=1))
        info = (f"{key}: crop {meta['width']}x{meta['height']} of {meta.get('source_width')}x{meta.get('source_height')}, "
                f"{meta.get('fps_measured') or 0:.1f} fps, {n} frames ({int((~posed).sum())} without pose), "
                f"{len(segs_idx)} segments for {len(labels)} labels" + (f" ({len(short)} under {MIN_T} frames ignored)" if short else ""))
        if len(segs_idx) != len(labels):
            problems.append(info + " — not scored; check the contact sheet, then --assign")
            continue
        print(info)
        aspect = meta["width"] / meta["height"]
        for sg, idx in zip(segs, segs_idx):
            if sg["label"] is None:
                continue
            truth = sg["label"] if isinstance(sg["label"], list) else [sg["label"]]
            pos = {j: k for k, j in enumerate(idx)}
            seg_rows = df[df["frame"].isin(idx)].assign(frame=lambda d: d["frame"].map(pos))
            no_pose = [pos[j] for j in idx if not posed[j]]  # the app's NO_POSE frame: nothing detected
            seg_rows.loc[seg_rows["frame"].isin(no_pose), ["x", "y", "z"]] = np.nan
            x, info_s = segment_input(seg_rows, [ts[j] for j in idx], aspect, a.max_frames)
            score({"source": "video", "condition": cond, "video": stem, "block": block, "segment": sg["index"], **info_s}, truth, x)

    for rec in sorted(a.recordings.rglob("*.json")) if a.recordings else []:
        r = json.loads(rec.read_text())
        cond = rec.parent.relative_to(a.recordings).as_posix() or "app"
        truth = r["label"].split()
        if "timestamps_ms" not in r:
            problems.append(f"{rec}: no timestamps_ms — not scored (the app resamples by capture time)")
            continue
        x, info_s = segment_input(frames_to_rows(r["frames"]), r["timestamps_ms"], 0.75, a.max_frames)
        score({"source": "app", "condition": cond, "video": rec.stem, "block": None, "segment": 0, **info_s}, truth, x)

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
        piv = single.pivot_table(index=["source", "condition", "video", "segment"], columns="model", values="exact", aggfunc="first")
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
