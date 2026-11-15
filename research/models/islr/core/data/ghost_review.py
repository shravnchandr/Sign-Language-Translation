"""Visual review of the ghost-hand rule: raw hand slots and pose wrists drawn on the video frames.

The rule (contract.resolve_ghosts) rejects one hand slot when the two hands' 2D boxes overlap.
Statistics cannot say whether a rejected hand was a duplicate fit of one physical hand (a ghost)
or a real second hand touching the first (PRAY, HELP). This draws, for the chosen clips, frames
where the rule fires: left_hand slot red, right_hand slot blue (a rejected slot dashed), pose
wrists 15 / 16 as filled circles in the slot colours, with IoU and statuses. It also logs per
frame the evidence the rule does not use: how far each pose wrist is from the nearer of the two
hand wrists, in shoulder widths — a real second hand has its own pose wrist next to it.

Writes to --out: `<video id>.png` contact sheets and `ghost_review.csv` (one row per frame with
both hands detected).

  python -m islr.core.data.ghost_review --input-root {INPUT} --video-root {INPUT} \
    --keyword wlasl --ids 44132,01627 --out {OUT}
"""

import argparse
import glob
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import contract as K
from .audit_raw import shard_clips

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
RED, BLUE = (40, 40, 230), (230, 120, 30)  # BGR: left_hand slot, right_hand slot
NAMES = {K.ABSENT: "-", K.KEPT: "kept", K.GHOST: "GHOST", K.AMBIGUOUS: "ambig"}


def frame_evidence(raw: dict, status: np.ndarray, aspect: float) -> pd.DataFrame:
    """Per frame with both hands detected: IoU, statuses, each slot's wrist distance to both pose
    wrists, each pose wrist's distance to the nearer hand wrist, the two hand wrists' and the two pose
    wrists' distance (shoulder widths, aspect-scaled) and the slots' mean landmark distance / hand size."""
    lh, rh, pose = raw["left_hand"], raw["right_hand"], raw["pose"]
    scale = np.array([aspect, 1.0])
    sw = np.nanmedian(np.linalg.norm((pose[:, 11, :2] - pose[:, 12, :2]) * scale, axis=-1))
    rows = []
    for t in np.nonzero(~np.isnan(lh[:, 0, 0]) & ~np.isnan(rh[:, 0, 0]))[0]:
        hw = [lh[t, 0, :2], rh[t, 0, :2]]
        near = [min(np.linalg.norm((pose[t, w, :2] - h) * scale) for h in hw) / sw for w in (15, 16)]
        d = {f"{s}_to_pose{w}": float(np.linalg.norm((h[t, 0, :2] - pose[t, w, :2]) * scale) / sw)
             for s, h in (("lh", lh), ("rh", rh)) for w in (15, 16)}
        sizes = [np.linalg.norm((h[t, 9, :2] - h[t, 0, :2]) * scale) for h in (lh, rh)]  # wrist → middle MCP, projected
        boxes = [np.linalg.norm((K._box(h[t])[2:] - K._box(h[t])[:2]) * scale) for h in (lh, rh)]  # box diagonal
        size = np.mean(sizes)
        d.update({"lh_size": sizes[0] / sw, "rh_size": sizes[1] / sw, "lh_box": boxes[0] / sw, "rh_box": boxes[1] / sw})
        rows.append({"frame": int(t), "iou": K._iou(K._box(lh[t]), K._box(rh[t])), **d,
                     # mean landmark distance between the two slots / hand size: ~0 = one hand fitted twice
                     "slot_landmark_dist": float(np.linalg.norm((lh[t, :, :2] - rh[t, :, :2]) * scale, axis=-1).mean()
                                                 / max(size, 1e-6)),
                     "left_status": NAMES[int(status[t, 0])], "right_status": NAMES[int(status[t, 1])],
                     "pose15_to_hand": near[0], "pose16_to_hand": near[1],
                     "hand_wrists_apart": float(np.linalg.norm((hw[0] - hw[1]) * scale) / sw),
                     "pose_wrists_apart": float(np.linalg.norm((pose[t, 15, :2] - pose[t, 16, :2]) * scale) / sw)})
    return pd.DataFrame(rows)


def _draw_hand(img, h, color, dashed):
    import cv2

    H, W = img.shape[:2]
    pts = [(int(x * W), int(y * H)) for x, y in h[:, :2]]
    for a, b in HAND_EDGES:
        if dashed:
            for s in np.linspace(0, 1, 6)[::2]:
                p = tuple(int(pts[a][i] + (pts[b][i] - pts[a][i]) * s) for i in (0, 1))
                q = tuple(int(pts[a][i] + (pts[b][i] - pts[a][i]) * min(1, s + 0.2)) for i in (0, 1))
                cv2.line(img, p, q, color, 1, cv2.LINE_AA)
        else:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)


def contact_sheet(video: str, raw: dict, status: np.ndarray, ev: pd.DataFrame, title: str, n: int = 6):
    """Up to n frames where the rule fired (or, failing that, any two-hand frames), evenly spaced."""
    import cv2

    fired = ev[(ev["left_status"] != "kept") | (ev["right_status"] != "kept")]
    pick = (fired if len(fired) else ev)["frame"].to_numpy()
    pick = pick[np.linspace(0, len(pick) - 1, min(n, len(pick))).round().astype(int)] if len(pick) else []
    cap = cv2.VideoCapture(video)
    decoded, t_ = {}, 0  # read in decode order, as the extraction numbered frames (seeking can be inexact)
    while len(decoded) < len(pick):
        ok, img = cap.read()
        if not ok:
            break
        if t_ in set(int(x) for x in pick):
            decoded[t_] = img
        t_ += 1
    cap.release()
    tiles = []
    for t in pick:
        img = decoded.get(int(t))
        if img is None:
            continue
        f = 360 / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * f), 360), interpolation=cv2.INTER_CUBIC)
        H, W = img.shape[:2]
        for slot, part, color in ((0, "left_hand", RED), (1, "right_hand", BLUE)):
            _draw_hand(img, raw[part][t], color, dashed=status[t, slot] == K.GHOST)
        for w, color in ((15, RED), (16, BLUE)):
            x, y = raw["pose"][t, w, :2]
            if x == x:
                cv2.circle(img, (int(x * W), int(y * H)), 7, color, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x * W), int(y * H)), 7, (255, 255, 255), 1, cv2.LINE_AA)
        r = ev[ev["frame"] == t].iloc[0]
        for i, s in enumerate([f"t{t} IoU {r.iou:.2f} L {r.left_status} R {r.right_status}",
                               f"wrists {r.pose15_to_hand:.2f} / {r.pose16_to_hand:.2f} sw"]):
            cv2.putText(img, s, (4, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, s, (4, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    head = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, sheet])


BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, np.inf]  # slot landmark distance / hand size (last = tail)
LOW_IOU = 0.1  # control band: IoU in [LOW_IOU, AMBIGUOUS_IOU), any distance — bin -1


LABELS = """Label each numbered row: video crop 3 frames before | the frame | the same frame with the
left_hand slot red, the right_hand slot blue, pose wrists as dots | 3 frames after | full frame (crop
boxed). Judge the middle frame.
  copy     both skeletons track the same physical hand — even when a second real hand is visible
           elsewhere in the crop
  two      the skeletons track two different physical hands
  unclear  cannot tell from the tile (blur, occlusion, resolution) — never guess
For a copy, also the side of the physical hand both skeletons track: left / right (the signer's own) /
unclear. Judge it from the video by following the arm to its shoulder (raw crops, neighbouring frames,
full frame) — not from image position (arms cross) and not from the pose-wrist dots, which are the
evidence being evaluated.
"""

WRIST_MARGIN, NEAR = 0.1, 0.3  # shoulder widths: own-side margin (v4 baseline), pose wrist "at the hands"


def wrist_stratum(ev: pd.DataFrame) -> pd.Series:
    """both_own: each slot's wrist nearer its own pose wrist by > WRIST_MARGIN (v4 baseline: two hands);
    otherwise stacked (both pose wrists within NEAR of a hand wrist — stacked / crossed real hands and
    PRAY / HELP-style copies) or one_far (a pose wrist away from the hands)."""
    own = ((ev["lh_to_pose15"] + WRIST_MARGIN < ev["lh_to_pose16"])
           & (ev["rh_to_pose16"] + WRIST_MARGIN < ev["rh_to_pose15"]))
    near = (ev["pose15_to_hand"] < NEAR) & (ev["pose16_to_hand"] < NEAR)
    return pd.Series(np.where(own, "both_own", np.where(near, "stacked", "one_far")), index=ev.index)


def sample_strata(ev: pd.DataFrame, per_stratum: int, seed: int, exclude: set) -> pd.DataFrame:
    """One frame per recording, per_stratum per wrist stratum and per set, among frames with both slots
    detected and IoU >= LOW_IOU (the v4 rule's scope), from recordings whose content is not in
    `exclude` (recordings already seen), each recording used once. Sets by recording content, as in
    sample_frames."""
    import zlib

    c = ev[(ev["iou"] >= LOW_IOU) & ~ev["content"].isin(exclude)].copy()
    c["stratum"] = wrist_stratum(c)
    c["bin"] = pd.cut(c["slot_landmark_dist"], BINS, right=False, labels=False)
    c["set"] = ["holdout" if zlib.crc32(str(v).encode()) % 2 else "calibration" for v in c["content"]]
    c = c.sample(frac=1, random_state=seed)
    picked, used = [], set()
    for _, g in c.groupby("set"):
        # scarcest stratum first; a recording is used once, the next unused one refills the quota
        for st in g["stratum"].value_counts().sort_values().index:
            cand = g[(g["stratum"] == st) & ~g["content"].isin(used)].drop_duplicates("content").head(per_stratum)
            used |= set(cand["content"])
            picked.append(cand)
    return pd.concat(picked, ignore_index=True)


def sample_frames(ev: pd.DataFrame, per_bin: int, seed: int) -> pd.DataFrame:
    """One frame per recording, per_bin per band and per set. Bands: copy distance (both hands detected,
    IoU >= AMBIGUOUS_IOU) up to an open tail, plus a control band of frames just below the overlap gate
    (LOW_IOU <= IoU < AMBIGUOUS_IOU) — v4 decides copies only above the gate, so labels there test it. Sets are assigned by recording *content* (raw-landmark hash), not video
    id: WLASL files identical recordings under several ids, which must not land in both sets — the
    threshold is chosen on `calibration` recordings and checked on different `holdout` recordings."""
    import zlib

    c = ev[ev["iou"] >= LOW_IOU].copy()
    c["bin"] = pd.cut(c["slot_landmark_dist"], BINS, right=False, labels=False)
    c.loc[c["iou"] < K.AMBIGUOUS_IOU, "bin"] = -1  # below the overlap gate: controls (a copy should not get here)
    c = c[c["bin"].notna()]
    c["set"] = ["holdout" if zlib.crc32(str(v).encode()) % 2 else "calibration" for v in c["content"]]
    c = c.sample(frac=1, random_state=seed).drop_duplicates(["content", "bin"])  # one frame per recording and bin
    return pd.concat([g.head(per_bin) for _, g in c.groupby(["set", "bin"])], ignore_index=True)


def frame_tile(img, raw: dict, t: int, label: str, before=None, after=None):
    """One row: [video crop 3 frames before | crop | same crop with left slot red, right slot blue, pose
    wrists 15 / 16 | crop 3 frames after | full frame with the crop boxed], cropped around the hands.
    Both slots are drawn solid: the tile must not show what the current rule decided."""
    import cv2

    H, W = img.shape[:2]
    pts = np.concatenate([raw["left_hand"][t, :, :2], raw["right_hand"][t, :, :2]])  # crop on the hands
    pts = pts[~np.isnan(pts).any(1)] * [W, H]
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    side = max(x1 - x0, y1 - y0) * 2.2 + 20
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    ann = img.copy()
    for part, color in (("left_hand", RED), ("right_hand", BLUE)):
        _draw_hand(ann, raw[part][t], color, dashed=False)
    for w, color in ((15, RED), (16, BLUE)):
        x, y = raw["pose"][t, w, :2]
        if x == x:
            cv2.circle(ann, (int(x * W), int(y * H)), 5, color, -1, cv2.LINE_AA)
            cv2.circle(ann, (int(x * W), int(y * H)), 5, (255, 255, 255), 1, cv2.LINE_AA)

    def crop(im):
        return np.zeros((220, 220, 3), np.uint8) if im is None else \
            cv2.resize(im[b:d, a:c], (220, 220), interpolation=cv2.INTER_CUBIC)

    full = img.copy()
    cv2.rectangle(full, (a, b), (c, d), (0, 255, 255), max(1, W // 300))
    full = cv2.resize(full, (int(W * 220 / H), 220), interpolation=cv2.INTER_AREA)
    sep = np.full((220, 6, 3), 255, np.uint8)
    tile = np.hstack([crop(before), sep, crop(img), crop(ann), sep, crop(after), sep, full])
    cv2.putText(tile, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(tile, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def read_frames(video: str, wanted: set) -> dict:
    import cv2

    cap, got, t = cv2.VideoCapture(video), {}, 0
    while len(got) < len(wanted):
        ok, img = cap.read()
        if not ok:
            break
        if t in wanted:
            got[t] = img
        t += 1
    cap.release()
    return got


def _clip_evidence(job: tuple):
    """(stem, gloss, full path, aspect, draw) → (evidence frame rows, and for a clip to draw: pose-kept
    raw arrays, status, original frame ids)."""
    stem, gloss, full, aspect, draw = job
    df = pd.read_parquet(full)
    content = hashlib.sha1(pd.util.hash_pandas_object(df[["frame", "type", "landmark_index", "x", "y", "z"]],
                                                      index=False).values.tobytes()).hexdigest()
    raw = K.raw_arrays(df)
    orig = np.nonzero(~np.isnan(raw["pose"][:, 0, 0]))[0]  # contract step 1: pose-less frames dropped
    raw = {k: v[orig] for k, v in raw.items()}
    status, _ = K.resolve_ghosts(raw, aspect)
    ev = frame_evidence(raw, status, aspect)
    if not ev.empty:
        ev.insert(0, "video", stem)
        ev.insert(1, "gloss", gloss)
        ev.insert(2, "content", content)
    return (ev, raw, status, orig) if draw else (ev, None, None, None)


def main() -> None:
    import cv2
    from concurrent.futures import ProcessPoolExecutor

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-root", required=True, type=Path, help="extraction shard folders")
    p.add_argument("--video-root", required=True, type=Path, help="searched recursively for the videos")
    p.add_argument("--keyword", required=True, help="shard csv path keyword (wlasl / popsign / aslcitizen)")
    p.add_argument("--ids", required=True, help="comma-separated video ids (file stems) to draw")
    p.add_argument("--evidence-all", action="store_true",
                   help="frame evidence for every clip in the shards (ghost_review.csv), not only --ids")
    p.add_argument("--sample", type=int, default=0,
                   help="labelling set: this many frames per distance bin and per set (needs --evidence-all)")
    p.add_argument("--frames", default="", help="label exactly these frames instead of a sample: video:frame,... "
                   "(frame as in ghost_review.csv); pass their videos as --ids")
    p.add_argument("--strata", type=int, default=0,
                   help="labelling set by wrist stratum (both_own / stacked / one_far): this many frames per "
                        "stratum and set, IoU >= LOW_IOU, excluding --exclude-content (needs --evidence-all)")
    p.add_argument("--exclude-content", type=Path, help="CSV with a `content` column: recordings already seen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = [s.strip() for s in a.ids.split(",") if s.strip()]
    clips = shard_clips(a.input_root, a.keyword)
    clips["stem"] = [Path(str(v)).stem for v in clips["video"]]
    clips["aspect"] = (clips["width"] / clips["height"]).fillna(0.75) if "width" in clips else 0.75
    if not a.evidence_all:
        clips = clips[clips["stem"].isin(ids)]
    videos = {Path(v).stem: v for ext in ("mp4", "mov", "MOV", "webm")
              for v in glob.glob(f"{a.video_root}/**/*.{ext}", recursive=True) if Path(v).stem in ids}
    print(f"{len(clips)} clips; {int(clips['stem'].isin(ids).sum())} of {len(ids)} to draw, {len(videos)} videos found")
    jobs = list(zip(clips["stem"], clips["gloss"], clips["full_path"], clips["aspect"], clips["stem"].isin(videos)))
    rows = []
    with ProcessPoolExecutor(os.cpu_count() or 4) as pool:
        for r, (ev, raw, status, orig) in zip(clips.itertuples(), pool.map(_clip_evidence, jobs, chunksize=16)):
            if ev.empty:
                continue
            rows.append(ev)
            if raw is None:
                continue
            # the extraction numbers frames from 0 in decode order; map back past dropped pose-less frames
            n = int(orig.max()) + 1
            raw_v = {k: np.full((n,) + v.shape[1:], np.nan) for k, v in raw.items()}
            st_v = np.zeros((n, 2), np.int8)
            for k in raw:
                raw_v[k][orig] = raw[k]
            st_v[orig] = status
            sheet = contact_sheet(videos[r.stem], raw_v, st_v, ev.assign(frame=orig[ev["frame"].to_numpy()]),
                                  f"{r.stem} {r.gloss} signer {r.participant} {int(r.width)}x{int(r.height)}: "
                                  f"ghost {int((status == K.GHOST).any(1).sum())}/{len(status)} frames")
            if sheet is not None:
                cv2.imwrite(str(a.out / f"{r.stem}.png"), sheet)
    if not rows:
        raise SystemExit("no clip with a frame where both hands are detected")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(a.out / "ghost_review.csv", index=False)
    fired = (out["left_status"] == "GHOST") | (out["right_status"] == "GHOST")
    both = fired & (out["pose15_to_hand"] < 0.3) & (out["pose16_to_hand"] < 0.3)
    print(f"{int(fired.sum())} ghost frames in {out.loc[fired, 'video'].nunique()} clips; both pose wrists "
          f"within 0.3 shoulder widths of a hand in {int(both.sum())} of them "
          f"({out.loc[both, 'video'].nunique()} clips)")
    print(f"{len(list(a.out.glob('*.png')))} contact sheets → {a.out}")
    if a.strata:
        excl = set(pd.read_csv(a.exclude_content)["content"]) if a.exclude_content else set()
        pick = sample_strata(out, a.strata, a.seed, excl)
        print(f"strata (frames available after exclusion): "
              f"{wrist_stratum(out[(out['iou'] >= LOW_IOU) & ~out['content'].isin(excl)]).value_counts().to_dict()}")
        label_sheets(out, clips, videos_by_stem(a.video_root, None), 0, a.seed, a.out, pick)
    elif a.sample or a.frames:
        label_sheets(out, clips, videos_by_stem(a.video_root, None), a.sample, a.seed, a.out,
                     chosen_frames(out, a.frames) if a.frames else None)


def videos_by_stem(root: Path, ids) -> dict:
    return {Path(v).stem: v for ext in ("mp4", "mov", "MOV", "webm")
            for v in glob.glob(f"{root}/**/*.{ext}", recursive=True) if ids is None or Path(v).stem in ids}


def chosen_frames(ev: pd.DataFrame, spec: str) -> pd.DataFrame:
    """`video:frame,...` (frame = index after pose-less frames are dropped, as in ghost_review.csv) →
    those evidence rows, band -2, sets by recording content as in sample_frames. Exits if one is absent."""
    import zlib

    want = [tuple(x.split(":")) for x in spec.split(",") if x.strip()]
    key = ev.assign(_k=ev["video"].astype(str) + ":" + ev["frame"].astype(str))
    pick = key[key["_k"].isin([f"{v}:{f}" for v, f in want])].drop(columns="_k")
    if len(pick) != len(want):
        raise SystemExit(f"{len(want) - len(pick)} of {len(want)} --frames not found among two-hand frames")
    pick = pick.assign(bin=-2, set=["holdout" if zlib.crc32(str(v).encode()) % 2 else "calibration"
                                    for v in pick["content"]])
    return pick.reset_index(drop=True)


def label_sheets(ev: pd.DataFrame, clips: pd.DataFrame, videos: dict, per_bin: int, seed: int, out: Path,
                 pick: pd.DataFrame | None = None) -> None:
    """sample_frames (or the given pick) → per set, shuffled tiles numbered #n with nothing else shown,
    6 per sheet (`label_<set>_<page>.png`) + `labels_<set>.csv` (n, empty label) to fill in, and `label_key.csv`
    (n → recording, frame, distance, bin, evidence) — kept apart so the score being calibrated is not
    seen while labelling. Exits unless every selected frame is rendered and every sheet written."""
    import cv2

    pick = (sample_frames(ev, per_bin, seed) if pick is None else pick).sample(frac=1, random_state=seed + 1)  # shuffled
    pick.insert(0, "n", range(1, len(pick) + 1))
    clip = clips.set_index("stem")
    tiles: dict = {}
    for stem, g in pick.groupby("video"):
        if stem not in videos:
            continue
        r = clip.loc[stem]
        _, raw, _, orig = _clip_evidence((stem, r["gloss"], r["full_path"], r["aspect"], True))
        frames = read_frames(videos[stem], {int(orig[f]) + k for f in g["frame"] for k in (-3, 0, 3)} - {-3, -2, -1})
        for x in g.itertuples():
            t = int(orig[x.frame])
            if t in frames:
                tiles[x.n] = frame_tile(frames[t], raw, int(x.frame), f"#{x.n}", frames.get(t - 3), frames.get(t + 3))
    missing = sorted(set(pick["n"]) - set(tiles))
    if missing:
        m = pick[pick["n"].isin(missing)]
        raise SystemExit(f"{len(missing)} of {len(pick)} selected frames not rendered (video missing or frame "
                         f"not decoded): {m[['video', 'frame']].head(10).to_dict('records')}")
    pick.to_csv(out / "label_key.csv", index=False)
    (out / "label_instructions.txt").write_text(LABELS)
    for st, g in pick.groupby("set"):
        ns = g["n"].sort_values().tolist()
        g[["n"]].sort_values("n").assign(label="", side="").to_csv(out / f"labels_{st}.csv", index=False)
        for page in range(0, len(ns), 6):
            ts = [tiles[n] for n in ns[page:page + 6]]
            w = max(t.shape[1] for t in ts)
            ts = [cv2.copyMakeBorder(t, 0, 4, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in ts]
            f = out / f"label_{st}_{page // 6 + 1:02d}.png"
            if not cv2.imwrite(str(f), np.vstack(ts)):
                raise SystemExit(f"could not write {f}")
    print(f"{len(pick)} frames rendered ({pick.groupby('set').size().to_dict()}) → label_<set>_<page>.png, "
          f"labels_<set>.csv; key in label_key.csv")


if __name__ == "__main__":
    main()
