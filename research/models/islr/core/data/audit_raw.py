"""Full-data audit of raw landmark extractions (and the ASL Signs competition parquets), before any
LMDB build. Reports, per dataset, everything a build would silently carry into training:

  PYTHONPATH=research/models python -m islr.core.data.audit_raw --dataset popsign \
      --input-root /kaggle/input --out /kaggle/working/out [--inventory inv.csv] \
      [--asl-signs-lmdb <legacy is.lmdb.mdb dir>] [--wlasl-json WLASL_v0.3.json]

Per clip (all clips, parallel): raw structure (contract.check_raw), frame ids 0..T-1 without gaps,
543 rows per frame, T = the shard's n_frames, partially detected parts, coordinates far outside the
frame, pose-less frames, hand detection rates, single hands in the slot of the other side's pose
wrist (aspect-corrected), ghost / ambiguous frames (contract.prepare), mirroring (lr_dx), extraction
version sidecar, exact raw duplicates, the stored aspect correction k against the geometric one
(W / H) / 0.75 the browser applies.
Per dataset: coverage against the selection (inventory / selected_<k>.csv), labels (gloss → sign
mapping, unmapped glosses, clips per sign), signer ids, split disjointness, near-duplicate
recordings (same gloss and length, frame-wise pose distance below the same-video threshold) within
the dataset and across splits; WLASL: instance frame ranges (clips that would need trimming).
--asl-signs-lmdb: every clip compared with the ASL Signs clips of the same sign and frame count
(same-recording test: two extractions of one video differ by <= 0.095 frame-wise, measured on 160
PopSign videos; the closest different recording was 0.123). A match means the same recording, or
the same person signing it in the same posture — either way the ASL Signs clip is not independent.
Writes <out>/audit_<dataset>.json (summary) and audit_<dataset>_clips.csv (per clip).
"""

import argparse
import glob
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SAME_VIDEO = 0.095  # frame-wise pose distance (shoulder widths), max over 160 videos extracted twice
JOINTS = (11, 12, 13, 14, 15, 16)
EXPECTED_VERSION = "mp0.10.14-fresh-tracker-v1"


def pose_traj(pose_xy: np.ndarray) -> np.ndarray:
    """(T, 33, 2) nose-relative pose → upper body / mean shoulder width (T, 6, 2)."""
    x = pose_xy[:, JOINTS] - pose_xy[:, :1]
    sw = np.nanmean(np.linalg.norm(x[:, 0] - x[:, 1], axis=-1))
    return x / max(float(sw), 1e-6)


def _clip(job: tuple) -> dict:
    path, full, aspect, n_frames = job
    from islr.core.data import contract as K

    out = {"path": path}
    try:
        df = pd.read_parquet(full)
        out["raw_sha1"] = hashlib.sha1(pd.util.hash_pandas_object(df[["frame", "type", "landmark_index", "x", "y", "z"]],
                                                                  index=False).values.tobytes()).hexdigest()
        try:
            K.check_raw(df)
        except ValueError as e:
            out["raw_error"] = str(e)
            return out
        frames = np.sort(df["frame"].unique())
        T = len(frames)
        out["T"] = T
        out["frame_gaps"] = int((frames != np.arange(frames[0], frames[0] + T)).sum()) if T else 0
        out["frames_start"] = int(frames[0]) if T else None
        out["rows_per_frame_ok"] = bool(len(df) == 543 * T)
        out["n_frames_csv_match"] = bool(n_frames != n_frames or int(n_frames) == T)
        raw = K.raw_arrays(df)
        for part in ("pose", "left_hand", "right_hand", "face"):
            a = raw[part]
            nan = np.isnan(a[..., 0])
            out[f"{part}_frames"] = float((~nan.all(1)).mean())
            out[f"{part}_partial_frames"] = int((nan.any(1) & ~nan.all(1)).sum())
            v = (a[:, :23] if part == "pose" else a)[~(nan[:, :23] if part == "pose" else nan)]  # pose: upper body (legs are extrapolated off-frame)
            out[f"{part}_far_outside"] = float(((v[:, 0] < -0.5) | (v[:, 0] > 1.5) | (v[:, 1] < -0.5) | (v[:, 1] > 1.5)).mean()) if len(v) else 0.0
        pose = raw["pose"]
        posed = ~np.isnan(pose[:, 0, 0])
        out["no_pose_frames"] = int((~posed).sum())
        # hand in the other side's slot: a single detected hand nearer the opposite pose wrist
        a_ = aspect if aspect == aspect else 0.75
        scale = np.array([a_, 1.0])
        wrong = single = 0
        for h, own, other in ((raw["left_hand"], 15, 16), (raw["right_hand"], 16, 15)):
            o = raw["right_hand" if own == 15 else "left_hand"]
            only = posed & ~np.isnan(h[:, 0, 0]) & np.isnan(o[:, 0, 0])
            if only.any():
                d_own = np.linalg.norm((h[only, 0, :2] - pose[only, own, :2]) * scale, axis=-1)
                d_oth = np.linalg.norm((h[only, 0, :2] - pose[only, other, :2]) * scale, axis=-1)
                wrong += int((d_oth < 0.5 * d_own).sum())
                single += int(only.sum())
        out["single_hand_frames"], out["wrong_slot_frames"] = single, wrong
        clip = K.prepare(df, a_)
        st = clip["status"]
        out.update({"kept_T": int(len(st)), "ghost_frames": int((st == K.GHOST).sum()),
                    "ambiguous_frames": int((st == K.AMBIGUOUS).sum()),
                    "both_hands_frames": int(((st == K.KEPT) | (st == K.AMBIGUOUS)).all(1).sum()),
                    "no_hand_clip": bool(not clip["presence"].any())})
        if posed.sum() >= 2:
            p = pose[posed][:, :, :2]
            x = p[:, 11:13]  # shoulders, normalised units
            sw = np.nanmedian(np.abs(x[:, 0, 0] - x[:, 1, 0]))
            drop = np.nanmedian((x[:, 0, 1] + x[:, 1, 1]) / 2 - p[:, 0, 1])
            out["ratio_raw"] = float(drop / sw) if sw > 0 else np.nan  # shoulder drop / width, frame units
            out["lr_dx_median"] = float(np.nanmedian(x[:, 0, 0] - x[:, 1, 0]))
            out["traj"] = pose_traj(p).astype(np.float32).tobytes()
        side = full[: -len(".parquet")] + ".meta.json"
        if os.path.exists(side):
            m = json.load(open(side))
            out["extraction_version"] = m.get("version")
            out["fresh_tracker"] = m.get("fresh_tracker")
    except Exception as e:  # noqa: BLE001 — reported per clip
        out["read_error"] = f"{type(e).__name__}: {e}"
    return out


def shard_clips(root: Path, keyword: str) -> pd.DataFrame:
    rows = []
    for c in sorted(glob.glob(f"{root}/**/shard_*.csv", recursive=True)):
        if keyword not in c.lower():
            continue
        d = pd.read_csv(c, dtype={"video": str})
        d["full_path"] = [str(Path(c).parent / p) for p in d["path"]]
        d["shard_csv"] = c
        sel = Path(c).with_name(Path(c).name.replace("shard_", "selected_"))
        d["has_selection_file"] = sel.exists()
        rows.append(d)
    if not rows:
        raise SystemExit(f"no {keyword} shard_*.csv under {root}")
    return pd.concat(rows, ignore_index=True)


def near_duplicates(res: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Pairs of clips with the same group key and length whose frame-wise pose distance is below
    the same-video threshold (one recording listed twice, possibly under different ids / splits)."""
    pairs = []
    ok = res[res["traj"].notna()]
    for _, g in ok.groupby(group_cols + ["kept_T"]):
        if len(g) < 2:
            continue
        tr = [np.frombuffer(b, np.float32).reshape(-1, len(JOINTS), 2) for b in g["traj"]]
        paths = g["path"].tolist()
        for i in range(len(tr)):
            for j in range(i + 1, len(tr)):
                d = float(np.nanmean(np.abs(tr[i] - tr[j])))
                if d <= SAME_VIDEO:
                    pairs.append({"a": paths[i], "b": paths[j], "distance": d})
    return pd.DataFrame(pairs, columns=["a", "b", "distance"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, choices=["popsign", "aslcitizen", "wlasl", "aslsigns"])
    p.add_argument("--input-root", required=True, type=Path, help="shard folders, or the competition folder (aslsigns)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--inventory", type=Path)
    p.add_argument("--asl-signs-lmdb", type=Path, help="folder of the legacy ASL Signs is.lmdb.mdb + train.csv (same-recording test)")
    p.add_argument("--wlasl-json", type=Path)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"dataset": a.dataset}

    if a.dataset == "aslsigns":
        df = pd.read_csv(a.input_root / "train.csv")
        df["full_path"] = [str(a.input_root / x) for x in df["path"]]
        df = df.rename(columns={"participant_id": "participant", "sign": "gloss"})
        df["width"] = df["height"] = np.nan
        df["split"] = "train"
    else:
        df = shard_clips(a.input_root, a.dataset)
    if a.limit:
        df = df.head(a.limit)
    aspect = (df["width"] / df["height"]) if {"width", "height"} <= set(df) else pd.Series(np.nan, index=df.index)
    jobs = list(zip(df["path"], df["full_path"], aspect, df.get("n_frames", pd.Series(np.nan, index=df.index))))
    with ProcessPoolExecutor(a.workers) as pool:
        res = pd.DataFrame(list(pool.map(_clip, jobs, chunksize=16)))
    res = df.drop(columns=["full_path"]).merge(res, on="path", how="left")
    print(f"{len(res)} clips audited")

    # --- per-clip integrity ------------------------------------------------------------------
    def count(col, cond=lambda s: s.astype(bool)):
        return int(cond(res[col].fillna(0)).sum()) if col in res else 0
    rep["clips"] = len(res)
    rep["integrity"] = {
        "read_errors": count("read_error", lambda s: s.astype(str).ne("0")),
        "raw_structure_errors": count("raw_error", lambda s: s.astype(str).ne("0")),
        "raw_error_examples": res["raw_error"].dropna().head(5).tolist() if "raw_error" in res else [],
        "clips_with_frame_gaps": count("frame_gaps", lambda s: s > 0),
        "clips_not_starting_at_frame_0": int((res["frames_start"].fillna(0) != 0).sum()),
        "clips_rows_per_frame_wrong": int((res["rows_per_frame_ok"] == False).sum()),  # noqa: E712
        "clips_T_ne_shard_n_frames": int((res["n_frames_csv_match"] == False).sum()),  # noqa: E712
        "clips_with_partial_parts": {pt: count(f"{pt}_partial_frames", lambda s: s > 0) for pt in ("pose", "left_hand", "right_hand", "face")},
        "far_outside_frame_fraction_mean": {pt: float(res[f"{pt}_far_outside"].mean()) for pt in ("pose", "left_hand", "right_hand", "face")},
        "clips_with_no_pose_frames": count("no_pose_frames", lambda s: s > 0),
        "clips_all_frames_without_pose": int((res["kept_T"].fillna(-1) == 0).sum()),
        "no_hand_clips": int(res["no_hand_clip"].fillna(False).astype(bool).sum()),
        "extraction_versions": res.get("extraction_version", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "fresh_tracker": res.get("fresh_tracker", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
    }
    rep["detection"] = {
        "hand_frame_rate": {h: float(res[f"{h}_frames"].mean()) for h in ("left_hand", "right_hand")},
        "pose_frame_rate": float(res["pose_frames"].mean()), "face_frame_rate": float(res["face_frames"].mean()),
        "wrong_slot_single_hand_frames": [int(res["wrong_slot_frames"].sum()), int(res["single_hand_frames"].sum())],
        "clips_with_wrong_slot": int((res["wrong_slot_frames"].fillna(0) > 0).sum()),
        "ghost_frames": int(res["ghost_frames"].sum()), "clips_with_ghosts": int((res["ghost_frames"].fillna(0) > 0).sum()),
        "ambiguous_frames": int(res["ambiguous_frames"].sum()),
        "clips_with_both_hands_in_a_frame": int((res["both_hands_frames"].fillna(0) > 0).sum()),
        "mirrored_clips (pose 11 x <= pose 12 x)": int((res["lr_dx_median"] <= 0).sum()),
    }
    if res["width"].notna().any():
        geo = (res["width"] / res["height"]) / 0.75
        k_ratio = res["ratio_raw"] / 0.40  # what the builders' per-clip correction applies (x scaled so ratio = 0.40)
        rep["aspect"] = {
            "frame_sizes": (res["width"].astype("Int64").astype(str) + "x" + res["height"].astype("Int64").astype(str)).value_counts().head(8).to_dict(),
            "geometric_k (W/H)/0.75 quantiles": geo.quantile([.1, .5, .9]).round(3).to_dict(),
            "per_clip_ratio_k quantiles": k_ratio.quantile([.1, .5, .9]).round(3).to_dict(),
            "ratio_k / geometric_k quantiles": (k_ratio / geo).quantile([.1, .5, .9]).round(3).to_dict(),
        }
    rep["T"] = res["kept_T"].quantile([0, .01, .5, .99, 1]).to_dict()

    # --- labels, signers, splits ---------------------------------------------------------------
    rep["labels"] = {"glosses": int(res["gloss"].nunique()), "clips_per_gloss": res["gloss"].value_counts().quantile([0, .5, 1]).to_dict()}
    pid = res["participant"]
    rep["signers"] = {"distinct": int(pid.nunique()), "missing": int(pid.isna().sum()),
                      "examples": pid.dropna().astype(str).unique()[:5].tolist()}
    if "split" in res:
        s = {k: set(g["participant"].astype(str)) for k, g in res.groupby("split")}
        rep["splits"] = {"clips": res["split"].value_counts().to_dict(),
                         "signers": {k: len(v) for k, v in s.items()},
                         "signers_in_two_splits": sorted({x for k1 in s for k2 in s if k1 < k2 for x in s[k1] & s[k2]})}

    # --- coverage --------------------------------------------------------------------------------
    if a.inventory:
        inv = pd.read_csv(a.inventory)
        sel, listed = set(inv["path"]), set(res["path"])
        rep["coverage"] = {"selected": len(sel), "listed": len(listed), "missing": len(sel - listed),
                           "listed_not_selected": len(listed - sel), "missing_examples": sorted(sel - listed)[:10]}

    # --- duplicates --------------------------------------------------------------------------------
    exact = res[res["raw_sha1"].notna() & res.duplicated("raw_sha1", keep=False)]
    rep["exact_raw_duplicates"] = {"clips": len(exact), "groups": int(exact["raw_sha1"].nunique()),
                                   "groups_with_different_glosses": int((exact.groupby("raw_sha1")["gloss"].nunique() > 1).sum()),
                                   "groups_across_splits": int((exact.groupby("raw_sha1")["split"].nunique() > 1).sum()) if "split" in exact else None}
    near = near_duplicates(res, ["gloss"])
    if len(near):
        m = res.set_index("path")
        near["split_a"], near["split_b"] = m.loc[near["a"], "split"].values, m.loc[near["b"], "split"].values
        near["signer_a"], near["signer_b"] = m.loc[near["a"], "participant"].astype(str).values, m.loc[near["b"], "participant"].astype(str).values
    # one signer repeating a sign in the same posture can come this close too: only pairs under
    # DIFFERENT signer ids (or across splits) indicate one recording listed twice
    diff = near[near["signer_a"] != near["signer_b"]] if len(near) else near
    rep["near_duplicate_recordings_same_gloss"] = {
        "pairs_same_signer (info: repeated takes or duplicates)": int(len(near) - len(diff)),
        "pairs_different_signer_ids": len(diff),
        "pairs_across_splits": int((near["split_a"] != near["split_b"]).sum()) if len(near) else 0,
        "examples_different_signers": diff.head(5).to_dict("records") if len(diff) else []}
    near.to_csv(a.out / f"audit_{a.dataset}_near_duplicates.csv", index=False)

    # --- WLASL instance frame ranges ----------------------------------------------------------------
    if a.dataset == "wlasl":
        js = a.wlasl_json or next(iter(sorted(Path(h) for h in glob.glob(f"{a.input_root}/**/WLASL_v0.3.json", recursive=True))), None)
        if js:
            inst = {str(i["video_id"]): i for e in json.loads(Path(js).read_text()) for i in e["instances"]}
            vid = res["video"].astype(str).str.replace(".mp4", "", regex=False)
            fs = vid.map(lambda v: inst.get(v, {}).get("frame_start", 1))
            fe = vid.map(lambda v: inst.get(v, {}).get("frame_end", -1))
            trim = (fs != 1) | (fe != -1)
            span = (fe - fs + 1).where(fe != -1)
            rep["wlasl_frame_ranges"] = {
                "instances_with_a_range": int(trim.sum()),
                "of_which_T_matches_range": int((trim & (res["kept_T"] + res["no_pose_frames"].fillna(0) - span).abs().le(2)).sum()),
                "of_which_T_longer_than_range": int((trim & ((res["kept_T"] + res["no_pose_frames"].fillna(0)) > span + 2)).sum()),
                "examples_longer": res.loc[trim & ((res["kept_T"] + res["no_pose_frames"].fillna(0)) > span + 2), "video"].head(5).tolist()}

    # --- same-recording test against ASL Signs ------------------------------------------------------
    if a.asl_signs_lmdb:
        import lmdb
        import torch

        from ..config import ALL_COLUMNS

        legacy = hashlib.md5(("|".join(ALL_COLUMNS) + "|v2_fallback").encode()).hexdigest()[:8]
        isdf = pd.read_csv(a.asl_signs_lmdb / "train.csv")
        by_lower = {s.lower(): s for s in isdf["sign"].unique()}
        env = lmdb.open(str(a.asl_signs_lmdb / "is.lmdb.mdb"), subdir=False, readonly=True, lock=False,
                        map_size=os.path.getsize(a.asl_signs_lmdb / "is.lmdb.mdb") + (1 << 28))
        lens = {}
        with env.begin() as txn:
            for x in isdf["path"]:
                n = txn.get(f"{legacy}:len:{x}".encode())
                lens[x] = int(n) if n else -1
        isdf["T_raw"] = isdf["path"].map(lens)
        pose_cols = [63 + 3 * j + c for j in range(33) for c in (0, 1)]
        best, match_path = [], []
        with env.begin() as txn:
            groups = {k: g for k, g in isdf.groupby(["sign", "T_raw"])}
            cache = {}
            for r in res.itertuples():
                sign = by_lower.get(str(r.gloss).lower())
                g = groups.get((sign, int(r.T))) if sign and r.T == r.T else None
                if g is None or not isinstance(r.traj, bytes):
                    best.append(np.nan); match_path.append(None)
                    continue
                mine = np.frombuffer(r.traj, np.float32).reshape(-1, len(JOINTS), 2)
                bd, bp = np.inf, None
                for c in g["path"]:
                    if c not in cache:
                        v = torch.load(io.BytesIO(bytes(txn.get(f"{legacy}:{c}".encode()))), weights_only=True).numpy()
                        pz = v[:, pose_cols].reshape(len(v), 33, 2)
                        cache[c] = pose_traj(pz)  # stored pose is already nose-relative (nose = 0)
                    t = cache[c]
                    if len(t) == len(mine):
                        d = float(np.nanmean(np.abs(t - mine)))
                        if d < bd:
                            bd, bp = d, c
                best.append(bd if bd < np.inf else np.nan); match_path.append(bp)
        res["asl_signs_best_d"], res["asl_signs_best_path"] = best, match_path
        hit = res["asl_signs_best_d"] <= SAME_VIDEO
        rep["same_recording_as_asl_signs"] = {
            "clips_compared": int(res["asl_signs_best_d"].notna().sum()), "matches": int(hit.sum()),
            "best_distance_quantiles": res["asl_signs_best_d"].quantile([0, .01, .05, .5]).round(4).to_dict(),
            "matched_asl_signs_signers": isdf.set_index("path").loc[res.loc[hit, "asl_signs_best_path"], "participant_id"].value_counts().to_dict() if hit.any() else {}}

    res.drop(columns=["traj"], errors="ignore").to_csv(a.out / f"audit_{a.dataset}_clips.csv", index=False)
    (a.out / f"audit_{a.dataset}.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
