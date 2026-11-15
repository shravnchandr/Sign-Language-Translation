"""Repeated-frame check, in two committed stages (criteria: audit doc, 9a59d20 + a25b6a0).

  calibrate  PopSign calibration signers only (crc32(user) % 2 == 0): the global-s4 threshold with the
             best balanced accuracy (repeated ×4 = probed 120 fps vs ordinary = 29–31 fps), written to
             --threshold-file. Commit that file before running `evaluate`.
  evaluate   reads the frozen threshold (never refits) and applies criteria 2–4: PopSign evaluation
             signers (recall ×4, false positives on ordinary and on held-pose controls = ordinary clips
             in the lowest quartile of mean motion), the 60 fps ×2 clips separately, then ASL Signs:
             candidates (ASL side of a 120 fps PopSign match) vs controls (ASL side of a 29–31 fps match).

Unscorable clips (s4 NaN) are never flagged and stay in every denominator. Rates are compared unrounded:
candidates need >= 70 % (42 of 59), controls <= 2 % (at most 3 of the 192 distinct ASL Signs clips
behind the 198 control matches).

  PYTHONPATH=research/models python -m islr.evaluation.repeat_check calibrate \
      --scores results/kaggle/repeat-frames/out/repeat_frames.csv --probe results/kaggle/fps-popsign/video_probe.csv
"""

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "core" / "data"
THRESHOLD_FILE = HERE / "repeat_frames_threshold.json"
MATCHES = HERE / "popsign_asl_signs_matches.csv"
RECALL_MIN, FP_MAX, HELD_FP_MAX = 0.98, 0.005, 0.01
CAND_MIN, CTRL_MAX, P_MAX = 0.70, 0.02, 1e-3


def flagged(s4: pd.Series, th: float) -> pd.Series:
    return (s4 >= th).fillna(False).astype(bool)  # unscorable (NaN) = not flagged


def popsign(scores: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    p = scores[scores["dataset"] == "popsign"].merge(probe[["path", "fps"]], on="path", how="left")
    if p["fps"].isna().any():
        raise SystemExit(f"{int(p['fps'].isna().sum())} PopSign clips without a probed fps")
    p["label"] = np.select([p["fps"].round() == 120, p["fps"].round() == 60], ["rep4", "rep2"], "ordinary")
    p["user"] = [Path(x).name.split("-")[0] for x in p["path"]]
    p["half"] = ["eval" if zlib.crc32(u.encode()) % 2 else "calib" for u in p["user"]]
    return p


def load(a) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(a.scores)
    bad = s["error"].notna()
    print(f"read errors: {int(bad.sum())} {s[bad].groupby('dataset').size().to_dict()} (left out, listed in the scores file)")
    s = s[~bad]
    return s, popsign(s, pd.read_csv(a.probe))


def calibrate(a) -> None:
    s, p = load(a)
    cal = p[(p["half"] == "calib") & p["label"].isin(["rep4", "ordinary"])]
    pos, neg = cal[cal["label"] == "rep4"], cal[cal["label"] == "ordinary"]
    best = None
    for th in np.round(np.arange(0.26, 0.951, 0.0025), 4):
        rec, fp = flagged(pos["s4"], th).mean(), flagged(neg["s4"], th).mean()
        if best is None or (rec + 1 - fp) / 2 > best["balanced_accuracy"]:
            best = {"threshold": float(th), "recall_x4": float(rec), "fp_ordinary": float(fp),
                    "balanced_accuracy": float((rec + 1 - fp) / 2)}
    out = {**best, "calibration_signers": sorted(cal["user"].unique()), "n_x4": len(pos), "n_ordinary": len(neg),
           "unscorable": {"x4": int(pos["s4"].isna().sum()), "ordinary": int(neg["s4"].isna().sum())},
           "scores": str(a.scores)}
    a.threshold_file.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "calibration_signers"}, indent=1))
    print(f"wrote {a.threshold_file} — commit it before `evaluate`")


def evaluate(a) -> None:
    frozen = json.loads(a.threshold_file.read_text())
    th = frozen["threshold"]
    s, p = load(a)
    if sorted(p.loc[p["half"] == "calib", "user"].unique()) != frozen["calibration_signers"]:
        raise SystemExit("calibration signers differ from the frozen threshold file")
    ev = p[p["half"] == "eval"]
    pos, neg = ev[ev["label"] == "rep4"], ev[ev["label"] == "ordinary"]
    held = neg[neg["mean_motion"] <= neg["mean_motion"].quantile(0.25)]
    rec, fp, hfp = flagged(pos["s4"], th).mean(), flagged(neg["s4"], th).mean(), flagged(held["s4"], th).mean()
    sep = rec >= RECALL_MIN and fp <= FP_MAX and hfp <= HELD_FP_MAX
    print(f"frozen threshold s4 >= {th} (calibration: recall {frozen['recall_x4']:.2%}, FP {frozen['fp_ordinary']:.3%})")
    print(f"2. evaluation signers {ev['user'].nunique()}: recall x4 {rec:.2%} ({int(flagged(pos['s4'], th).sum())}/{len(pos)}, "
          f"unscorable {int(pos['s4'].isna().sum())}); FP ordinary {fp:.3%} ({int(flagged(neg['s4'], th).sum())}/{len(neg)}, "
          f"unscorable {int(neg['s4'].isna().sum())}); FP held-pose {hfp:.3%} ({int(flagged(held['s4'], th).sum())}/{len(held)})")
    print(f"   separates reliably (recall >= {RECALL_MIN:.0%}, FP <= {FP_MAX:.1%}, held-pose FP <= {HELD_FP_MAX:.0%}): {sep}")
    x2 = p[p["label"] == "rep2"]
    print(f"   x2 (60 fps, reported separately): n={len(x2)}, s2 median {x2['s2'].median():.3f}, s4 median "
          f"{x2['s4'].median():.3f}, flagged by the x4 threshold {int(flagged(x2['s4'], th).sum())}")
    print("   per-label quantiles (p1 / p50 / p99) of s4:", {k: g["s4"].quantile([.01, .5, .99]).round(3).tolist()
                                                         for k, g in p.groupby("label")})
    asl = s[s["dataset"] == "aslsigns"]
    m = pd.read_csv(MATCHES).merge(pd.read_csv(a.probe)[["path", "fps"]], on="path")
    res = {}
    for name, paths in (("candidates", m.loc[m["fps"].round() == 120, "asl_signs_path"]),
                        ("controls", m.loc[m["fps"].round() < 45, "asl_signs_path"])):
        paths = set(paths)
        g = asl[asl["path"].isin(paths)]
        f = flagged(g["s4"], th)
        res[name] = (int(f.sum()), len(paths))  # denominator: every listed clip (a clip missing from the scores counts as not flagged)
        print(f"3. ASL Signs {name}: flagged {int(f.sum())}/{len(paths)} = {f.sum() / len(paths):.2%} "
              f"(scored {len(g)}, unscorable {int(g['s4'].isna().sum())}), s4 median {g['s4'].median():.3f}")
    from scipy.stats import fisher_exact

    (ka, na), (kb, nb) = res["candidates"], res["controls"]
    pval = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="greater")[1]
    conv = sep and ka / na >= CAND_MIN and kb / nb <= CTRL_MAX and pval < P_MAX
    print(f"4. candidates {ka}/{na} >= {CAND_MIN:.0%}: {ka / na >= CAND_MIN}; controls {kb}/{nb} <= {CTRL_MAX:.0%}: "
          f"{kb / nb <= CTRL_MAX}; Fisher p {pval:.2e} < {P_MAX}: {pval < P_MAX} → convincing: {conv}")
    fa = flagged(asl["s4"], th)
    print(f"   ASL Signs all (acted on only if convincing): {int(fa.sum())}/{len(asl)} flagged ({fa.mean():.2%}), "
          f"unscorable {int(asl['s4'].isna().sum())}")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        asl.assign(flagged=fa).to_csv(a.out / "aslsigns_repeat_flags.csv", index=False)
        (a.out / "repeat_check.json").write_text(json.dumps({
            "threshold": th, "recall_x4": rec, "fp_ordinary": fp, "fp_held": hfp, "separates": sep,
            "candidates": res["candidates"], "controls": res["controls"], "fisher_p": pval, "convincing": conv,
            "aslsigns_flagged": int(fa.sum()), "aslsigns_unscorable": int(asl["s4"].isna().sum())}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["calibrate", "evaluate"])
    p.add_argument("--scores", required=True, type=Path, help="repeat_frames.csv")
    p.add_argument("--probe", required=True, type=Path, help="video_probe.csv of the full PopSign fps probe")
    p.add_argument("--threshold-file", type=Path, default=THRESHOLD_FILE)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    (calibrate if a.stage == "calibrate" else evaluate)(a)


if __name__ == "__main__":
    main()
