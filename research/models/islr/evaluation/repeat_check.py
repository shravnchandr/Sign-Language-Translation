"""Repeated-frame check, in two committed stages (criteria: audit doc, 9a59d20 + a25b6a0 + 3c68073).

  calibrate  PopSign calibration signers only (crc32(user) % 2 == 0): the global-s4 threshold with the
             best balanced accuracy (repeated ×4 = probed 120 fps vs ordinary = 29–31 fps), searched over
             every observed calibration score, written to --threshold-file. Commit that file before
             running `evaluate`.
  evaluate   reads the frozen threshold (never refits) and applies the gates in order, stopping at the
             first that fails: (2) PopSign evaluation signers — recall ×4, false positives on ordinary and
             on held-pose controls (ordinary clips in the lowest quartile of mean motion); the 60 fps ×2
             clips reported alongside, outside the decision; (3–4) ASL Signs candidates (ASL side of a
             120 fps PopSign match) vs controls (ASL side of a 29–31 fps match); only if (4) passes, flags
             for the whole of ASL Signs.

Input coverage is required, not assumed: any read error, any probed PopSign clip or any ASL Signs clip
(--aslsigns-paths) missing from the scores stops the run. A valid score of NaN (unscorable: too few
motion steps or no motion) is never flagged and stays in every denominator. Rates are compared
unrounded: candidates need >= 70 % (42 of 59), controls <= 2 % (at most 3 of the 192 distinct ASL Signs
clips behind the 198 control matches).

  PYTHONPATH=research/models python -m islr.evaluation.repeat_check calibrate \
      --scores results/kaggle/repeat-frames/out/repeat_frames.csv --probe results/kaggle/fps-popsign/video_probe.csv \
      --aslsigns-paths results/kaggle/release-aslsigns/out/train.csv
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


def load(a) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scores with complete coverage → (PopSign with labels and halves, ASL Signs, probe). Exits otherwise."""
    s = pd.read_csv(a.scores)
    probe = pd.read_csv(a.probe)
    want = {"popsign": set(probe["path"]), "aslsigns": set(pd.read_csv(a.aslsigns_paths)["path"])}
    problems = []
    bad = s["error"].notna()
    if bad.any():
        problems.append(f"{int(bad.sum())} read errors {s[bad].groupby('dataset').size().to_dict()}, e.g. "
                        f"{s.loc[bad, ['path', 'error']].head(2).to_dict('records')}")
    for ds, paths in want.items():
        got = s.loc[s["dataset"] == ds, "path"]
        if got.duplicated().any():
            problems.append(f"{ds}: {int(got.duplicated().sum())} clips scored twice")
        missing, extra = paths - set(got), set(got) - paths
        if missing or extra:
            problems.append(f"{ds}: {len(missing)} expected clips not scored, {len(extra)} unexpected "
                            f"(e.g. {sorted(missing)[:2] or sorted(extra)[:2]})")
    if problems:
        raise SystemExit("incomplete scores — " + "; ".join(problems))
    p = s[s["dataset"] == "popsign"].merge(probe[["path", "fps"]], on="path")
    p["label"] = np.select([p["fps"].round() == 120, p["fps"].round() == 60], ["rep4", "rep2"], "ordinary")
    p["user"] = [Path(x).name.split("-")[0] for x in p["path"]]
    p["half"] = ["eval" if zlib.crc32(u.encode()) % 2 else "calib" for u in p["user"]]
    print(f"coverage complete: {len(p)} PopSign, {int((s['dataset'] == 'aslsigns').sum())} ASL Signs clips; "
          f"unscorable {p['s4'].isna().sum()} / {int(s.loc[s['dataset'] == 'aslsigns', 's4'].isna().sum())}")
    return p, s[s["dataset"] == "aslsigns"], probe


def calibrate(a) -> None:
    p, _, _ = load(a)
    cal = p[(p["half"] == "calib") & p["label"].isin(["rep4", "ordinary"])]
    pos, neg = cal.loc[cal["label"] == "rep4", "s4"], cal.loc[cal["label"] == "ordinary", "s4"]
    best = None
    for th in [*np.unique(cal["s4"].dropna()), np.inf]:  # every observed score (flag >= it), and flag none
        rec, fp = float(flagged(pos, th).mean()), float(flagged(neg, th).mean())
        if best is None or (rec + 1 - fp) / 2 > best["balanced_accuracy"]:
            best = {"threshold": float(th), "recall_x4": rec, "fp_ordinary": fp, "balanced_accuracy": (rec + 1 - fp) / 2}
    out = {**best, "calibration_signers": sorted(cal["user"].unique()), "n_x4": int(len(pos)), "n_ordinary": int(len(neg)),
           "unscorable": {"x4": int(pos.isna().sum()), "ordinary": int(neg.isna().sum())}, "scores": str(a.scores)}
    a.threshold_file.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "calibration_signers"}, indent=1))
    print(f"wrote {a.threshold_file} — commit it before `evaluate`")


def evaluate(a) -> None:
    from scipy.stats import fisher_exact

    frozen = json.loads(a.threshold_file.read_text())
    th = frozen["threshold"]
    p, asl, probe = load(a)
    if sorted(p.loc[p["half"] == "calib", "user"].unique()) != frozen["calibration_signers"]:
        raise SystemExit("calibration signers differ from the frozen threshold file")
    report = {"threshold": th}

    def finish(stage: str) -> None:
        report["stopped_after"] = stage
        print(f"stopped after {stage}")
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "repeat_check.json").write_text(json.dumps(report, indent=1))

    ev = p[p["half"] == "eval"]
    pos, neg = ev[ev["label"] == "rep4"], ev[ev["label"] == "ordinary"]
    held = neg[neg["mean_motion"] <= neg["mean_motion"].quantile(0.25)]
    rec, fp, hfp = (float(flagged(g["s4"], th).mean()) for g in (pos, neg, held))
    sep = bool(rec >= RECALL_MIN and fp <= FP_MAX and hfp <= HELD_FP_MAX)
    x2 = p[p["label"] == "rep2"]
    report.update({"recall_x4": rec, "fp_ordinary": fp, "fp_held": hfp, "separates": sep,
                   "n": {"x4": len(pos), "ordinary": len(neg), "held": len(held)},
                   "unscorable": {"x4": int(pos["s4"].isna().sum()), "ordinary": int(neg["s4"].isna().sum())},
                   "x2": {"n": len(x2), "s2_median": float(x2["s2"].median()), "flagged_by_x4_threshold": int(flagged(x2["s4"], th).sum())}})
    print(f"frozen threshold s4 >= {th} (calibration: recall {frozen['recall_x4']:.2%}, FP {frozen['fp_ordinary']:.3%})")
    print(f"2. evaluation signers {ev['user'].nunique()}: recall x4 {rec:.2%} ({int(flagged(pos['s4'], th).sum())}/{len(pos)}, "
          f"unscorable {report['unscorable']['x4']}); FP ordinary {fp:.3%} ({int(flagged(neg['s4'], th).sum())}/{len(neg)}, "
          f"unscorable {report['unscorable']['ordinary']}); FP held-pose {hfp:.3%} ({int(flagged(held['s4'], th).sum())}/{len(held)})")
    print(f"   x2 (60 fps, outside the decision): {report['x2']}")
    print(f"   separates reliably (recall >= {RECALL_MIN:.0%}, FP <= {FP_MAX:.1%}, held-pose FP <= {HELD_FP_MAX:.0%}): {sep}")
    if not sep:
        return finish("gate 2 (PopSign separation failed; ASL Signs not examined)")

    m = pd.read_csv(MATCHES).merge(probe[["path", "fps"]], on="path")
    res = {}
    for name, paths in (("candidates", m.loc[m["fps"].round() == 120, "asl_signs_path"]),
                        ("controls", m.loc[m["fps"].round() < 45, "asl_signs_path"])):
        g = asl[asl["path"].isin(set(paths))]  # coverage checked in load(): every listed clip is scored
        f = flagged(g["s4"], th)
        res[name] = (int(f.sum()), len(g))
        report[name] = {"flagged": int(f.sum()), "n": len(g), "unscorable": int(g["s4"].isna().sum())}
        print(f"3. ASL Signs {name}: flagged {int(f.sum())}/{len(g)} = {f.mean():.2%} (unscorable {int(g['s4'].isna().sum())}), "
              f"s4 median {g['s4'].median():.3f}")
    (ka, na), (kb, nb) = res["candidates"], res["controls"]
    pval = float(fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="greater")[1])
    conv = bool(ka / na >= CAND_MIN and kb / nb <= CTRL_MAX and pval < P_MAX)
    report.update({"fisher_p": pval, "convincing": conv})
    print(f"4. candidates {ka}/{na} >= {CAND_MIN:.0%}: {ka / na >= CAND_MIN}; controls {kb}/{nb} <= {CTRL_MAX:.0%}: "
          f"{kb / nb <= CTRL_MAX}; Fisher p {pval:.2e} < {P_MAX}: {pval < P_MAX} → convincing: {conv}")
    if not conv:
        return finish("gate 4 (proxy comparison failed; no whole-dataset flags)")

    fa = flagged(asl["s4"], th)
    report.update({"aslsigns_flagged": int(fa.sum()), "aslsigns_n": len(asl), "aslsigns_unscorable": int(asl["s4"].isna().sum())})
    print(f"   ASL Signs all: {int(fa.sum())}/{len(asl)} flagged ({fa.mean():.2%}), unscorable {report['aslsigns_unscorable']}")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        asl.assign(flagged=fa).to_csv(a.out / "aslsigns_repeat_flags.csv", index=False)
    finish("all gates passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage", choices=["calibrate", "evaluate"])
    p.add_argument("--scores", required=True, type=Path, help="repeat_frames.csv")
    p.add_argument("--probe", required=True, type=Path, help="video_probe.csv of the full PopSign fps probe (every clip expected)")
    p.add_argument("--aslsigns-paths", required=True, type=Path, help="CSV with `path`: every ASL Signs clip expected")
    p.add_argument("--threshold-file", type=Path, default=THRESHOLD_FILE)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    (calibrate if a.stage == "calibrate" else evaluate)(a)


if __name__ == "__main__":
    main()
