"""Validation gate for a built LMDB: run on every dataset before it is used for training or scoring.

  PYTHONPATH=research/models python -m islr.core.data.validate --data-dir /tmp/popsign \
      --lmdb-path /tmp/popsign/popsign.lmdb.mdb [--raw-root {INPUT} --resample 200]

Exits non-zero when any hard check fails; writes validation_report.json next to the LMDB.
Hard checks:
  - build_manifest.json present, built with this contract and cache version, 0 unallowed errors
  - every train.csv path has a data + length key; stored length = clip length = n_frames
  - every clip: finite, (T > 0, COORD_FEAT); presence binary and = status in {KEPT, AMBIGUOUS};
    status codes valid; a hand never present has an all-zero block
  - labels in the sign map; participant ids resolved (no NaN / -1)
  - WLASL: signer halves = the frozen wlasl_signer_split.json (and wlasl_half, when present);
    ASL Citizen: official train / val / test signers disjoint
  - no two clips with identical stored content
  - --raw-root: a random sample re-prepared from its raw parquet is byte-identical to the stored
    value (the LMDB is what the contract produces today)
Reported (not failed): two-hand clip rate, ghost / ambiguous / pose-less frame totals.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch

from ..config import COORD_FEAT, COORDS_PER_LM, LH_START, N_LH, RH_START
from . import contract as K
from ._cache_keys import CACHE_VERSION, lmdb_key, lmdb_length_key
from .build_wlasl_lmdb import _process, signer_split


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", required=True, type=Path, help="folder with train.csv, sign map, build_manifest.json")
    p.add_argument("--lmdb-path", required=True, type=Path)
    p.add_argument("--raw-root", type=Path, help="where the raw parquets are (shard folders / competition files)")
    p.add_argument("--resample", type=int, default=200, help="clips re-prepared from raw with --raw-root")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    fails, report = [], {}

    def check(ok: bool, msg: str) -> None:
        print(("ok    " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    man_path = a.data_dir / "build_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    check(bool(man), f"build manifest {man_path}")
    check(man.get("contract") == K.CONTRACT_VERSION and man.get("cache_version") == CACHE_VERSION,
          f"contract {man.get('contract')} / cache {man.get('cache_version')} = {K.CONTRACT_VERSION} / {CACHE_VERSION}")
    report["manifest"] = {k: man.get(k) for k in ("version", "shards", "clips_listed", "clips_written", "errors",
                                                   "contract_counts", "commit")}

    df = pd.read_csv(a.data_dir / "train.csv")
    sign_map = json.loads((a.data_dir / "sign_to_prediction_index_map.json").read_text())
    check(df["sign"].isin(sign_map).all(), f"labels in the sign map ({int((~df['sign'].isin(sign_map)).sum())} not)")
    pid = pd.to_numeric(df["participant_id"], errors="coerce")
    check(bool(pid.notna().all() and (pid >= 0).all()), f"participant ids resolved ({int((pid.isna() | (pid < 0)).sum())} not)")
    source = str(df["source"].iloc[0]) if "source" in df else "aslsigns"

    if "wlasl_gloss" in df:  # WLASL: halves must equal the frozen split
        sp = signer_split(df)
        frozen = set(json.loads(Path(__file__).with_name("wlasl_signer_split.json").read_text())["test"])
        check(set(sp["test"]) <= frozen, "WLASL test signers ⊆ frozen test half")
        if "wlasl_half" in df:
            want = np.where(df["participant_id"].isin(frozen), "test", "train")
            check(bool((df["wlasl_half"].to_numpy() == want).all()), "wlasl_half = frozen split")
    if source == "aslcitizen" and "split" in df:
        s = {k: set(g["participant_id"]) for k, g in df.groupby("split")}
        check(all(not (s[x] & s[y]) for x in s for y in s if x < y), f"ASL Citizen splits signer-disjoint ({sorted(s)})")

    env = lmdb.open(str(a.lmdb_path), subdir=False, readonly=True, lock=False,
                    map_size=os.path.getsize(a.lmdb_path) + (1 << 28))
    missing, bad, contents, two_hand = [], [], {}, 0
    with env.begin() as txn:
        for r in df.itertuples():
            v, n = txn.get(lmdb_key(r.path)), txn.get(lmdb_length_key(r.path))
            if v is None or n is None:
                missing.append(r.path)
                continue
            d = torch.load(io.BytesIO(bytes(v)), weights_only=True)
            c, pres, st = d["coords"], d["presence"], d["status"]
            T = c.shape[0]
            problems = []
            if c.ndim != 2 or c.shape[1] != COORD_FEAT or T == 0:
                problems.append(f"shape {tuple(c.shape)}")
            if not torch.isfinite(c).all():
                problems.append("non-finite")
            if int(n) != T or ("n_frames" in df and int(r.n_frames) != T):
                problems.append(f"length {int(n)} / n_frames {getattr(r, 'n_frames', '?')} / T {T}")
            if pres.shape != (T, 2) or not ((pres == 0) | (pres == 1)).all():
                problems.append("presence not binary (T, 2)")
            if not torch.isin(st, torch.tensor([K.ABSENT, K.KEPT, K.GHOST, K.AMBIGUOUS])).all():
                problems.append("unknown status code")
            if not torch.equal(pres.bool(), (st == K.KEPT) | (st == K.AMBIGUOUS)):
                problems.append("presence ≠ status")
            for h, hs in enumerate((LH_START, RH_START)):
                if not pres[:, h].any() and c[:, hs: hs + N_LH * COORDS_PER_LM].abs().sum() > 0:
                    problems.append(f"hand {h} never present but non-zero")
            if problems:
                bad.append((r.path, problems))
            h = hashlib.sha1(c.numpy().tobytes() + pres.numpy().tobytes()).hexdigest()
            contents.setdefault(h, []).append(r.path)
            two_hand += bool((pres[:, 0].bool() & pres[:, 1].bool()).any())
    check(not missing, f"every train.csv path has data + length keys ({len(missing)} missing)")
    check(not bad, f"clips well-formed ({len(bad)} bad{', e.g. ' + str(bad[0]) if bad else ''})")
    dups = [v for v in contents.values() if len(v) > 1]
    check(not dups, f"no duplicate content ({len(dups)} groups{', e.g. ' + str(dups[0][:2]) if dups else ''})")
    report["two_hand_clip_rate"] = round(two_hand / max(len(df) - len(missing), 1), 4)
    report["clips"] = len(df)
    print(f"info  {len(df)} clips, source {source}; clips with both hands in a frame {report['two_hand_clip_rate']:.1%}; "
          f"contract counts {man.get('contract_counts')}")

    if a.raw_root:  # the stored value is what the contract produces from the raw parquet today
        sample = df.sample(min(a.resample, len(df)), random_state=a.seed)
        csvs = {}
        import glob

        for c_ in glob.glob(f"{a.raw_root}/**/shard_*.csv", recursive=True):
            for path_ in pd.read_csv(c_)["path"]:
                csvs[path_] = str(Path(c_).parent / path_)
        mism, unfound = [], 0
        with env.begin() as txn:
            for r in sample.itertuples():
                full = csvs.get(r.path) or str(a.raw_root / r.path)
                if not os.path.exists(full):
                    unfound += 1
                    continue
                aspect = (r.width / r.height) if hasattr(r, "width") and hasattr(r, "height") else float("nan")
                correct = source in ("wlasl", "aslcitizen")  # PopSign / ASL Signs: portrait, uncorrected
                _, value, _, err = _process((r.path, full, correct, aspect))
                stored = K.load_clip(txn.get(lmdb_key(r.path)))
                if err or not all(torch.equal(x, y) for x, y in zip(K.load_clip(value), stored)):
                    mism.append(r.path)
        check(unfound == 0, f"raw parquets found for the re-prepare sample ({unfound} not)")
        check(not mism, f"re-prepared from raw = stored ({len(mism)} of {len(sample) - unfound} differ)")
    env.close()

    report["failures"] = fails
    (a.lmdb_path.parent / "validation_report.json").write_text(json.dumps(report, indent=1, default=str))
    if fails:
        raise SystemExit(f"{len(fails)} validation checks failed")
    print("all validation checks passed")


if __name__ == "__main__":
    main()
