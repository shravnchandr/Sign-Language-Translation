"""Validation gate for a built LMDB: run on every dataset before it is used for training or scoring.

  PYTHONPATH=research/models python -m islr.core.data.validate --data-dir /tmp/popsign \
      --lmdb-path /tmp/popsign/popsign.lmdb.mdb [--raw-root {INPUT} --resample 200]

Exits non-zero when any hard check fails; writes validation_report.json into --data-dir. Training
(`islr.phrases.train`) refuses a dataset without a passing report whose digests match the data it
opens (LMDB content, train.csv, sign map), and one validated without --raw-root.
Hard checks:
  - build_manifest.json present, built with this contract and cache version; build errors within
    the allowance recorded at build time; dataset type = the train.csv schema; content digest
    recomputed from the LMDB = the manifest's (every stored field with its dtype and shape, and the
    length key); train.csv digest = the manifest's (annotations); sign map digest = the manifest's
  - sign map: distinct names onto exactly the integers 0..N-1, equal to the canonical ASL Signs map
    (vocab.py; WLASL all glosses: the canonical 250 unchanged, extra classes from 250)
  - coverage: selected = listed + missing (missing within the build's allowance; the selection
    recorded by the extraction or re-run by scripts/extraction_inventory.py), listed = filtered +
    build errors + excluded duplicates + written, written = train.csv rows
  - preparation parameters recorded: prep_aspect per clip (ASL Signs: the manifest's fixed 0.75)
  - every train.csv path has a data + length key; stored length = clip length = n_frames
  - every clip: the current record layout (contract.RECORD_KEYS, contract.RECORD_DTYPES); finite, (T > 0, COORD_FEAT);
    presence binary and = status KEPT; status codes valid; assignment_uncertain binary (T,) and set
    only in copy frames (one slot KEPT, the other GHOST); a hand never present has an all-zero block
  - labels in the sign map; participant ids resolved (no NaN / -1)
  - WLASL: signer halves = the frozen wlasl_signer_split.json (and wlasl_half, when present);
    ASL Citizen: official train / val / test signers disjoint
  - no two clips with identical stored content
  - --raw-root: a random sample re-prepared from its raw parquet is byte-identical to the stored
    value — every stored field (contract.RECORD_KEYS; the LMDB is what the contract produces today)
Reported (not failed): two-hand clip rate, copy / assignment-uncertain / pose-less frame totals, clips in
which no hand is ever present (low information; listed with their split, kept and scored as-is).
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

from ..revision import code_revision
from ..config import COORD_FEAT, COORDS_PER_LM, LH_START, N_LH, RH_START
from . import contract as K
from ._cache_keys import CACHE_VERSION, lmdb_key, lmdb_length_key
from .build_wlasl_lmdb import _process, signer_split
from .vocab import check_map


# Bump whenever a check is added or tightened: reports written under an older policy no longer
# admit a dataset (it must be validated again), so a release cannot pass on yesterday's checks.
VALIDATION_POLICY = 5  # 1: 33b6971 (no coverage / prep_aspect checks); 2: e765dd4+ (coverage, prep_aspect, no-hand report); 3: contract v4 (copy rule, assignment_uncertain); 4: sign map = bijection onto 0..N-1 + fingerprinted, stored dtypes, digest over dtype / shape / length key; 5: sign map = the canonical 250 (WLASL all glosses: canonical 250 first, extras from 250)


def require_accepted(data_dir, lmdb_path) -> dict:
    """Acceptance for training and scoring: a dataset is used only with a passing validation report
    (written by this module, with --raw-root) whose digests match what is opened now: the train.csv
    in data_dir and every clip it lists in the LMDB at lmdb_path. Exits otherwise. Returns
    {"content", "csv", "lmdb"} — the release an experiment spec records."""
    from .dataset import _open_lmdb_env  # the datasets' own handle (an env opens once per process)

    rep_path = Path(data_dir) / "validation_report.json"
    if not rep_path.exists():
        raise SystemExit(f"{data_dir}: no validation_report.json — run islr.core.data.validate --raw-root first")
    rep = json.loads(rep_path.read_text())
    csv_path = Path(data_dir) / "train.csv"
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()[:16]
    map_digest = hashlib.sha256((Path(data_dir) / "sign_to_prediction_index_map.json").read_bytes()).hexdigest()[:16]
    problems = [m for ok, m in [
        (rep.get("policy") == VALIDATION_POLICY,
         f"validated under policy {rep.get('policy')}, current is {VALIDATION_POLICY} — run validate again"),
        (rep.get("passed") is True and not rep.get("failures"), f"validation failed: {rep.get('failures')}"),
        (rep.get("raw_resampled", 0) > 0, "validated without --raw-root (stored clips never compared with raw)"),
        ((rep.get("contract"), rep.get("cache_version")) == (K.CONTRACT_VERSION, CACHE_VERSION),
         f"validated under contract {rep.get('contract')} / cache {rep.get('cache_version')}"),
        (rep.get("lmdb") == Path(lmdb_path).name, f"report is for {rep.get('lmdb')}, run reads {Path(lmdb_path).name}"),
        (rep.get("csv_digest") == csv_digest, f"train.csv changed since validation ({csv_digest} ≠ {rep.get('csv_digest')})"),
        (rep.get("sign_map_digest") == map_digest,
         f"sign map changed since validation ({map_digest} ≠ {rep.get('sign_map_digest')})"),
    ] if not ok]
    if not problems:
        try:
            content = K.content_digest(_open_lmdb_env(str(Path(lmdb_path).resolve())), pd.read_csv(csv_path)["path"].tolist())
        except ValueError as e:  # a record that no longer loads (layout, dtype) or a missing length key
            content = f"unreadable ({e})"
        if content != rep.get("content_digest"):
            problems.append(f"LMDB content {content} ≠ validated {rep.get('content_digest')}")
    if problems:
        raise SystemExit(f"{data_dir}: not an accepted dataset — " + "; ".join(problems))
    print(f"Validated dataset {data_dir}: content {rep['content_digest']}, train.csv {csv_digest}, "
          f"{rep.get('clips')} clips ({rep.get('raw_resampled')} re-prepared from raw)")
    return {"content": rep["content_digest"], "csv": csv_digest, "sign_map": map_digest, "lmdb": Path(lmdb_path).name}


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
    check(man.get("errors", 1) <= man.get("allow_errors", 0),
          f"build errors {man.get('errors')} within the allowance it was built with ({man.get('allow_errors', 0)})")
    cov = man.get("coverage") or {}
    check(cov.get("selected") is not None, f"selection recorded (coverage {({k: v for k, v in cov.items() if k != 'missing_paths'}) or 'absent'})")
    if cov.get("selected") is not None:
        check(cov["selected"] == cov.get("listed", -1) + cov.get("missing", 0) and cov.get("missing", 0) <= cov.get("allow_missing", 0),
              f"selected {cov['selected']} = listed {cov.get('listed')} + missing {cov.get('missing')} (allowed {cov.get('allow_missing', 0)})")
        check(cov.get("listed") == sum(cov.get(k, -10**9) for k in ("filtered", "build_errors", "excluded_duplicates", "written")),
              f"listed {cov.get('listed')} = filtered {cov.get('filtered')} + errors {cov.get('build_errors')} + "
              f"excluded {cov.get('excluded_duplicates')} + written {cov.get('written')}")
    report["manifest"] = {k: man.get(k) for k in ("version", "dataset", "aspect_correction", "shards", "clips_listed",
                                                   "clips_written", "errors", "allow_errors", "content_digest", "csv_digest",
                                                   "contract_counts", "code")}

    csv_digest = hashlib.sha256((a.data_dir / "train.csv").read_bytes()).hexdigest()[:16]
    check(csv_digest == man.get("csv_digest"),
          f"train.csv digest {csv_digest} = manifest {man.get('csv_digest')} (labels / signers / splits unchanged since the build)")
    df = pd.read_csv(a.data_dir / "train.csv")
    map_path = a.data_dir / "sign_to_prediction_index_map.json"
    map_digest = hashlib.sha256(map_path.read_bytes()).hexdigest()[:16]
    check(map_digest == man.get("sign_map_digest"), f"sign map digest {map_digest} = manifest {man.get('sign_map_digest')}")
    extended = "wlasl_half" in df and "wlasl_gloss" in df  # WLASL --all-glosses: extra labels after the 250
    bad_map = check_map(map_path, extended)
    check(not bad_map, f"sign map = the canonical {'250 + extras from 250' if extended else '250'} ({'; '.join(bad_map) or 'ok'})")
    sign_map = json.loads(map_path.read_text())
    check(df["sign"].isin(sign_map).all(), f"labels in the sign map ({int((~df['sign'].isin(sign_map)).sum())} not)")
    pid = pd.to_numeric(df["participant_id"], errors="coerce")
    check(bool(pid.notna().all() and (pid >= 0).all()), f"participant ids resolved ({int((pid.isna() | (pid < 0)).sum())} not)")
    # dataset type from the manifest (WLASL's `source` column holds its web sources), else the schema
    schema = ("wlasl" if "wlasl_gloss" in df else "aslcitizen" if "aslc_gloss" in df
              else "popsign" if "popsign_gloss" in df else "aslsigns")
    dataset = man.get("dataset") or schema
    check(dataset == schema, f"manifest dataset {man.get('dataset')!r} matches the train.csv schema ({schema})")
    correct = man.get("aspect_correction", dataset in ("wlasl", "aslcitizen"))
    check(cov.get("written") == len(df), f"train.csv rows {len(df)} = clips written {cov.get('written')}")
    if dataset == "aslsigns":
        prep_aspect = pd.Series(man.get("prep_aspect", np.nan), index=df.index, dtype=float)
    else:
        prep_aspect = pd.to_numeric(df.get("prep_aspect", pd.Series(np.nan, index=df.index)), errors="coerce")
    check(bool(prep_aspect.notna().all() and (prep_aspect > 0).all()),
          f"preparation aspect recorded for every clip ({int(prep_aspect.isna().sum())} missing)")

    if "wlasl_gloss" in df:  # WLASL: halves must equal the frozen split
        sp = signer_split(df)
        frozen = set(json.loads(Path(__file__).with_name("wlasl_signer_split.json").read_text())["test"])
        check(set(sp["test"]) <= frozen, "WLASL test signers ⊆ frozen test half")
        if "wlasl_half" in df:
            want = np.where(df["participant_id"].isin(frozen), "test", "train")
            check(bool((df["wlasl_half"].to_numpy() == want).all()), "wlasl_half = frozen split")
    if dataset == "aslsigns":  # validation folds are frozen (dataset.signer_fold_split)
        folds = json.loads(Path(__file__).with_name("asl_signs_folds.json").read_text())["folds"]
        unknown = sorted(set(df["participant_id"].astype(str)) - set(folds))
        check(not unknown, f"every ASL Signs signer in the frozen folds ({len(unknown)} not, e.g. {unknown[:3]})")
    if dataset == "aslcitizen" and "split" in df:
        s = {k: set(g["participant_id"]) for k, g in df.groupby("split")}
        check(all(not (s[x] & s[y]) for x in s for y in s if x < y), f"ASL Citizen splits signer-disjoint ({sorted(s)})")

    env = lmdb.open(str(a.lmdb_path), subdir=False, readonly=True, lock=False,
                    map_size=os.path.getsize(a.lmdb_path) + (1 << 28))
    missing, bad, contents, two_hand, no_hand = [], [], {}, 0, []
    with env.begin() as txn:
        for r in df.itertuples():
            v, n = txn.get(lmdb_key(r.path)), txn.get(lmdb_length_key(r.path))
            if v is None or n is None:
                missing.append(r.path)
                continue
            try:
                d = K.load_record(v)
            except ValueError as e:
                bad.append((r.path, [str(e)]))
                continue
            c, pres, st, unc = d["coords"], d["presence"], d["status"], d["assignment_uncertain"]
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
            if st.shape != (T, 2) or not torch.isin(st, torch.tensor([K.ABSENT, K.KEPT, K.GHOST])).all():
                problems.append("unknown status code")
            if not torch.equal(pres.bool(), st == K.KEPT):
                problems.append("presence ≠ status")
            copy = ((st == K.KEPT) & (st.flip(1) == K.GHOST)).any(1)
            if unc.shape != (T,) or not ((unc == 0) | (unc == 1)).all() or (unc.bool() & ~copy).any():
                problems.append("assignment_uncertain not binary (T,) or outside a copy frame")
            for h, hs in enumerate((LH_START, RH_START)):
                if not pres[:, h].any() and c[:, hs: hs + N_LH * COORDS_PER_LM].abs().sum() > 0:
                    problems.append(f"hand {h} never present but non-zero")
            if problems:
                bad.append((r.path, problems))
            h = hashlib.sha1(c.numpy().tobytes() + pres.numpy().tobytes()).hexdigest()
            contents.setdefault(h, []).append(r.path)
            two_hand += bool((pres[:, 0].bool() & pres[:, 1].bool()).any())
            if not pres.any():
                no_hand.append(r.path)
    check(not missing, f"every train.csv path has data + length keys ({len(missing)} missing)")
    check(not bad, f"clips well-formed ({len(bad)} bad{', e.g. ' + str(bad[0]) if bad else ''})")
    dups = [v for v in contents.values() if len(v) > 1]
    check(not dups, f"no duplicate content ({len(dups)} groups{', e.g. ' + str(dups[0][:2]) if dups else ''})")
    digest = K.content_digest(env, [p_ for p_ in df["path"] if p_ not in set(missing)])
    check(digest == man.get("content_digest"), f"content digest {digest} = manifest {man.get('content_digest')}")
    report["two_hand_clip_rate"] = round(two_hand / max(len(df) - len(missing), 1), 4)
    report["clips"] = len(df)
    split_col = next((c for c in ("wlasl_half", "split") if c in df), None)
    nh = df[df["path"].isin(no_hand)]
    report["no_hand_clips"] = {"count": len(nh), "by_split": nh[split_col].value_counts().to_dict() if split_col else None,
                               "paths": nh["path"].tolist()}
    print(f"info  clips with no hand ever present (low information, kept and scored as-is): {len(nh)}"
          + (f" by {split_col} {report['no_hand_clips']['by_split']}" if split_col and len(nh) else ""))
    print(f"info  {len(df)} clips, dataset {dataset} (aspect correction {correct}); clips with both hands in a frame {report['two_hand_clip_rate']:.1%}; "
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
                _, value, _, err = _process((r.path, full, correct, float(prep_aspect[r.Index])))
                stored = K.load_record(txn.get(lmdb_key(r.path)))
                if err or not all(torch.equal(K.load_record(value)[k], stored[k]) for k in K.RECORD_KEYS):
                    mism.append(r.path)
        check(unfound == 0, f"raw parquets found for the re-prepare sample ({unfound} not)")
        check(not mism, f"re-prepared from raw = stored ({len(mism)} of {len(sample) - unfound} differ)")
        report["raw_resampled"] = len(sample) - unfound
    env.close()

    report.update({"policy": VALIDATION_POLICY, "failures": fails, "passed": not fails, "lmdb": a.lmdb_path.name, "content_digest": digest,
                   "csv_digest": csv_digest, "sign_map_digest": map_digest, "contract": K.CONTRACT_VERSION, "cache_version": CACHE_VERSION,
                   "raw_resampled": report.get("raw_resampled", 0), "code": code_revision()})
    (a.data_dir / "validation_report.json").write_text(json.dumps(report, indent=1, default=str))
    if fails:
        raise SystemExit(f"{len(fails)} validation checks failed")
    print("all validation checks passed")


if __name__ == "__main__":
    main()
