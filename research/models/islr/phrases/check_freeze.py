"""Check an experiment_spec.json (the reference of a frozen comparison, or any run of it) against
core/data/frozen_inputs.json: code revision committed and clean, contract / cache version, every
dataset release it read (content, train.csv and sign-map digests), the vocabulary (digest of the
canonical 250 + WLASL-all map), the initial checkpoint bytes, seed and validation fold, and that the
recorded runtime is a GPU; with --profile, also the full recipe and the exact list of extra sources of
that comparison run (phrases05 / phrases06 / phrases07, from the recipes in frozen_inputs.json): the
profile's arguments are run through phrases.train's own parser (training stubbed out), so every
argument the profile does not set must equal the trainer's default. Path arguments are compared by the
release / init / vocab digests instead. Check every comparison run's own spec, not only the reference:
a spec only shows the releases that run read. Exits non-zero on any mismatch.

  PYTHONPATH=research/models python -m islr.phrases.check_freeze --spec <experiment_spec.json> \
      --vocab-map <accepted-wlasl-all>/sign_to_prediction_index_map.json --init <best_phase1.pth> \
      --commit 2de95c1 --profile phrases05
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

FROZEN = Path(__file__).resolve().parents[1] / "core" / "data" / "frozen_inputs.json"
# compared through digests (release / init / vocab) or free by design in a comparison
PATH_ARGS = {"data_dir", "lmdb_path", "init", "vocab", "extra_data_dir", "extra_lmdb_path", "out_dir",
             "frozen_spec", "allow_diff", "num_workers", "spec_only"}


def profile_recipe(recipes: dict, profile: str) -> tuple[dict, list]:
    """(full effective recipe, extra LMDB names in order) of a frozen profile, from phrases.train's parser."""
    import shlex
    import sys

    from . import train as T

    key = next(k for k in recipes if k.split()[0] == profile)
    argv = shlex.split(recipes["common"])[1:] + shlex.split(recipes[key]) + ["--out-dir", "x"]
    got = {}
    old_train, old_argv = T.train, sys.argv
    T.train, sys.argv = (lambda args: got.update(vars(args))), ["phrases.train", *argv]
    try:
        T.main()
    finally:
        T.train, sys.argv = old_train, old_argv
    lmdbs = [Path(x).name for x in got.get("extra_lmdb_path") or []]
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in got.items()}, lmdbs


def short(data: bytes) -> str:  # phrases.train._sha
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--frozen", type=Path, default=FROZEN)
    p.add_argument("--vocab-map", required=True, type=Path, help="the WLASL-all release's label map (the run's --vocab)")
    p.add_argument("--init", required=True, type=Path, help="local copy of the frozen initial checkpoint")
    p.add_argument("--commit", help="expected code revision (default: any clean commit containing the freeze record)")
    p.add_argument("--profile", choices=["phrases05", "phrases06", "phrases07", "phrases07-w50"],
                   help="also require this comparison run's full recipe and extra sources")
    a = p.parse_args()
    spec, fz = json.loads(a.spec.read_text()), json.loads(a.frozen.read_text())
    rel = fz["releases"]
    checks = []

    def check(ok: bool, msg: str) -> None:
        checks.append((bool(ok), msg))
        print(("ok    " if ok else "FAIL  ") + msg)

    code = spec.get("code") or {}
    check(not code.get("dirty"), f"code {str(code.get('commit'))[:7]} committed and clean")
    if a.commit:
        full = subprocess.run(["git", "rev-parse", "--verify", f"{a.commit}^{{commit}}"], capture_output=True, text=True).stdout.strip()
        check(code.get("commit") == full, f"code = {a.commit}")
    else:
        has = subprocess.run(["git", "merge-base", "--is-ancestor", "2de95c1", str(code.get("commit"))]).returncode == 0
        check(has, "code contains the freeze record (2de95c1)")
    check((spec.get("contract"), spec.get("cache_version")) == (rel["aslsigns"]["contract"], rel["aslsigns"]["cache_version"]),
          f"contract / cache {spec.get('contract')} / {spec.get('cache_version')}")

    def same_release(got: dict, name: str) -> bool:
        r = rel[name]
        return got == {"content": r["content_digest"], "csv": r["csv_digest"], "sign_map": r["sign_map_digest"], "lmdb": r["lmdb"]}

    check(same_release(spec["data"]["release"], "aslsigns"), f"main data = the frozen ASL Signs release {rel['aslsigns']['content_digest']}")
    by_lmdb = {r["lmdb"]: n for n, r in rel.items()}
    for e in spec.get("extra_data", []):
        name = by_lmdb.get((e.get("release") or {}).get("lmdb"))
        check(name is not None and same_release(e["release"], name),
              f"extra data {(e.get('release') or {}).get('lmdb')} = the frozen {name} release")
    vm = json.loads(a.vocab_map.read_text())
    check(hashlib.sha256(a.vocab_map.read_bytes()).hexdigest() == fz["vocabulary"]["wlasl_all_map_sha256"],
          "--vocab-map file = the frozen WLASL-all map")
    check(spec.get("vocab") == short(json.dumps(vm, sort_keys=True).encode()), "spec vocab = the WLASL-all map (canonical 250 first)")
    check(hashlib.sha256(a.init.read_bytes()).hexdigest() == fz["init"]["sha256"], "--init file = the frozen checkpoint bytes")
    check(spec.get("init") == short(a.init.read_bytes()), f"spec init = the frozen checkpoint ({spec.get('init')})")
    rc = spec.get("recipe", {})
    check(rc.get("seed") == fz["seeds"]["seed"] and rc.get("val_fold") == fz["splits"]["asl_signs_folds"]["val_fold"]
          and rc.get("val_phrases") == fz["seeds"]["val_phrases"], f"seed {rc.get('seed')}, val fold {rc.get('val_fold')}, val phrases {rc.get('val_phrases')}")
    if a.profile:
        same = subprocess.run(["git", "diff", "--quiet", str(code.get("commit")), "--", str(Path(__file__).with_name("train.py"))]).returncode == 0
        check(same, "phrases/train.py here = at the spec's commit (the profile is read through its parser)")
        want, lmdbs = profile_recipe(fz["recipes"], a.profile)
        got = spec.get("recipe", {})
        diff = {k: (want.get(k), got.get(k)) for k in sorted((set(want) | set(got)) - PATH_ARGS) if want.get(k) != got.get(k)}
        check(not diff, f"recipe = the frozen {a.profile} profile (all arguments but paths){'; differs: ' + str(diff) if diff else ''}")
        have = [(e.get("release") or {}).get("lmdb") for e in spec.get("extra_data", [])]
        check(have == lmdbs, f"extra sources {have} = {lmdbs} (in order)")
        if "research" in next(k for k in fz["recipes"] if k.split()[0] == a.profile):
            check(any(rel[n].get("research_only") for n in (by_lmdb.get(x) for x in have) if n), "research-only profile reads the research-only release")
    rt = spec.get("runtime") or {}
    check(rt.get("device") not in (None, "cpu") and rt.get("torch_cuda"), f"runtime on GPU: {rt}")
    check(bool(spec.get("val_set")), f"validation phrases digest {spec.get('val_set')}")
    bad = [m for ok, m in checks if not ok]
    if bad:
        raise SystemExit(f"{len(bad)} freeze checks failed")
    print(f"spec matches the freeze record ({len(checks)} checks)")


if __name__ == "__main__":
    main()
