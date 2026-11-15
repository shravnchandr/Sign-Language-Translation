"""Check an experiment_spec.json (the reference of a frozen comparison, or any run of it) against
core/data/frozen_inputs.json: code revision committed and clean, contract / cache version, every
dataset release it read (content, train.csv and sign-map digests), the vocabulary (digest of the
canonical 250 + WLASL-all map), the initial checkpoint bytes, seed and validation fold, and that the
recorded runtime is a GPU. Exits non-zero on any mismatch.

  PYTHONPATH=research/models python -m islr.phrases.check_freeze --spec <experiment_spec.json> \
      --vocab-map <accepted-wlasl-all>/sign_to_prediction_index_map.json --init <best_phase1.pth> [--commit 2de95c1]
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

FROZEN = Path(__file__).resolve().parents[1] / "core" / "data" / "frozen_inputs.json"


def short(data: bytes) -> str:  # phrases.train._sha
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--frozen", type=Path, default=FROZEN)
    p.add_argument("--vocab-map", required=True, type=Path, help="the WLASL-all release's label map (the run's --vocab)")
    p.add_argument("--init", required=True, type=Path, help="local copy of the frozen initial checkpoint")
    p.add_argument("--commit", help="expected code revision (default: any clean commit containing the freeze record)")
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
    rt = spec.get("runtime") or {}
    check(rt.get("device") not in (None, "cpu") and rt.get("torch_cuda"), f"runtime on GPU: {rt}")
    check(bool(spec.get("val_set")), f"validation phrases digest {spec.get('val_set')}")
    bad = [m for ok, m in checks if not ok]
    if bad:
        raise SystemExit(f"{len(bad)} freeze checks failed")
    print(f"spec matches the freeze record ({len(checks)} checks)")


if __name__ == "__main__":
    main()
