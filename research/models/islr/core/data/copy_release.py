"""Copy a built release (another notebook's output, read-only) to a writable folder so it can be
validated again under a newer VALIDATION_POLICY without rebuilding: every file except the old
validation_report.json. The content digest is recomputed by validate, so the copy is the same release.

  python scripts/kaggle_run.py push revalidate-popsign --bare --no-copy --kernel <user>/release-popsign ... -- \
      islr.core.data.copy_release --src {KERNEL_release_popsign} --dst {OUT} :: \
      islr.core.data.validate --data-dir {OUT} --lmdb-path {OUT}/popsign.lmdb.mdb --raw-root {INPUT}
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    a = p.parse_args()
    src = a.src / "out" if (a.src / "out" / "build_manifest.json").exists() else a.src  # kaggle_run writes {OUT} = out/
    if not (src / "build_manifest.json").exists():
        raise SystemExit(f"{a.src}: no build_manifest.json — not a release")
    a.dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name not in ("validation_report.json", "run.log"):
            shutil.copy2(f, a.dst / f.name)
            print(f"copied {f.name} ({f.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
