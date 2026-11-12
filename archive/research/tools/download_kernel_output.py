"""
Stream a Kaggle notebook's saved output to disk.

Same API call as `kaggle kernels output`, but safe for multi-GB files. The
kaggle 1.8.2 CLI buffers each file fully in RAM (`response.content`, ~2× the
file size at peak while requests joins the chunks) — enough to OOM-kill a pod
on a 15–20 GB LMDB — and it ignores pagination of the file list. This script:

  - streams in 8 MB chunks (flat memory), with a progress bar
  - follows next_page_token so no output file is skipped
  - checks free disk space before each file
  - resumes interrupted downloads (HTTP Range) and retries with fresh signed URLs
  - verifies each file's final size against Content-Length

Credentials: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY
(the kaggle package authenticates on import).

Usage:
  python research/tools/download_kernel_output.py shravnchandr/build-islt-lmdb \\
      -p data/.kaggle_download --include '*.mdb' train.csv --exclude lock.mdb
"""

import argparse
import fnmatch
import os
import shutil
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

_CHUNK = 8 << 20  # 8 MB
_RETRIES = 5


def list_output_files(kernel: str) -> dict[str, str]:
    """Return {file_name: signed_url} for every file in the kernel's latest output."""
    import kaggle  # authenticates on import; raises if no credentials
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiListKernelSessionOutputRequest,
    )

    owner, slug = kernel.split("/", 1)
    files: dict[str, str] = {}
    token = None
    with kaggle.api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            if token:
                req.page_token = token
            resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
            for f in resp.files or []:
                files[f.file_name] = f.url
            token = resp.next_page_token
            if not token:
                return files


def _selected(name: str, include: list[str], exclude: list[str]) -> bool:
    base = os.path.basename(name)
    if any(fnmatch.fnmatch(base, p) for p in exclude):
        return False
    return not include or any(fnmatch.fnmatch(base, p) for p in include)


def download(kernel: str, url: str, name: str, dest: Path) -> Path:
    """Stream one file to dest/name, resuming from a .part file across retries."""
    out = (dest / name).resolve()
    if dest.resolve() not in out.parents:
        raise ValueError(f"refusing to write outside {dest}: {name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")

    for attempt in range(1, _RETRIES + 1):
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=(30, 300)) as r:
                if r.status_code == 416:  # range past end: .part already complete
                    break
                r.raise_for_status()
                if done and r.status_code != 206:  # server ignored Range: restart
                    done = 0
                total = done + int(r.headers.get("Content-Length", 0))
                free = shutil.disk_usage(out.parent).free
                if total - done > free:
                    raise SystemExit(
                        f"Not enough disk for {name}: need {(total - done) / 1e9:.1f} GB, "
                        f"have {free / 1e9:.1f} GB free"
                    )
                with open(part, "ab" if done else "wb") as fh, tqdm(
                    total=total or None, initial=done, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=name, miniters=1,
                ) as bar:
                    for chunk in r.iter_content(_CHUNK):
                        fh.write(chunk)
                        bar.update(len(chunk))
            if total and part.stat().st_size != total:
                raise IOError(f"size mismatch: {part.stat().st_size} != {total}")
            break
        except (requests.RequestException, IOError) as e:
            if attempt == _RETRIES:
                raise
            wait = 10 * attempt
            print(f"  {name}: {e} — retry {attempt}/{_RETRIES - 1} in {wait}s", file=sys.stderr)
            time.sleep(wait)
            url = list_output_files(kernel).get(name, url)  # signed URLs can expire
    os.replace(part, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kernel", help="<owner>/<kernel-slug>")
    p.add_argument("-p", "--path", required=True, help="destination directory")
    p.add_argument("--include", nargs="*", default=[], help="basename globs to download (default: all)")
    p.add_argument("--exclude", nargs="*", default=[], help="basename globs to skip")
    args = p.parse_args()

    if args.kernel.count("/") != 1:
        p.error("kernel must be <owner>/<kernel-slug>")
    dest = Path(args.path)
    dest.mkdir(parents=True, exist_ok=True)

    files = list_output_files(args.kernel)
    if not files:
        raise SystemExit(
            f"{args.kernel} has no saved output. Run it with 'Save Version' "
            "(interactive-session output is not kept)."
        )
    chosen = {n: u for n, u in files.items() if _selected(n, args.include, args.exclude)}
    print(f"{args.kernel}: {len(files)} output file(s), downloading {len(chosen)}:")
    for n in sorted(files):
        print(f"  {'↓' if n in chosen else '·'} {n}")
    if not chosen:
        raise SystemExit("no output files match --include/--exclude")

    for name, url in sorted(chosen.items()):
        out = download(args.kernel, url, name, dest)
        print(f"  saved {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
