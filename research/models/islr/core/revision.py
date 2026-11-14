"""The code revision a build, validation or training run executes: the git commit of this checkout
(the Kaggle runner clones the pushed commit) and whether tracked files differ from it."""

import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]


def code_revision() -> dict:
    """{"commit": sha or None (not a git checkout), "dirty": tracked files modified}."""
    def git(*a: str) -> str | None:
        r = subprocess.run(["git", "-C", str(_ROOT), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    return {"commit": commit, "dirty": bool(git("status", "--porcelain", "--untracked-files=no")) if commit else None}
