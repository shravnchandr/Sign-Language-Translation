"""The canonical label map — part of every trained model's interface (output index → sign).

`asl_signs_vocab.json` is a byte copy of the Google ASL Signs competition's
`sign_to_prediction_index_map.json` (Kaggle competition `asl-signs`, downloaded 2026-11-14,
sha256 below; the app's labels.json has the same order). Its 250 assignments are immutable:
never edit, reorder or extend the file. A wider vocabulary (WLASL all glosses, `--vocab`) keeps
these 250 assignments and numbers its extra classes from 250 on.

  base_problems(m)      exact agreement with the canonical map
  extended_problems(m)  the canonical 250 first, extra classes 250..N-1
  load_base(path)       a builder's --sign-map (checked before any gloss is remapped) or any reader of the
                        base vocabulary: training, evaluation, export, review tools (exits otherwise)
  load_map(path, extended=True)  a reader that may also take a permitted extension (cross_dataset)
"""

import hashlib
import json
from pathlib import Path

CANONICAL_PATH = Path(__file__).with_name("asl_signs_vocab.json")
CANONICAL_SHA256 = "1fe747c2f44c68dbb396947e35193c96d363f3dede0be8defa5e08546400bf5d"
N_BASE = 250


def canonical() -> dict:
    data = CANONICAL_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != CANONICAL_SHA256:
        raise SystemExit(f"{CANONICAL_PATH} has changed: its assignments are immutable (restore it from git)")
    return json.loads(data)


def map_problems(path: Path) -> list[str]:
    """The file must map distinct sign names onto exactly the integers 0..N-1."""
    pairs = json.loads(Path(path).read_text(), object_pairs_hook=lambda kv: kv)
    if not isinstance(pairs, list):
        return ["not a JSON object"]
    names, idx = [k for k, _ in pairs], [v for _, v in pairs]
    out = []
    if len(set(names)) != len(names):
        out.append(f"repeated sign names {sorted({n for n in names if names.count(n) > 1})[:5]}")
    if not all(type(v) is int for v in idx):
        out.append(f"non-integer indices, e.g. {next(v for v in idx if type(v) is not int)!r}")
    elif sorted(idx) != list(range(len(idx))):
        dup = sorted({v for v in idx if idx.count(v) > 1})
        out.append(f"indices not exactly 0..{len(idx) - 1} (repeated {dup[:5]}, "
                   f"missing {sorted(set(range(len(idx))) - set(idx))[:5]})")
    return out


def _base_diff(m: dict) -> list[str]:
    c = canonical()
    diff = sorted(k for k in c if m.get(k) != c[k])
    return [f"{len(diff)} of the {N_BASE} canonical assignments differ, e.g. "
            + ", ".join(f"{k}: {m.get(k)} (canonical {c[k]})" for k in diff[:3])] if diff else []


def base_problems(m: dict) -> list[str]:
    if len(m) != N_BASE:
        return [f"{len(m)} labels, the base vocabulary has {N_BASE}"]
    return _base_diff(m)


def extended_problems(m: dict) -> list[str]:
    out = _base_diff(m)
    extra = sorted(v for k, v in m.items() if k not in canonical())
    if extra != list(range(N_BASE, N_BASE + len(extra))):
        out.append(f"extra classes are not exactly {N_BASE}..{N_BASE + len(extra) - 1}")
    return out


def check_map(path: Path, extended: bool = False) -> list[str]:
    """All problems of a map file: structure, then the canonical assignments."""
    return map_problems(path) or (extended_problems if extended else base_problems)(json.loads(Path(path).read_text()))


def load_map(path: Path, extended: bool = False) -> dict:
    problems = check_map(path, extended)
    if problems:
        raise SystemExit(f"{path}: not the canonical ASL Signs label map ({CANONICAL_PATH.name})"
                         + (" or an extension of it" if extended else "") + ": " + "; ".join(problems))
    return json.loads(Path(path).read_text())


def load_base(path: Path) -> dict:
    return load_map(path)
