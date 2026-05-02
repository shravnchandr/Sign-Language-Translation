"""
LMDB cache key helpers — isolated from the training stack so build_lmdb.py
can import them without pulling in torch, sklearn, or any other heavy dep.
"""
import hashlib
from ..config import ALL_COLUMNS

# Digest changes whenever ALL_COLUMNS changes (face selection, depth flag, etc.)
# or when preprocessing normalization logic changes (_NORM_VERSION bump).
_NORM_VERSION = "v2_fallback"  # bump when normalize_values logic in preprocessing.py changes
CACHE_VERSION = hashlib.md5(("|".join(ALL_COLUMNS) + "|" + _NORM_VERSION).encode()).hexdigest()[:8]


def lmdb_key(path: str) -> bytes:
    return f"{CACHE_VERSION}:{path}".encode()


def lmdb_length_key(path: str) -> bytes:
    return f"{CACHE_VERSION}:len:{path}".encode()
