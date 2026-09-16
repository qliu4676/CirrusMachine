"""Validated NPZ caches for deterministic synthesis targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


CACHE_SCHEMA_VERSION = 1


def target_fingerprint(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> str:
    """Hash input arrays and configuration metadata for cache invalidation."""

    digest = hashlib.sha256()
    payload = {"schema": CACHE_SCHEMA_VERSION, **metadata}
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def cached_target_arrays(
    path: str | Path,
    fingerprint: str,
    compute: Callable[[], Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], bool]:
    """Load a matching cache or compute and atomically replace it.

    Returns the arrays and a boolean that is true on a cache hit.
    """

    cache_path = Path(path)
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                saved_fingerprint = str(saved["fingerprint"].item())
                if saved_fingerprint == fingerprint:
                    return {
                        key: np.asarray(saved[key])
                        for key in saved.files
                        if key != "fingerprint"
                    }, True
        except (OSError, ValueError, KeyError):
            pass

    arrays = {key: np.asarray(value) for key, value in compute().items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, fingerprint=np.asarray(fingerprint), **arrays)
    temporary.replace(cache_path)
    return arrays, False
