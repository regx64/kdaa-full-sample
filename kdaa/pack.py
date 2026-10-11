"""Session shards (PLAN.md §3): one `mel_packed/<session_id>.npz` per label CSV instead of one .npy per clip.

Shard arrays: logmel (N, n_mels, frames) float32 · clip_id (N,) str · param_hash · mel_spec_version · session_id.
Clips whose .npy is missing (meligation skipped them, with the reason in mel/_skipped_<session>.csv) are left out.
The reader is classism's `src.dataset` (shard layout in its module docstring); tests/test_pack.py checks the two agree.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .util import read_json


class ShardError(ValueError):
    pass


def csv_clip_ids(csv_path: Path) -> list[str]:
    """clip ids in CSV order, named the way classism's dataset.load_samples names them."""
    sid = csv_path.stem
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return [(row.get("clip_id") or "").strip() or f"{sid}_{i:05d}" for i, row in enumerate(csv.DictReader(f))]


@dataclass
class PackResult:
    session_id: str
    rows: int
    packed: int
    missing: int
    path: Path


def pack_session(work: Path, session_id: str, *, remove_clip_npy: bool = False) -> PackResult:
    meta = read_json(work / "mel" / "_meta.json")
    if not meta:
        raise ShardError(f"{work / 'mel' / '_meta.json'} missing: run the mel stage first")
    ids = csv_clip_ids(work / "labels" / f"{session_id}.csv")
    shape = (int(meta["n_mels"]), int(meta["frames"]))
    have, arrays = [], []
    for cid in ids:
        p = work / "mel" / f"{cid}.npy"
        if not p.exists():
            continue
        a = np.load(p, allow_pickle=False)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.shape != shape or not np.issubdtype(a.dtype, np.floating):
            raise ShardError(f"{p}: shape {a.shape} / dtype {a.dtype}, expected float {shape}")
        have.append(cid)
        arrays.append(a.astype(np.float32, copy=False))
    logmel = np.stack(arrays) if arrays else np.zeros((0, *shape), np.float32)
    if not np.isfinite(logmel).all():
        raise ShardError(f"session {session_id}: log-mel contains NaN/Inf")
    out_dir = work / "mel_packed"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{session_id}.npz"
    tmp = out_dir / f"{session_id}.npz.tmp"
    with open(tmp, "wb") as fh:  # np.savez appends .npz to a bare path; a handle keeps the name
        np.savez(fh, logmel=logmel, clip_id=np.array(have, dtype=str), param_hash=str(meta["param_hash"]),
                 mel_spec_version=str(meta["mel_spec_version"]), session_id=session_id)
    os.replace(tmp, final)
    if remove_clip_npy:
        for cid in have:
            (work / "mel" / f"{cid}.npy").unlink(missing_ok=True)
    return PackResult(session_id, len(ids), len(have), len(ids) - len(have), final)
