import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
REPOS = {name: ROOT / "third_party" / d for name, d in
         (("sound_autodaq", "sound-autodaq"), ("meligation", "meligation"), ("classism", "classism"))}
sys.path.insert(0, str(ROOT))

if any(not p.exists() or not any(p.iterdir()) for p in REPOS.values()):
    raise RuntimeError("third_party/ is empty: run `git submodule update --init`")

HAS_TORCH = importlib.util.find_spec("torch") is not None


def classism_hash() -> str:
    from kdaa.mel_shim import load_classism_melio
    return load_classism_melio(REPOS["classism"]).REGIME.param_hash()


def make_work(root: Path, sessions: dict[str, list[str]], *, param_hash: str | None = None, missing=()) -> Path:
    """A work dir as export+mel leave it: labels/<sid>.csv + mel/<clip_id>.npy + mel/_meta.json."""
    (root / "labels").mkdir(parents=True)
    (root / "mel").mkdir()
    rng = np.random.default_rng(0)
    for sid, symbols in sessions.items():
        lines = ["onset_s,jamo,keytype,shift,scenario,participant,clip_id"]
        for i, sym in enumerate(symbols):
            kt = {"<sp>": "space", "<bs>": "backspace"}.get(sym, "normal")
            cid = f"{sid}_{i:08d}"
            lines.append(f"{i * 0.5:.3f},{sym},{kt},0,S01,{sid.split('_')[0]},{cid}")
            if cid not in missing:
                a = rng.standard_normal((64, 48)).astype(np.float32)
                np.save(root / "mel" / f"{cid}.npy", (a - a.mean()) / a.std())
        (root / "labels" / f"{sid}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import json
    (root / "mel" / "_meta.json").write_text(json.dumps({
        "mel_spec_version": "1.0", "param_hash": param_hash or classism_hash(), "sample_rate": 48000,
        "n_mels": 64, "frames": 48}), encoding="utf-8")
    return root


@pytest.fixture
def repos():
    return REPOS
