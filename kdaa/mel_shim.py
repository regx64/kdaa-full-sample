"""Stopgap around aworse/meligation until its P0 items land (PLAN.md B1/B2).

- B2: meligation ships as one extension-less file `code`; it is loaded by path (a later `mel_build.py` is preferred).
- B1: its `Regime.param_hash()` hashes different field names than classism's `Regime`, so the hashes never match
  although the values are the same. This shim compares the VALUES field by field and, only when every one agrees,
  stamps classism's hash into `mel/_meta.json`. Any value mismatch raises: that is real regime drift, not naming.
  Once meligation's own hash equals classism's, its hash is used as is and the shim does nothing.

The extraction itself is meligation's (`MelExtractor`, `build_session`), unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .util import load_module, sha256_files, sha256_text, write_json

CANDIDATES = ("mel_build.py", "code")


class MelRegimeError(RuntimeError):
    """meligation's extraction parameters differ from classism's frozen regime."""


def meligation_source(repo: Path) -> Path:
    for name in CANDIDATES:
        if (repo / name).is_file():
            return repo / name
    raise FileNotFoundError(f"no meligation source ({' / '.join(CANDIDATES)}) in {repo}")


def _name(prefix: str, path: Path) -> str:
    return f"{prefix}_{sha256_text(str(path.resolve()))[:8]}"


def load_meligation(repo: Path) -> ModuleType:
    src = meligation_source(repo)
    return load_module(_name("kdaa_meligation", src), src)


def load_classism_melio(repo: Path) -> ModuleType:
    """classism's melio.py has no relative imports, so it loads standalone (no torch, no `src` on sys.path)."""
    src = repo / "src" / "melio.py"
    return load_module(_name("kdaa_classism_melio", src), src)


def regime_mismatches(mel, cls) -> list[str]:
    """Field-by-field comparison of meligation `Regime` values with classism `Regime` values."""
    pairs = [
        ("sample_rate", mel.sample_rate, cls.sample_rate),
        ("n_fft", mel.n_fft, cls.n_fft),
        ("win_length/win_samples", mel.win_length, cls.win_samples),
        ("hop_length/hop_samples", mel.hop_length, cls.hop_samples),
        ("n_mels", mel.n_mels, cls.n_mels),
        ("fmin", float(mel.fmin), float(cls.fmin)),
        ("fmax", float(mel.fmax), float(cls.fmax)),
        ("frames", mel.frames, cls.frames),
        ("log_eps", float(mel.log_eps), float(cls.log_eps)),
        ("pre_ms/onset_pre_ms", float(mel.pre_ms), float(cls.onset_pre_ms)),
        ("pre_ms+post_ms/clip_ms", float(mel.pre_ms + mel.post_ms), float(cls.clip_ms)),
    ]
    out = [f"{name}: meligation {a!r} != classism {b!r}" for name, a, b in pairs if a != b]
    # meligation hard-codes per-clip z-score and float32 in MelExtractor.logmel
    if cls.normalization != "per_clip_zscore":
        out.append(f"normalization: meligation per_clip_zscore != classism {cls.normalization!r}")
    if cls.dtype != "float32":
        out.append(f"dtype: meligation float32 != classism {cls.dtype!r}")
    return out


@dataclass
class MelBackend:
    mod: ModuleType
    source: Path
    param_hash: str          # the hash classism will accept
    own_hash: str            # meligation's own hash
    hash_source: str         # "meligation" | "kdaa-shim"

    @property
    def code_fingerprint(self) -> str:
        return sha256_files(self.source)

    def meta(self) -> dict:
        r = self.mod.Regime()
        return {"mel_spec_version": self.mod.MEL_SPEC_VERSION, "param_hash": self.param_hash,
                "sample_rate": r.sample_rate, "n_mels": r.n_mels, "frames": r.frames,
                "regime": r.canonical_dict(), "meligation_param_hash": self.own_hash,
                "param_hash_source": self.hash_source}

    def build_session(self, csv_path: Path, work: Path) -> tuple[int, int, list]:
        """-> (rows, written, skipped[(clip_id, reason)]). Writes work/mel/<clip_id>.npy and _meta.json."""
        ext = self.mod.MelExtractor(self.mod.Regime())
        _, n, ok, skipped = self.mod.build_session(csv_path, work, work, ext)
        write_json(work / "mel" / "_meta.json", self.meta())
        return n, ok, skipped


def open_backend(meligation_repo: Path, classism_repo: Path) -> MelBackend:
    mod = load_meligation(meligation_repo)
    cls = load_classism_melio(classism_repo).REGIME
    own, want = mod.Regime().param_hash(), cls.param_hash()
    if own == want:
        return MelBackend(mod, meligation_source(meligation_repo), own, own, "meligation")
    bad = regime_mismatches(mod.Regime(), cls)
    if bad:
        raise MelRegimeError("meligation and classism extract with different parameters:\n  " + "\n  ".join(bad))
    return MelBackend(mod, meligation_source(meligation_repo), want, own, "kdaa-shim")
