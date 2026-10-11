"""pipeline.yaml: YAML over built-in defaults. Relative paths resolve against the config file's directory."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "paths": {"raw": "data/raw", "work": "work", "runs": "runs"},
    "repos": {
        "sound_autodaq": "third_party/sound-autodaq",
        "meligation": "third_party/meligation",
        "classism": "third_party/classism",
    },
    "sessions": {"include_incomplete": False},
    "gates": {
        "max_skip_rate": 0.01,       # mel clips skipped / exported clips, per session (above -> session held)
        "min_sessions": 3,           # classism session split needs train/val/test sessions
        "require_coverage": False,   # True: stop before training when coverage is short
    },
    "pack": {"remove_clip_npy": False},
    "classism": {
        "config": None,              # classism YAML override; None = classism's configs/afe.yaml
        "epochs": None,
        "split": None,               # session | participant | random; None = classism config
        "device": "cpu",
    },
    "evaluate": {"split": "test"},
    "export": {"enabled": "auto"},   # auto: run when onnx + tensorflow import; true: required; false: skip
    "python": None,                  # interpreter for stage subprocesses; None = this one
}

PATH_KEYS = ("paths", "repos")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    data: dict
    base_dir: Path

    def path(self, section: str, key: str) -> Path:
        p = Path(self.data[section][key])
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    @property
    def raw(self) -> Path:
        return self.path("paths", "raw")

    @property
    def work(self) -> Path:
        return self.path("paths", "work")

    @property
    def runs(self) -> Path:
        return self.path("paths", "runs")

    def repo(self, name: str) -> Path:
        return self.path("repos", name)

    def classism_config(self) -> Path | None:
        c = self.data["classism"]["config"]
        if not c:
            return None
        p = Path(c)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def __getitem__(self, key: str):
        return self.data[key]


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """`path` defaults to ./pipeline.yaml when present, else the built-in defaults rooted at the cwd."""
    p = Path(path) if path else Path("pipeline.yaml")
    over: dict = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            over = yaml.safe_load(f) or {}
    elif path:
        raise FileNotFoundError(f"config {p} not found")
    data = _merge(_merge(DEFAULTS, over), overrides or {})
    base = p.resolve().parent if p.exists() else Path.cwd()
    if data["export"]["enabled"] not in ("auto", True, False):
        raise ValueError("export.enabled must be auto | true | false")
    return Config(data, base)
