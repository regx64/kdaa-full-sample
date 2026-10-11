"""Small shared helpers: hashing, atomic JSON, loading a module from a file path, repo commits."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def sha256_files(*paths: Path, extra: str = "") -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.name).encode())
        h.update(p.read_bytes() if p.exists() else b"<absent>")
    h.update(extra.encode())
    return h.hexdigest()


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_module(name: str, path: Path) -> ModuleType:
    """Import a single source file by path (works for extension-less files such as meligation's `code`)."""
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    no_pyc, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # leave the other repos' checkouts clean
    try:
        loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    finally:
        sys.dont_write_bytecode = no_pyc
    return mod


def repo_commit(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
