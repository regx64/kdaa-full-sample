"""Run a classism command on session shards, with the param_hash gate. Runs in its own process (torch stays out of
the orchestrator):

    python -m kdaa.classism_runner --classism third_party/classism --data work --sessions A,B,C \
        {check,train,evaluate,export} [-- classism args]

classism is used unmodified. Every classism entry point reads data through `src.dataset.load_samples` and
`src.dataset.LogMelDataset`; this runner swaps those two for versions that
  - keep only the listed sessions (held sessions stay out),
  - drop CSV rows whose log-mel meligation skipped (counted and reported),
  - serve tensors from the in-memory shard store instead of one .npy per clip,
and refuses to start when a shard's param_hash differs from classism's frozen regime (PLAN.md B1, NFR-2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pack import ShardError, ShardStore

EXIT_HASH = 3


def install(classism_repo: Path, data: Path, sessions: list[str]) -> dict:
    """Patch classism's dataset module in this process. Returns the context used by `check`."""
    sys.path.insert(0, str(classism_repo))
    from src import dataset, melio  # noqa: E402  (classism's package is literally named `src`)

    want = melio.REGIME.param_hash()
    store = ShardStore(data / "mel_packed", sessions, expected_hash=want)
    keep = set(sessions)
    original_load = dataset.load_samples
    ctx = {"param_hash": want, "store": store, "dropped": 0, "dataset": dataset}

    def load_samples(root, *, check_files: bool = True):
        samples = [s for s in original_load(root, check_files=False) if s.session_id in keep]
        kept = [s for s in samples if s.clip_id in store]
        ctx["dropped"] = len(samples) - len(kept)
        return kept

    class ShardLogMelDataset(dataset.LogMelDataset):
        def __getitem__(self, i: int):
            s = self.samples[i]
            return melio.check_logmel(store.get(s.clip_id))[None], s.label

    dataset.load_samples = load_samples
    dataset.LogMelDataset = ShardLogMelDataset
    return ctx


def check(ctx: dict, data: Path, config: str | None, out: Path | None) -> dict:
    from src.config import load_config

    dataset = ctx["dataset"]
    cfg = load_config(config)
    samples = dataset.load_samples(data)
    rep = dataset.coverage_report(samples, cfg["data"]["min_per_jamo"], cfg["data"]["min_per_special"])
    doc = {
        "param_hash": ctx["param_hash"],
        "shard_hashes": ctx["store"].hashes,
        "n_samples": len(samples),
        "n_dropped_no_mel": ctx["dropped"],
        "sessions": sorted({s.session_id for s in samples}),
        "participants": sorted({s.participant for s in samples}),
        "split": cfg["train"]["split"],
        "coverage_ok": rep["ok"],
        "coverage_short": rep["short"][:60],
        "coverage_short_total": len(rep["short"]),
        "counts": rep["counts"],
    }
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return doc


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="kdaa.classism_runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "train", "evaluate", "export"])
    ap.add_argument("--classism", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--sessions", required=True, help="comma-separated session ids to use")
    ap.add_argument("--config", default=None, help="classism YAML (check only; pass --config after -- for others)")
    ap.add_argument("--out", type=Path, default=None, help="check: write the JSON report here")
    a = ap.parse_args(argv)
    sessions = [s for s in a.sessions.split(",") if s]
    try:
        ctx = install(a.classism.resolve(), a.data.resolve(), sessions)
    except ShardError as e:
        print(f"kdaa: {e}", file=sys.stderr)
        return EXIT_HASH
    if a.command == "check":
        check(ctx, a.data, a.config, a.out)
        return 0
    module = {"train": "src.train", "evaluate": "src.evaluate", "export": "src.export_esp32"}[a.command]
    __import__(module)
    sys.modules[module].main(["--data", str(a.data), *passthrough])
    return 0


if __name__ == "__main__":
    sys.exit(main())
