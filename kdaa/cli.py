"""python -m kdaa {run,status,watch}

  run     process new/changed sessions and train when the data changed      (exit 0 ok, 2 stopped at a gate, 1 failed)
  status  sessions and their stage state, latest runs
  watch   `run` every --interval seconds (picks up sessions as they complete)
"""
from __future__ import annotations

import argparse
import sys
import time

from .config import load_config
from .stages import STAGES, Pipeline, discover
from .util import read_json


def cmd_run(cfg, a) -> int:
    summary = Pipeline(cfg, force=a.force, force_train=a.force_train).run(until=a.until)
    stop = summary.get("stopped")
    return stop["code"] if stop else 0


def cmd_status(cfg, a) -> int:
    sessions = discover(cfg.raw) if cfg.raw.exists() else []
    print(f"raw  {cfg.raw}\nwork {cfg.work}\n")
    print(f"{'session':32} {'recording':10} {'export':>7} {'mel':>11} {'pack':>6}  held")
    for s in sessions:
        st = read_json(cfg.work / "state" / f"{s.uid}.json", {}) or {}
        exp, mel, pack = st.get("export") or {}, st.get("mel") or {}, st.get("pack") or {}
        mel_txt = f"{mel['written']}/{mel['rows']}" if mel else "-"
        print(f"{s.uid:32} {'complete' if s.complete else 'recording':10} {exp.get('clips', '-')!s:>7} "
              f"{mel_txt:>11} {pack.get('clips', '-')!s:>6}  {st.get('held') or ''}")
    runs = sorted(cfg.runs.glob("*/run.json"))[-5:] if cfg.runs.exists() else []
    if runs:
        print(f"\n{'run':28} {'status':9} {'clips':>6} {'top1':>6} {'top5':>6}")
    for p in runs:
        r = read_json(p) or {}
        m = r.get("metrics_summary") or {}
        f = lambda k: f"{m[k]:.3f}" if isinstance(m.get(k), float) else "-"  # noqa: E731
        print(f"{r.get('run_id', p.parent.name):28} {r.get('status', '?'):9} {r.get('n_clips', '-')!s:>6} "
              f"{f('top1'):>6} {f('top5'):>6}")
    return 0


def cmd_watch(cfg, a) -> int:
    print(f"kdaa watch: every {a.interval}s (Ctrl+C to stop)")
    try:
        while True:
            Pipeline(cfg).run(until=a.until)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kdaa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="pipeline.yaml (default: ./pipeline.yaml)")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the pipeline once")
    r.add_argument("--until", choices=STAGES, default="report", help="stop after this stage")
    r.add_argument("--force", action="store_true", help="redo export/mel/pack even when unchanged")
    r.add_argument("--force-train", action="store_true", help="train even when the data is unchanged")
    sub.add_parser("status", help="show sessions and runs")
    w = sub.add_parser("watch", help="run repeatedly")
    w.add_argument("--interval", type=int, default=600)
    w.add_argument("--until", choices=STAGES, default="report")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    return {"run": cmd_run, "status": cmd_status, "watch": cmd_watch}[a.command](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
