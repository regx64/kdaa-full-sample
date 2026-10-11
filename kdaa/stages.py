"""The pipeline (PLAN.md §4): export -> mel -> pack -> check -> train -> evaluate -> export_int8 -> report.

Per-session stages are incremental: each stores a fingerprint of its inputs in work/state/<session>.json and is
skipped while the fingerprint is unchanged. Training runs when the set of usable sessions (or the classism setup)
changed since the last complete run; otherwise that run is reused.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .config import Config
from .mel_shim import MelBackend, MelRegimeError, open_backend
from .pack import pack_session
from .report import summarize, write_report
from .util import read_json, repo_commit, sha256_files, sha256_text, write_json

STAGES = ("export", "mel", "pack", "check", "train", "evaluate", "export_int8", "report")
KDAA_ROOT = Path(__file__).resolve().parent.parent
EXPORT_MODULES = ("onnx", "onnxruntime", "onnx2tf", "tensorflow")


class PipelineStop(Exception):
    """A gate stopped the run (not a crash). `code` 2 = gate, 1 = failure."""

    def __init__(self, stage: str, reason: str, code: int = 2):
        super().__init__(f"[{stage}] {reason}")
        self.stage, self.reason, self.code = stage, reason, code


@dataclass
class Session:
    uid: str
    dir: Path
    complete: bool
    state: dict = field(default_factory=dict)
    held: str | None = None

    @property
    def usable(self) -> bool:
        return self.held is None


def discover(raw: Path) -> list[Session]:
    """Every sound-autodaq session directory under `raw` (one with a session.json)."""
    out = []
    for sj in sorted(raw.rglob("session.json")):
        doc = read_json(sj) or {}
        if "session_uid" not in doc:
            continue
        summary = read_json(sj.parent / "session_summary.json") or {}
        out.append(Session(doc["session_uid"], sj.parent, summary.get("completed") is True))
    uids = [s.uid for s in out]
    dup = sorted({u for u in uids if uids.count(u) > 1})
    if dup:
        raise PipelineStop("discover", f"session uid used by more than one directory: {dup}", 1)
    return out


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Pipeline:
    def __init__(self, cfg: Config, *, log=print, force: bool = False, force_train: bool = False):
        self.cfg, self.log, self.force, self.force_train = cfg, log, force, force_train
        self.work, self.runs = cfg.work, cfg.runs
        self.python = cfg["python"] or sys.executable
        self.commits = {name: repo_commit(cfg.repo(name)) for name in ("sound_autodaq", "meligation", "classism")}
        self._backend: MelBackend | None = None

    # ---- helpers ------------------------------------------------------------------------------
    def _state_path(self, uid: str) -> Path:
        return self.work / "state" / f"{uid}.json"

    def _save(self, s: Session) -> None:
        s.state["held"] = s.held
        write_json(self._state_path(s.uid), s.state)

    def _env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(KDAA_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _call(self, cmd: list[str], *, cwd: Path, log_path: Path | None = None, echo: bool = True) -> tuple[int, str]:
        """Run a stage subprocess, streaming its output; returns (exit code, last lines)."""
        tail: deque[str] = deque(maxlen=30)
        logf = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(log_path, "a", encoding="utf-8")
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, env=self._env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                tail.append(line.rstrip())
                if logf:
                    logf.write(line)
                if echo:
                    self.log("    " + line.rstrip())
            rc = proc.wait()
        finally:
            if logf:
                logf.close()
        return rc, "\n".join(tail)

    def _runner(self, command: str, sessions: list[Session], passthrough: list[str], *, out: Path | None = None,
                log_path: Path | None = None) -> tuple[int, str]:
        cmd = [self.python, "-m", "kdaa.classism_runner", command, "--classism", str(self.cfg.repo("classism")),
               "--data", str(self.work), "--sessions", ",".join(s.uid for s in sessions)]
        if command == "check" and self.cfg.classism_config():
            cmd += ["--config", str(self.cfg.classism_config())]
        if out:
            cmd += ["--out", str(out)]
        if passthrough:
            cmd += ["--", *passthrough]
        return self._call(cmd, cwd=KDAA_ROOT, log_path=log_path)

    @property
    def backend(self) -> MelBackend:
        if self._backend is None:
            try:
                self._backend = open_backend(self.cfg.repo("meligation"), self.cfg.repo("classism"))
            except MelRegimeError as e:
                raise PipelineStop("mel", str(e), 1) from None
            if self._backend.hash_source == "kdaa-shim":
                self.log(f"  mel: meligation hash {self._backend.own_hash} -> classism {self._backend.param_hash} "
                         f"(values verified equal; kdaa shim, PLAN.md B1)")
        return self._backend

    # ---- per-session stages -------------------------------------------------------------------
    def stage_export(self, s: Session) -> None:
        d = s.dir
        fp = sha256_files(d / "session.json", d / "manifest.jsonl", d / "session_summary.json",
                          extra=self.commits["sound_autodaq"])
        prev = s.state.get("export", {})
        csv_path = self.work / "labels" / f"{s.uid}.csv"
        if not self.force and prev.get("fp") == fp and csv_path.exists():
            return
        self.log(f"  export {s.uid}")
        rc, tail = self._call([self.python, "-m", "experiments.recording", "--export-classism", str(d),
                               "--out", str(self.work)], cwd=self.cfg.repo("sound_autodaq"), echo=False)
        if rc != 0:
            s.state["export"] = {"fp": None, "error": tail, "at": _now()}
            s.held = "export failed (see work/state)"
            self.log(f"    export failed:\n{tail}")
            return
        with open(csv_path, newline="", encoding="utf-8") as f:
            clips = sum(1 for _ in csv.DictReader(f))
        s.state["export"] = {"fp": fp, "clips": clips, "report": tail, "at": _now()}
        s.state.pop("mel", None)
        s.state.pop("pack", None)

    def stage_mel(self, s: Session) -> None:
        b = self.backend
        fp = sha256_text(s.state["export"]["fp"], b.code_fingerprint, b.param_hash)
        prev = s.state.get("mel", {})
        if self.force or prev.get("fp") != fp:
            self.log(f"  mel {s.uid}")
            rows, written, skipped = b.build_session(self.work / "labels" / f"{s.uid}.csv", self.work)
            s.state["mel"] = {"fp": fp, "rows": rows, "written": written, "skipped": len(skipped),
                              "skip_rate": (rows - written) / rows if rows else 0.0, "param_hash": b.param_hash,
                              "hash_source": b.hash_source, "at": _now()}
            s.state.pop("pack", None)
        m = s.state["mel"]
        limit = self.cfg["gates"]["max_skip_rate"]
        if m["rows"] == 0:
            s.held = "no exported clips"
        elif m["skip_rate"] > limit:
            s.held = f"mel skipped {m['skip_rate']:.1%} of clips (> {limit:.1%}); see work/mel/_skipped_{s.uid}.csv"

    def stage_pack(self, s: Session) -> None:
        fp = sha256_text(s.state["mel"]["fp"], "pack-v1")
        shard = self.work / "mel_packed" / f"{s.uid}.npz"
        if not self.force and s.state.get("pack", {}).get("fp") == fp and shard.exists():
            return
        r = pack_session(self.work, s.uid, remove_clip_npy=self.cfg["pack"]["remove_clip_npy"])
        self.log(f"  pack {s.uid}: {r.packed} clips -> {shard.relative_to(self.work)}")
        s.state["pack"] = {"fp": fp, "clips": r.packed, "missing": r.missing, "at": _now()}

    # ---- whole-dataset stages -----------------------------------------------------------------
    def stage_check(self, sessions: list[Session]) -> dict:
        out = self.work / "check.json"
        rc, tail = self._runner("check", sessions, [], out=out)
        if rc == 3:
            raise PipelineStop("check", tail.strip().splitlines()[-1] if tail.strip() else "param_hash mismatch", 1)
        if rc != 0:
            raise PipelineStop("check", f"classism check failed:\n{tail}", 1)
        chk = read_json(out)
        split = self.cfg["classism"]["split"] or chk["split"]
        groups = {"session": len(chk["sessions"]), "participant": len(chk["participants"])}.get(split)
        need = self.cfg["gates"]["min_sessions"]
        chk["gates"] = {"param_hash": "pass", "split": split, "groups": groups, "min_groups": need,
                        "coverage_ok": chk["coverage_ok"]}
        write_json(out, chk)
        if groups is not None and groups < need:
            raise PipelineStop("check", f"{split} split needs >= {need} {split}s, have {groups} "
                                        f"({', '.join(chk['sessions' if split == 'session' else 'participants'])})")
        if not chk["coverage_ok"]:
            msg = f"coverage short in {chk['coverage_short_total']} (participant, symbol) cells"
            if self.cfg["gates"]["require_coverage"]:
                raise PipelineStop("check", msg)
            self.log(f"  check: {msg} -> this run is marked reference-only")
        self.log(f"  check: {chk['n_samples']} clips, {len(chk['sessions'])} sessions, "
                 f"param_hash {chk['param_hash']} ok")
        return chk

    def _datahash(self, sessions: list[Session], chk: dict) -> str:
        c = self.cfg["classism"]
        cfg_file = self.cfg.classism_config() or (self.cfg.repo("classism") / "configs" / "afe.yaml")
        return sha256_text(json.dumps(sorted((s.uid, s.state["pack"]["fp"]) for s in sessions)),
                           sha256_files(cfg_file), json.dumps(c, sort_keys=True), self.commits["classism"],
                           chk["param_hash"])[:8]

    def _find_run(self, datahash: str) -> Path | None:
        if not self.runs.exists():
            return None
        for run_json in sorted(self.runs.glob("*/run.json"), reverse=True):
            doc = read_json(run_json) or {}
            if doc.get("datahash") == datahash and doc.get("status") == "complete":
                return run_json.parent
        return None

    def stage_train(self, sessions: list[Session], chk: dict) -> tuple[Path, bool]:
        """-> (run dir, trained now?)"""
        datahash = self._datahash(sessions, chk)
        existing = None if self.force_train else self._find_run(datahash)
        if existing:
            self.log(f"  train: data unchanged since run {existing.name}; reusing it (--force-train to retrain)")
            return existing, False
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S") + f"_{datahash}"
        run = self.runs / run_id
        run.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_id, "datahash": datahash, "status": "training", "started": _now(),
            "kdaa_version": __version__, "commits": self.commits, "param_hash": chk["param_hash"],
            "reference_only": not chk["coverage_ok"], "check": chk.get("gates"),
            "sessions": [{"uid": s.uid, "clips": s.state["pack"]["clips"], "mel_skipped": s.state["mel"]["skipped"],
                          "pack_fp": s.state["pack"]["fp"]} for s in sessions],
            "n_clips": chk["n_samples"], "participants": chk["participants"],
        }
        write_json(run / "run.json", manifest)
        shutil.copy2(self.work / "check.json", run / "check.json")
        c = self.cfg["classism"]
        args = ["--out", str(run), "--device", str(c["device"])]
        if self.cfg.classism_config():
            args += ["--config", str(self.cfg.classism_config())]
        if c["epochs"]:
            args += ["--epochs", str(c["epochs"])]
        if c["split"]:
            args += ["--split", c["split"]]
        self.log(f"  train -> {run}")
        rc, tail = self._runner("train", sessions, args, log_path=run / "logs" / "train.log")
        if rc != 0 or not (run / "best.pt").exists():
            manifest["status"] = "failed"
            manifest["error"] = tail
            write_json(run / "run.json", manifest)
            raise PipelineStop("train", f"classism train failed (log: {run / 'logs' / 'train.log'})", 1)
        return run, True

    def stage_evaluate(self, sessions: list[Session], run: Path) -> dict:
        split = self.cfg["evaluate"]["split"]
        args = ["--ckpt", str(run / "best.pt"), "--split", split]
        if self.cfg.classism_config():
            args += ["--config", str(self.cfg.classism_config())]
        self.log(f"  evaluate ({split})")
        rc, tail = self._runner("evaluate", sessions, args, log_path=run / "logs" / "evaluate.log")
        metrics = read_json(run / f"eval_{split}" / "metrics.json")
        if rc != 0 or metrics is None:
            raise PipelineStop("evaluate", f"classism evaluate failed (log: {run / 'logs' / 'evaluate.log'})", 1)
        return metrics

    def _export_available(self) -> bool:
        probe = "import importlib.util,sys; sys.exit(any(importlib.util.find_spec(m) is None for m in %r))" % (
            EXPORT_MODULES,)
        return subprocess.run([self.python, "-c", probe], env=self._env()).returncode == 0

    def stage_export_int8(self, sessions: list[Session], run: Path) -> dict:
        mode = self.cfg["export"]["enabled"]
        if mode is False:
            return {"status": "disabled"}
        if not self._export_available():
            if mode is True:
                raise PipelineStop("export_int8", f"export toolchain missing ({', '.join(EXPORT_MODULES)}); "
                                                  f"pip install -r third_party/classism/requirements-export.txt", 1)
            self.log("  export_int8: toolchain not installed, skipped (export.enabled: auto)")
            return {"status": "skipped", "reason": "toolchain not installed"}
        out = run / "export"
        args = ["--ckpt", str(run / "best.pt"), "--out", str(out)]
        if self.cfg.classism_config():
            args += ["--config", str(self.cfg.classism_config())]
        self.log("  export_int8")
        rc, _ = self._runner("export", sessions, args, log_path=run / "logs" / "export.log")
        deploy = read_json(out / "deployment.json")
        if rc != 0 or deploy is None:
            raise PipelineStop("export_int8", f"classism export failed (log: {run / 'logs' / 'export.log'})", 1)
        # O-O5: the int8 input quantization the Mel component needs on device
        write_json(run / "handoff" / "mel_int8_input.json", {
            "param_hash": deploy["input_regime"]["param_hash"], "input_regime": deploy["input_regime"],
            "quantization": deploy["quantization"], "note": deploy.get("note")})
        return {"status": "done", "report": read_json(out / "export_report.json")}

    # ---- driver -------------------------------------------------------------------------------
    def run(self, until: str = "report") -> dict:
        if until not in STAGES:
            raise ValueError(f"unknown stage {until!r}; one of {STAGES}")
        stop_after = STAGES.index(until)
        self.log(f"kdaa: raw={self.cfg.raw} work={self.work}")
        summary: dict = {"sessions": {}, "stopped": None}
        try:
            sessions = discover(self.cfg.raw)
            for s in sessions:
                s.state = read_json(self._state_path(s.uid), {}) or {}
                if not s.complete and not self.cfg["sessions"]["include_incomplete"]:
                    s.held = "recording not complete"
            self.log(f"  {len(sessions)} sessions found, {sum(s.complete for s in sessions)} complete")
            for i, (name, fn) in enumerate((("export", self.stage_export), ("mel", self.stage_mel),
                                            ("pack", self.stage_pack))):
                if i > stop_after:
                    break
                for s in sessions:
                    if s.usable:
                        fn(s)
                    self._save(s)
            summary["sessions"] = {s.uid: (s.held or "ok") for s in sessions}
            for s in sessions:
                if s.held and s.complete:
                    self.log(f"  held {s.uid}: {s.held}")
            usable = [s for s in sessions if s.usable]
            if stop_after < STAGES.index("check"):
                return summary
            if not usable:
                raise PipelineStop("check", "no usable session (record and complete one first)")
            chk = self.stage_check(usable)
            if stop_after < STAGES.index("train"):
                return summary
            run, trained = self.stage_train(usable, chk)
            summary["run"], summary["trained"] = str(run), trained
            if not trained:
                return summary
            manifest = read_json(run / "run.json")
            if stop_after >= STAGES.index("evaluate"):
                manifest["metrics_summary"] = summarize(self.stage_evaluate(usable, run))
            if stop_after >= STAGES.index("export_int8"):
                manifest["export"] = self.stage_export_int8(usable, run)
            manifest["status"] = "complete" if stop_after >= STAGES.index("report") else "partial"
            manifest["finished"] = _now()
            write_json(run / "run.json", manifest)
            if stop_after >= STAGES.index("report"):
                path = write_report(self.runs, run, manifest)
                self.log(f"  report -> {path}")
            return summary
        except PipelineStop as e:
            self.log(f"kdaa: stopped at {e.stage}: {e.reason}")
            summary["stopped"] = {"stage": e.stage, "reason": e.reason, "code": e.code}
            return summary
