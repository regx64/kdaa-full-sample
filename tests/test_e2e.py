"""Whole pipeline on sound-autodaq's hardware-free SyntheticBackend: record -> export -> mel -> pack -> check ->
train (1 epoch) -> evaluate -> report. The accuracy of this data means nothing; the plumbing is what is tested."""
import json
import sys

import pytest
from conftest import HAS_TORCH, REPOS

from kdaa.cli import main as cli
from kdaa.config import load_config
from kdaa.stages import Pipeline
from kdaa.util import read_json

pytestmark = [pytest.mark.torch, pytest.mark.skipif(not HAS_TORCH, reason="needs PyTorch")]


def record(tmp_path, participant: str, session: str) -> str:
    """One 38-class keypress session (1 repetition per class) under tmp_path/data. Returns its session uid."""
    sys.path.insert(0, str(REPOS["sound_autodaq"]))
    from experiments.recording.tests.test_keylog import _config, _run
    cfg = _config(tmp_path, participant={"id": participant}, session={"id": session},
                  trial={"capture": "keypress", "pre_roll_ms": 500, "input_window_ms": 5000, "post_roll_ms": 500})
    _, summary, _ = _run(cfg, lambda n, target, rep: target)
    assert summary.completed, summary.stop_reason
    return f"{participant}_S01_{session}"


def pipeline_cfg(tmp_path):
    return load_config(None, overrides={
        "paths": {"raw": str(tmp_path / "data"), "work": str(tmp_path / "work"), "runs": str(tmp_path / "runs")},
        "repos": {k: str(v) for k, v in REPOS.items()},
        "classism": {"epochs": 1},
        "export": {"enabled": False},
    })


def test_record_to_report(tmp_path, capsys):
    a = record(tmp_path, "P01", "SESSION01")
    b = record(tmp_path, "P01", "SESSION02")
    cfg = pipeline_cfg(tmp_path)

    # two sessions: everything up to the split gate runs, then it stops (code 2) without training
    s = Pipeline(cfg).run()
    assert s["stopped"]["stage"] == "check" and s["stopped"]["code"] == 2, s
    assert "needs >= 3" in s["stopped"]["reason"]
    st_a = read_json(tmp_path / "work" / "state" / f"{a}.json")
    assert st_a["export"]["clips"] == 38 and st_a["mel"]["written"] == 38 and st_a["pack"]["clips"] == 38
    assert st_a["mel"]["param_hash"] == read_json(tmp_path / "work" / "mel" / "_meta.json")["param_hash"]
    assert not (tmp_path / "runs").exists()

    # a third session: only it is processed, then the model trains
    c = record(tmp_path, "P02", "SESSION01")
    s = Pipeline(cfg).run()
    assert s["stopped"] is None and s["trained"] is True, s
    assert read_json(tmp_path / "work" / "state" / f"{a}.json")["export"]["at"] == st_a["export"]["at"]
    run = tmp_path / "runs" / s["run"].split("/")[-1].split("\\")[-1]
    doc = read_json(run / "run.json")
    assert doc["status"] == "complete" and doc["reference_only"] is True
    assert {x["uid"] for x in doc["sessions"]} == {a, b, c} and doc["n_clips"] == 114
    assert doc["metrics_summary"]["n"] == 38 and doc["export"] == {"status": "disabled"}
    assert set(json.loads((run / "split.json").read_text())) >= {"train", "val", "test"}
    assert (run / "best.pt").exists() and (run / "eval_test" / "metrics.json").exists()
    chk = read_json(run / "check.json")  # classism read the shards and verified their hash
    assert sorted(chk["shard_sessions"]) == sorted([a, b, c]) and chk["param_hash"] == doc["param_hash"]
    assert "require_param_hash: true" in (run / "classism_config.yaml").read_text(encoding="utf-8")
    assert "학습 리포트" in (run / "report.md").read_text(encoding="utf-8")

    # nothing new: no retraining, the run is reused
    s2 = Pipeline(cfg).run()
    assert s2["trained"] is False and s2["run"] == s["run"]

    # a session still being recorded is held out
    (tmp_path / "data" / "P02" / "S01" / "SESSION01" / "session_summary.json").unlink()
    s3 = Pipeline(cfg).run(until="pack")
    assert s3["sessions"][c] == "recording not complete"

    capsys.readouterr()
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text(json.dumps(cfg.data), encoding="utf-8")  # JSON is YAML
    assert cli(["--config", str(cfg_file), "status"]) == 0
    out = capsys.readouterr().out
    assert a in out and doc["run_id"] in out
