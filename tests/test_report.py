from kdaa.report import summarize, write_report
from kdaa.util import write_json

METRICS = {"n": 40, "top1": 0.5, "top3": 0.7, "top5": 0.8, "keytype_acc": 0.9, "ece": 0.04, "calibrated": True,
           "per_scenario": {"near": {"n": 40, "top1": 0.5, "top5": 0.8}},
           "acceptance": {"AC-3_near_top1": {"pass": False}, "AC-5_ece": {"pass": True}}}


def manifest(run_id, top1, export=None):
    return {"run_id": run_id, "datahash": "abcd1234", "status": "complete", "param_hash": "915acee6",
            "reference_only": False, "n_clips": 200, "participants": ["P01"],
            "commits": {"classism": "c6606a0aaaa", "meligation": "7acb1bdbbbb", "sound_autodaq": "c0823bdcccc"},
            "sessions": [{"uid": "P01_S01_SESSION01", "clips": 200, "mel_skipped": 0}],
            "metrics_summary": summarize({**METRICS, "top1": top1}), "export": export or {"status": "disabled"}}


def test_report_compares_with_the_previous_complete_run(tmp_path):
    old, new = tmp_path / "20261001-000000_aaaa", tmp_path / "20261002-000000_bbbb"
    write_json(old / "run.json", manifest(old.name, 0.4))
    new.mkdir()
    export = {"status": "done", "report": {"top1_drop_pp": 1.25, "pass_accuracy": True, "size_bytes": 120000,
                                           "pass_size": True}}
    text = write_report(tmp_path, new, manifest(new.name, 0.5, export)).read_text(encoding="utf-8")
    assert f"이전 ({old.name})" in text
    assert "| top-1 | 0.500 | 0.400 | +0.100 |" in text
    assert "| AC-3_near_top1 | 미달 |" in text and "| AC-5_ece | 통과 |" in text
    assert "handoff/mel_int8_input.json" in text and "1.250%p: 통과" in text
    assert "참고용" not in text
