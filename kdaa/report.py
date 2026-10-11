"""runs/<run_id>/report.md: what went in, how it scored, and the change against the previous complete run."""
from __future__ import annotations

from pathlib import Path

from .util import read_json

METRICS = (("top1", "top-1"), ("top5", "top-5"), ("keytype_acc", "keytype 정확도"), ("ece", "ECE"))


def summarize(m: dict) -> dict:
    return {
        "n": m["n"],
        **{k: m[k] for k, _ in METRICS},
        "calibrated": m.get("calibrated"),
        "per_scenario": {sc: {k: v.get(k) for k in ("n", "top1", "top5")} for sc, v in (m.get("per_scenario") or {}).items()},
        "acceptance": {k: v.get("pass") for k, v in (m.get("acceptance") or {}).items()},
    }


def previous_run(runs: Path, current: Path) -> dict | None:
    best = None
    for p in sorted(runs.glob("*/run.json")):
        if p.parent == current:
            continue
        doc = read_json(p) or {}
        if doc.get("status") == "complete" and doc.get("metrics_summary") and doc["run_id"] < current.name:
            best = doc
    return best


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def _verdict(v) -> str:
    return {True: "통과", False: "미달", None: "측정 불가"}.get(v, str(v))


def write_report(runs: Path, run: Path, manifest: dict) -> Path:
    cur = manifest.get("metrics_summary") or {}
    prev = previous_run(runs, run)
    ps = (prev or {}).get("metrics_summary") or {}
    lines = [f"# 학습 리포트 {manifest['run_id']}", ""]
    if manifest.get("reference_only"):
        lines += ["> **참고용 실행:** 자모/특수키 coverage가 기준(spec §7)에 못 미친 데이터로 학습했다. "
                  "정확도를 목표 판정에 쓰지 말 것.", ""]
    lines += [f"- 클립 {manifest['n_clips']}개, 세션 {len(manifest['sessions'])}개, 참가자 {len(manifest['participants'])}명",
              f"- param_hash `{manifest['param_hash']}`, 데이터 해시 `{manifest['datahash']}`",
              "- 커밋: " + ", ".join(f"{k} `{v[:7]}`" for k, v in manifest["commits"].items()), ""]
    if cur:
        head = f"이전 ({prev['run_id']})" if prev else "이전"
        lines += [f"## 지표 (평가 {cur['n']}클립)", "", f"| 지표 | 이번 | {head} | 변화 |", "| --- | --- | --- | --- |"]
        for key, name in METRICS:
            a, b = cur.get(key), ps.get(key)
            delta = f"{a - b:+.3f}" if isinstance(a, float) and isinstance(b, float) else "—"
            lines.append(f"| {name} | {_fmt(a)} | {_fmt(b)} | {delta} |")
        lines += ["", "| 판정 | 결과 |", "| --- | --- |"]
        lines += [f"| {k} | {_verdict(v)} |" for k, v in cur.get("acceptance", {}).items()]
        if cur.get("per_scenario"):
            lines += ["", "| 시나리오 | 클립 | top-1 | top-5 |", "| --- | --- | --- | --- |"]
            lines += [f"| {sc} | {v['n']} | {_fmt(v['top1'])} | {_fmt(v['top5'])} |" for sc, v in cur["per_scenario"].items()]
        lines.append("")
    exp = manifest.get("export") or {}
    lines += ["## int8 export", ""]
    ac6 = exp.get("report") or {}
    if exp.get("status") == "done":
        lines += ["- 완료. Mel 온디바이스 입력 양자화: `handoff/mel_int8_input.json`",
                  f"- AC-6 정확도 하락 {_fmt(ac6.get('top1_drop_pp'))}%p: {_verdict(ac6.get('pass_accuracy'))}, "
                  f"크기 {ac6.get('size_bytes')} B: {_verdict(ac6.get('pass_size'))}", ""]
    else:
        lines += [f"- {exp.get('status', '실행 안 함')}{': ' + exp['reason'] if exp.get('reason') else ''}", ""]
    lines += ["## 세션", "", "| 세션 | 클립 | Mel에서 건너뜀 |", "| --- | --- | --- |"]
    lines += [f"| {s['uid']} | {s['clips']} | {s['mel_skipped']} |" for s in manifest["sessions"]]
    path = run / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
