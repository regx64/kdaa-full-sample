"""Build the KDAA handbook site.

    python docs/handbook/build.py            # build docs/handbook/site/
    python docs/handbook/build.py --stamp kdaa/stages.py   # after reviewing notes against changed code

Source code is read from the repositories themselves (kdaa root and third_party/*), never copied by hand.
Explanations live in notes/<repo>/<path>.txt as blocks:

    sha: 3f2a9c1e5b7d            (first 12 hex of the source file's sha256 when the notes were last checked)
    @@ 1-6
    text for lines 1..6
    @@ 8
    text for line 8

The build fails when a non-blank source line has no block, when blocks overlap or point past the end, or when
the source changed since `sha` (re-read the notes, fix them, then --stamp). Note text is a small markdown:
paragraphs, "- " lists, `code`, **bold**, [text](url), [[python-concept]] and {{repo:path:line}} links.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from pygments.lexers import PythonLexer
from pygments.token import STANDARD_TYPES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "site"

# reading order inside each repository; `desc` is shown in the file list
REPOS = [
    {"id": "kdaa", "name": "kdaa", "full": "regx64/kdaa-full-sample", "root": ROOT, "files": [
        ("kdaa/__main__.py", "python -m kdaa 로 실행될 때 가장 먼저 도는 파일"),
        ("kdaa/cli.py", "명령(run, status, watch)을 해석하고 결과를 종료 코드로 돌려준다"),
        ("kdaa/config.py", "pipeline.yaml을 읽어 기본값과 합친다"),
        ("kdaa/util.py", "해시, JSON 저장, 파일 경로로 모듈 불러오기 같은 작은 도구"),
        ("kdaa/stages.py", "파이프라인 본체. 세션을 찾고 단계를 차례로 실행한다"),
        ("kdaa/mel_shim.py", "meligation을 불러와 classism 규격과 맞는지 확인한다(임시)"),
        ("kdaa/pack.py", "세션의 로그멜 파일들을 .npz 하나로 묶는다"),
        ("kdaa/report.py", "학습 결과를 report.md로 정리한다"),
        ("kdaa/__init__.py", "패키지 표시와 버전 번호"),
    ]},
    {"id": "mel", "name": "meligation", "full": "aworse/meligation", "root": ROOT / "third_party" / "meligation", "files": [
        ("code", "WAV에서 타건 100 ms를 잘라 로그멜(64×48)로 바꾸는 파일 하나짜리 프로그램"),
    ]},
    {"id": "afe", "name": "classism", "full": "regx64/classism", "root": ROOT / "third_party" / "classism", "files": [
        ("src/labels.py", "38개 기호와 번호"),
        ("src/melio.py", "로그멜 규격(Regime), param_hash, 입력 검사"),
        ("src/config.py", "configs/afe.yaml을 읽고 검사한다"),
        ("src/dataset.py", "라벨 CSV와 로그멜을 읽어 학습 데이터로 만들고, 나눈다"),
        ("src/model.py", "SmallCNN과 CoAtNetLite 모델"),
        ("src/train.py", "학습 루프"),
        ("src/infer.py", "로그멜 → 상위 N개 기호와 확률(kdaa.afe.v1)"),
        ("src/schema.py", "kdaa.afe.v1 문서 검사"),
        ("src/calibrate.py", "온도 보정과 ECE"),
        ("src/evaluate.py", "정확도, 혼동 행렬, 합격 판정, 그림"),
        ("src/export_esp32.py", "PyTorch → ONNX → TFLite int8 변환"),
        ("src/synth.py", "시험용 가짜 데이터"),
        ("src/hangul.py", "자모 조합과 빔 탐색 데모(실제 파이프라인에서는 안 씀)"),
    ]},
    {"id": "rec", "name": "sound-autodaq", "full": "regx64/sound-autodaq", "root": ROOT / "third_party" / "sound-autodaq", "files": [
        ("classism/labels.py", "녹음기가 쓰는 38개 기호(classism 사본)"),
        ("experiments/recording/__main__.py", "python -m experiments.recording 진입점"),
        ("experiments/recording/cli.py", "명령줄 옵션과 명령 실행"),
        ("experiments/recording/errors.py", "녹음기 전용 예외"),
        ("experiments/recording/config.py", "녹음 설정 YAML 읽기와 검사"),
        ("experiments/recording/labels.py", "classism/labels.py를 다시 내보냄"),
        ("experiments/recording/clock.py", "모든 시각을 재는 시계"),
        ("experiments/recording/scheduler.py", "어떤 키를 몇 번째에 누를지 순서 만들기"),
        ("experiments/recording/keylog.py", "눌린 키를 38개 기호로 바꾸고 맞았는지 판정"),
        ("experiments/recording/keyhook.py", "운영체제 키보드 훅(pynput)"),
        ("experiments/recording/recorder.py", "마이크 스트림과 링 버퍼, 구간 잘라내기"),
        ("experiments/recording/trial.py", "trial 하나의 단계(카운트다운→녹음→저장)와 상태"),
        ("experiments/recording/quality.py", "소리 크기, 클리핑 측정"),
        ("experiments/recording/writer.py", "WAV 저장과 manifest 기록"),
        ("experiments/recording/metadata.py", "session.json, session_summary.json"),
        ("experiments/recording/progress.py", "progress.json과 이어 녹음"),
        ("experiments/recording/hardware.py", "장치 목록, 마이크 테스트, 디스크 용량"),
        ("experiments/recording/ui.py", "터미널 화면과 운영자 키"),
        ("experiments/recording/validator.py", "녹음된 세션 검사(읽기 전용)"),
        ("experiments/recording/export_classism.py", "세션을 classism 학습 형식으로 내보내기"),
        ("experiments/recording/engine.py", "세션 전체를 진행하는 본체"),
        ("experiments/recording/webui/__init__.py", "대시보드 패키지 표시"),
        ("experiments/recording/webui/status.py", "대시보드가 보여 줄 상태 계산"),
        ("experiments/recording/webui/server.py", "대시보드 웹 서버"),
    ]},
]

PAGES = [  # (output, source fragment, <title>, nav label)
    ("index.html", "pages/index.html", "KDAA 코드 읽기", "처음"),
    ("python.html", "pages/python.html", "코드 읽기 전 파이썬", "파이썬"),
    ("reference.html", "pages/reference.html", "KDAA 계약과 설정", "참조"),
]


# ---- notes ------------------------------------------------------------------------------------
@dataclass
class Block:
    start: int
    end: int
    text: str


class NotesError(Exception):
    pass


def source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def notes_path(repo: dict, rel: str) -> Path:
    return HERE / "notes" / repo["id"] / (rel + ".txt")


def parse_notes(path: Path) -> tuple[str | None, list[Block]]:
    sha, blocks, cur = None, [], None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"@@\s*(\d+)(?:\s*-\s*(\d+))?\s*", raw)
        if m:
            a = int(m.group(1))
            cur = Block(a, int(m.group(2) or a), "")
            blocks.append(cur)
        elif cur is None:
            if raw.startswith("sha:"):
                sha = raw.split(":", 1)[1].strip()
        else:
            cur.text += raw + "\n"
    for b in blocks:
        b.text = b.text.strip()
    return sha, blocks


def check_blocks(rel: str, lines: list[str], blocks: list[Block]) -> list[str]:
    errs, covered = [], [False] * (len(lines) + 1)
    prev_end = 0
    for b in blocks:
        if b.start > b.end or b.start < 1 or b.end > len(lines):
            errs.append(f"{rel}: block @@ {b.start}-{b.end} is outside 1..{len(lines)}")
            continue
        if b.start <= prev_end:
            errs.append(f"{rel}: block @@ {b.start}-{b.end} overlaps or is out of order")
        if not b.text:
            errs.append(f"{rel}: block @@ {b.start}-{b.end} has no text")
        prev_end = b.end
        for i in range(b.start, b.end + 1):
            covered[i] = True
    missing = [i for i in range(1, len(lines) + 1) if lines[i - 1].strip() and not covered[i]]
    if missing:
        errs.append(f"{rel}: lines without notes: {compress(missing)}")
    return errs


def compress(nums: list[int]) -> str:
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(out)


# ---- code highlighting ------------------------------------------------------------------------
def css_class(ttype) -> str:
    while ttype not in STANDARD_TYPES and ttype.parent is not None:
        ttype = ttype.parent
    return STANDARD_TYPES.get(ttype, "")


def highlight_lines(src: str) -> list[str]:
    """One HTML string per source line; tokens spanning lines are split per line."""
    out, cur = [], []
    for ttype, value in PythonLexer().get_tokens(src):
        cls = css_class(ttype)
        parts = value.split("\n")
        for k, part in enumerate(parts):
            if k:
                out.append("".join(cur))
                cur = []
            if part:
                esc = html.escape(part)
                cur.append(f'<span class="{cls}">{esc}</span>' if cls else esc)
    if cur:
        out.append("".join(cur))
    n = src.count("\n") + (0 if src.endswith("\n") else 1)
    return (out + [""] * n)[:n]


# ---- mini markdown ----------------------------------------------------------------------------
CONCEPTS: dict[str, str] = {}   # id -> title, filled from pages/python.html
FILE_PAGES: dict[str, str] = {}  # "repo:path" -> page url (relative to site root)


def inline(text: str, prefix: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)

    def concept(m):
        key = m.group(1)
        label = m.group(2) or CONCEPTS.get(key, key)
        if key not in CONCEPTS:
            raise NotesError(f"unknown concept [[{key}]]")
        return f'<a class="cx" href="{prefix}python.html#py-{key}">{label}</a>'
    t = re.sub(r"\[\[([a-z0-9-]+)(?:\|([^\]]+))?\]\]", concept, t)

    def fileref(m):
        key, line, label = f"{m.group(1)}:{m.group(2)}", m.group(3), m.group(4)
        if key not in FILE_PAGES:
            return f"<code>{label or m.group(2)}</code>"
        anchor = f"#L{line}" if line else ""
        return f'<a href="{prefix}{FILE_PAGES[key]}{anchor}">{label or (m.group(2) + (":" + line if line else ""))}</a>'
    t = re.sub(r"\{\{([a-z]+):([^:}|]+)(?::(\d+))?(?:\|([^}]+))?\}\}", fileref, t)
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', t)
    return t


def render_md(text: str, prefix: str) -> str:
    out, para, items = [], [], []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para), prefix) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i, prefix)}</li>" for i in items) + "</ul>")
            items.clear()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s.startswith("- "):
            if para:
                flush()
            items.append(s[2:])
        elif items and line.startswith("  "):
            items[-1] += " " + s
        else:
            if items:
                flush()
            para.append(s)
    flush()
    return "".join(out)


# ---- pages ------------------------------------------------------------------------------------
def nav(prefix: str, current: str) -> str:
    links = [("index.html", "처음"), ("python.html", "파이썬"), ("code.html", "코드"), ("reference.html", "참조")]
    a = "".join(f'<a href="{prefix}{u}"{" class=\"on\"" if u == current else ""}>{t}</a>' for u, t in links)
    return f"""<header class="nav"><div class="nav-inner">
<a class="brand" href="{prefix}index.html">KDAA <span>코드 읽기</span></a>
<nav class="chapters" aria-label="페이지">{a}</nav>
<div class="search"><input id="q" type="search" placeholder="찾기  /" autocomplete="off" aria-label="핸드북 검색" data-prefix="{prefix}"><ul class="results" id="results" hidden></ul></div>
<button class="theme-btn" id="theme" type="button" aria-label="밝은/어두운 화면 전환">◐</button>
</div></header>"""


def page(title: str, body: str, prefix: str, current: str, *, full: bool = True) -> str:
    head = f"""<title>{html.escape(title)}</title>
<meta name="description" content="KDAA 파이프라인 네 저장소의 코드를 한 줄씩 설명하는 안내서">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&amp;family=JetBrains+Mono:wght@400;600&amp;display=swap">
<link rel="stylesheet" href="{prefix}assets/style.css">"""
    content = f"""{nav(prefix, current)}
{body}
<footer class="footer">KDAA 코드 읽기 · 코드는 각 저장소에서 빌드할 때 그대로 가져온 것 · <a href="{prefix}code.html">파일 목록</a></footer>
<script src="{prefix}assets/app.js"></script>"""
    if not full:  # artifact page: the viewer adds doctype/html/head/body
        return head + "\n" + content
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{head}
</head><body>
{content}
</body></html>
"""


def code_page(repo: dict, rel: str, blocks: list[Block], lines: list[str], hl: list[str], idx: int, files: list) -> str:
    prefix = "../../"
    rows = []
    # blank lines that no block covers are shown with the block before them (or the first block)
    bounds = []
    for k, b in enumerate(blocks):
        start = 1 if k == 0 else b.start
        end = (blocks[k + 1].start - 1) if k + 1 < len(blocks) else len(lines)
        bounds.append((start, end, b))
    for start, end, b in bounds:
        code = "".join(f'<span class="ln" id="L{i}"><a href="#L{i}">{i}</a>{hl[i - 1] or " "}</span>'
                       for i in range(start, end + 1))
        lab = f"{b.start}" if b.start == b.end else f"{b.start}–{b.end}"
        rows.append(f'<div class="row" id="B{b.start}"><div class="src"><pre>{code}</pre></div>'
                    f'<div class="ex"><span class="lines">{lab}줄</span>{render_md(b.text, prefix)}</div></div>')
    desc = dict(repo["files"])[rel]
    prev_l = next_l = ""
    if idx > 0:
        p = files[idx - 1]
        prev_l = f'<a class="prev" href="{prefix}{FILE_PAGES[p[0] + ":" + p[1]]}"><span>이전 파일</span>{p[1]}</a>'
    if idx + 1 < len(files):
        n = files[idx + 1]
        next_l = f'<a class="next" href="{prefix}{FILE_PAGES[n[0] + ":" + n[1]]}"><span>다음 파일</span>{n[1]}</a>'
    body = f"""<main class="codepage k-{repo['id']}">
<div class="file-head">
  <p class="eyebrow"><i class="dot"></i>{repo['name']} · <a href="{prefix}code.html#{repo['id']}">{repo['full']}</a></p>
  <h1 class="file-title">{html.escape(rel)}</h1>
  <p class="file-desc">{html.escape(desc)} · {len(lines)}줄</p>
  <p class="howto">왼쪽이 코드, 오른쪽이 그 줄들의 설명입니다. 파란 밑줄 단어는 <a href="{prefix}python.html">파이썬 개념</a> 설명으로 갑니다. 줄 번호를 누르면 그 줄 주소가 생깁니다.</p>
</div>
<div class="annotated">{''.join(rows)}</div>
<nav class="pager">{prev_l}{next_l}</nav>
</main>"""
    return page(f"{rel} · {repo['name']}", body, prefix, "code.html")


def files_list(status: dict) -> str:
    parts = []
    for repo in REPOS:
        items = []
        for rel, desc in repo["files"]:
            key = f"{repo['id']}:{rel}"
            n = status[key]["lines"]
            if key in FILE_PAGES:
                items.append(f'<li><a href="{FILE_PAGES[key]}"><code>{html.escape(rel)}</code><span>{html.escape(desc)}</span><small>{n}줄</small></a></li>')
            else:
                items.append(f'<li class="todo"><div><code>{html.escape(rel)}</code><span>{html.escape(desc)}</span><small>{n}줄 · 설명 준비 중</small></div></li>')
        done = sum(1 for rel, _ in repo["files"] if f"{repo['id']}:{rel}" in FILE_PAGES)
        parts.append(f"""<section class="repo k-{repo['id']}" id="{repo['id']}">
<h2><i class="dot"></i>{repo['name']} <small>{repo['full']} · 파일 {done}/{len(repo['files'])} 설명 완료</small></h2>
<p class="note">위에서 아래로 읽는 순서입니다.</p>
<ol class="files">{''.join(items)}</ol></section>""")
    return "".join(parts)


def code_index(status: dict) -> str:
    return f"""<main class="plainpage"><div class="page-head"><p class="name">코드</p>
<h1>파일 목록</h1><p class="lede">저장소마다 읽는 순서대로 놓았습니다. 앞 파일에서 만든 것을 뒤 파일이 쓰므로 위에서부터 읽는 편이 쉽습니다.</p></div>
{files_list(status)}</main>"""


# ---- search index -----------------------------------------------------------------------------
def strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


# ---- main -------------------------------------------------------------------------------------
def build(stamp: list[str]) -> int:
    py = (HERE / "pages" / "python.html").read_text(encoding="utf-8")
    for m in re.finditer(r'id="py-([a-z0-9-]+)"[^>]*>\s*<h2>(.*?)<', py, re.S):
        CONCEPTS[m.group(1)] = strip_tags(m.group(2))

    errors, status, entries = [], {}, []
    for repo in REPOS:
        for rel, _ in repo["files"]:
            src_path = repo["root"] / rel
            if not src_path.exists():
                errors.append(f"missing source {src_path} (git submodule update --init?)")
                continue
            key = f"{repo['id']}:{rel}"
            status[key] = {"lines": len(src_path.read_text(encoding="utf-8").splitlines())}
            np_ = notes_path(repo, rel)
            if np_.exists():
                FILE_PAGES[key] = f"code/{repo['id']}/{rel.replace('/', '__')}.html"
                if key.split(":", 1)[1] in stamp or f"{repo['id']}/{rel}" in stamp or rel in stamp:
                    txt = np_.read_text(encoding="utf-8")
                    txt = re.sub(r"^sha:.*\n", "", txt)
                    np_.write_text(f"sha: {source_sha(src_path)}\n" + txt, encoding="utf-8")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "assets").mkdir(parents=True)
    for f in ("style.css", "app.js"):
        shutil.copy2(HERE / "assets" / f, OUT / "assets" / f)

    pages_out = []
    for repo in REPOS:
        files = [(repo["id"], rel) for rel, _ in repo["files"] if f"{repo['id']}:{rel}" in FILE_PAGES]
        for idx, (rid, rel) in enumerate(files):
            src_path = repo["root"] / rel
            src = src_path.read_text(encoding="utf-8")
            lines = src.splitlines()
            sha, blocks = parse_notes(notes_path(repo, rel))
            if sha != source_sha(src_path):
                errors.append(f"{rid}:{rel}: source changed since the notes were checked (notes sha {sha}, "
                              f"source {source_sha(src_path)}); review the notes, then --stamp {rel}")
            errs = check_blocks(f"{rid}:{rel}", lines, blocks)
            errors += errs
            if errs:
                continue
            try:
                out = code_page(repo, rel, blocks, lines, highlight_lines(src), idx, files)
            except NotesError as e:
                errors.append(f"{rid}:{rel}: {e}")
                continue
            dest = OUT / FILE_PAGES[f"{rid}:{rel}"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(out, encoding="utf-8")
            pages_out.append(dest)
            for b in blocks:
                entries.append({"t": f"{rel} {b.start}–{b.end}줄", "g": repo["name"],
                                "u": f"{FILE_PAGES[f'{rid}:{rel}']}#B{b.start}", "x": strip_tags(render_md(b.text, ""))[:400]})

    (OUT / "code.html").write_text(page("KDAA 파일 목록", code_index(status), "", "code.html"), encoding="utf-8")
    for out_name, frag, title, _ in PAGES:
        body = (HERE / frag).read_text(encoding="utf-8")
        body = body.replace("<!--FILES-->", files_list(status))
        try:
            body = re.sub(r"\[\[([a-z0-9-]+)(?:\|([^\]]+))?\]\]", lambda m: inline(m.group(0), ""), body)
        except NotesError as e:
            errors.append(f"{frag}: {e}")
        (OUT / out_name).write_text(page(title, body, "", out_name), encoding="utf-8")
        if out_name == "index.html":
            (OUT / "_artifact_index.html").write_text(page(title, body, "", out_name, full=False), encoding="utf-8")
        for m in re.finditer(r'<(?:section|div)[^>]*id="([a-z0-9-]+)"[^>]*>\s*<h2>(.*?)</h2>', body, re.S):
            entries.append({"t": strip_tags(m.group(2)), "g": title, "u": f"{out_name}#{m.group(1)}", "x": ""})
    (OUT / "search.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    done = len(FILE_PAGES)
    total = sum(len(r["files"]) for r in REPOS)
    lines_done = sum(status[k]["lines"] for k in FILE_PAGES)
    lines_all = sum(v["lines"] for v in status.values())
    print(f"built {OUT}: {done}/{total} files annotated, {lines_done}/{lines_all} lines")
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", nargs="*", default=[], help="record the current source sha in these notes files")
    sys.exit(build(ap.parse_args().stamp))
