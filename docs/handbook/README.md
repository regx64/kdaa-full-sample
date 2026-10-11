# KDAA 코드 읽기 (핸드북)

네 저장소의 코드를 한 줄씩 설명하는 사이트입니다. 파이썬 기초 문법만 아는 사람을 대상으로 합니다.

```bash
python docs/handbook/build.py          # docs/handbook/site/ 생성
python docs/handbook/build.py --out /tmp/site   # 다른 폴더에 생성 (여러 사람이 동시에 검사할 때)
```

- 코드는 빌드할 때 저장소(kdaa 루트와 third_party/*)에서 그대로 읽습니다.
- 설명은 `notes/<repo>/<경로>.txt`에 `@@ 시작-끝` 블록으로 씁니다. 빈 줄이 아닌 모든 줄이 블록에 들어가야 빌드가 통과합니다.
- 각 설명 파일 첫 줄의 `sha:`는 설명을 확인했을 때의 소스 지문입니다. 코드가 바뀌면 빌드가 실패하므로, 설명을 고친 뒤 `python docs/handbook/build.py --stamp <경로>`로 지문을 갱신합니다.
- 설명 문법: 빈 줄로 문단, `- ` 목록, `` `코드` ``, `**굵게**`, `[[dataclass]]`(파이썬 개념 링크, `pages/python.html`의 id), `{{kdaa:kdaa/stages.py:172|글자}}`(다른 파일·줄 링크).
- `pages/`: 첫 페이지, 파이썬 개념, 참조. `assets/`: 공용 CSS·JS.
