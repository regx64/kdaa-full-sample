# kdaa-full-sample

녹음(sound-autodaq) → Mel(meligation) → 학습(classism)을 명령 하나로 돌리는 자동화 프로그램.
설계와 배경은 [`PLAN.md`](PLAN.md)를 본다.

```
python -m kdaa run       # 새/바뀐 세션만 처리하고, 데이터가 바뀌었으면 학습·평가·리포트까지
python -m kdaa status    # 세션별 단계 상태와 최근 학습 결과
python -m kdaa watch     # run을 주기적으로 (녹음이 끝난 세션을 자동으로 집어 감)
```

## 설치

```bash
git clone --recurse-submodules <this repo>        # 이미 받았다면: git submodule update --init
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU 머신이면 해당 CUDA 휠
pip install -r requirements.txt
# int8 export까지 하려면: pip install -r third_party/classism/requirements-export.txt
```

세 저장소는 `third_party/`에 git submodule로 커밋이 고정되어 있다. 올리려면
`git -C third_party/<repo> pull origin main` 후 submodule 커밋을 함께 커밋한다.
학습 결과(`runs/<id>/run.json`)에 세 커밋이 기록된다.

## 흐름

1. sound-autodaq로 녹음한다. 출력 루트가 `pipeline.yaml`의 `paths.raw`(기본 `data/raw`)와 같아야 한다.
2. `python -m kdaa run`을 실행한다.

| 단계 | 하는 일 | 산출물 |
|---|---|---|
| export | 완료된 세션을 검증하고 classism 형식으로 내보냄 (`--export-classism`) | `work/labels/<세션>.csv`, `work/audio/*.wav` |
| mel | meligation으로 로그멜 추출 | `work/mel/<clip_id>.npy`, `_meta.json` |
| pack | 세션별로 묶음 | `work/mel_packed/<세션>.npz` |
| check | classism `src.dataset`: param_hash 검사, coverage / kdaa: 세션 수 | `work/check.json` |
| train | classism 학습 | `runs/<id>/best.pt`, `split.json` |
| evaluate | 보정 + 평가 | `runs/<id>/eval_test/`, `calibration.json` |
| export_int8 | TFLite int8 (도구가 설치된 경우) | `runs/<id>/export/`, `handoff/mel_int8_input.json` |
| report | 이전 실행과 비교 | `runs/<id>/report.md`, `run.json` |

- **바뀐 것만 다시 한다.** 세션마다 입력 지문을 `work/state/<세션>.json`에 저장하고, 같으면 건너뛴다.
  사용 가능한 세션 묶음과 classism 설정이 지난 완료 실행과 같으면 학습도 건너뛰고 그 실행을 재사용한다.
  강제로 다시 하려면 `--force`(export/mel/pack), `--force-train`(학습)을 쓴다.
- **검사 게이트**에 걸리면 멈추고 종료 코드 2를 돌려준다. 실패는 1이다.
  - 녹음이 끝나지 않은 세션은 제외한다(`sessions.include_incomplete`).
  - Mel이 클립을 1% 넘게 건너뛴 세션은 보류한다(`gates.max_skip_rate`).
  - shard의 param_hash가 classism과 다르면 학습을 거부한다.
  - session split에 세션이 3개 미만이면 멈춘다(`gates.min_sessions`).
  - coverage(자모별 25개)가 모자라면 학습은 하되 리포트에 "참고용"으로 표시한다(`gates.require_coverage: true`면 멈춤).

## meligation 임시 처리 (meligation에 반영되면 제거)

`kdaa/mel_shim.py`가 meligation의 `code` 파일을 경로로 불러온다. 설정값이 classism `Regime`과 항목별로 같은지
확인한 뒤 classism의 param_hash(`915acee6`)를 `_meta.json`에 기록하고, 값이 하나라도 다르면 멈춘다.
meligation 자체 해시가 classism과 같아지면 그대로 쓰고 이 처리는 아무것도 하지 않는다. (PLAN.md B1·B2)

classism은 직접 쓴다. 실행마다 `runs/<id>/classism_config.yaml`을 만들어 사용할 세션(`data.sessions`),
Mel이 건너뛴 클립 제외(`data.on_missing_mel: drop`), param_hash 필수(`data.require_param_hash: true`)를 넘긴다.
shard 읽기와 해시 검사는 classism `src/dataset.py`가 한다.

## 테스트

```bash
python -m pytest -q        # tests/test_e2e.py: 합성 세션 3개로 녹음 → 리포트까지 (약 1분, torch 필요)
```
