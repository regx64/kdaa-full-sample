# KDAA 3-라인업 통합 기획 — 녹음부터 학습까지 자동화

대상 저장소 (기준 커밋)

| 역할 | 저장소 | 커밋 | 하는 일 |
|---|---|---|---|
| 녹음 | `regx64/sound-autodaq` | `c0823bd` | 키 입력 하나당 WAV 1개, 라벨 manifest, `--export-classism` |
| Mel | `aworse/meligation` (작업 중) | `7acb1bd` | WAV + onset → 100 ms 창 → 로그멜 `(64,48)` → `mel/<clip_id>.npy` |
| AFE | `regx64/classism` | `c6606a0` | 로그멜 → SmallCNN 38클래스 학습·평가·보정·int8 export |

---

## 구현 현황 (2026-10-11)

M0·M1(합성 데이터)까지 이 저장소에 구현했다. 사용법은 [`README.md`](README.md)에 있다.
- **완료:** 자동화 프로그램 `kdaa`(§4 단계·검사 게이트·증분 실행·리포트·`watch`), 세션 shard(§3 방식 B),
  CI(Linux 3.10/3.13 + Windows 3.12)
- **e2e 테스트:** 합성 세션 3개로 녹음 → 학습 1 epoch → 평가 → 리포트까지 통과했다.
- **classism 반영:** 학습 데이터 param_hash 검사, 세션 shard 읽기, `data.sessions`/`data.on_missing_mel`/
  `data.require_param_hash` 옵션을 classism에 넣었다(`regx64/classism` 브랜치 `claude/three-lineup-integration-plan-jkcqx2`,
  submodule은 이 커밋을 가리킨다. classism main에 합쳐지면 submodule을 main으로 옮긴다).
- **임시 처리:** B1·B2(meligation)는 `kdaa/mel_shim.py`가 맡는다. 설정값이 같은지 검증한 뒤 classism 해시를
  기록한다. meligation에 정식으로 반영되면 걷어낸다.
- **남은 것:**
  - 실제 녹음으로 하는 파일럿(M2, B3 onset 측정)
  - int8 export 실행 확인(TensorFlow 도구 필요)
  - 학습 중 실시간 추출과 데이터 변형(P2)

---

## 0. 요약

1. **세 저장소는 파일 레이아웃 기준으로 이미 연결되어 있다.** 합성 세션으로 확인했다:
   sound-autodaq export → meligation → classism `load_samples` 순서로 38클립이 문제없이 통과했다.
   빠진 것은 "한 번에 돌리는 지휘자(오케스트레이터)"와 몇 가지 계약 불일치뿐이다.
2. **블로커는 하나다. `param_hash`가 서로 다르다** (meligation `5711c12b`, classism `915acee6`).
   그런데 classism 학습 경로는 이 해시를 검사하지 않는다. 그래서 지금 상태로 돌려도 에러 없이 학습된다.
   스펙(NFR-2)이 막으려던 상황이 바로 이것이다.
3. **수만 개 소리 파일에서 문제는 용량이 아니라 파일 개수다.** 로그멜은 클립당 12 KB라서
   10만 클립이어도 약 1.2 GB이고, 전부 RAM에 올라간다. 그래서 **원본 WAV는 보존하고,
   학습 입력은 세션 단위 샤드(`.npz` 1개 = 세션 1개)로 묶어 한 번에 메모리에 올리는 방식**을 권장한다(§3).

---

## 1. 현재 데이터 흐름 (검증됨)

```
 sound-autodaq                       meligation (aworse)                classism
 ─────────────                       ───────────────────                ────────
 녹음 세션                            python code --in <root>            python -m src.train --data <root>
 data/raw/P01/S01/SESSION01/          │                                  │
   ├ audio/*.wav  (1 s, 48k, PCM16)   │ labels/*.csv 의 wav_onset_s 로    │ labels/*.csv + mel/<clip_id>.npy
   ├ manifest.jsonl                   │ -5 ms … +95 ms 컷 → 로그멜       │ → 세션 단위 split → SmallCNN
   └ session.json                     ▼                                  ▼
        │ --export-classism        <root>/mel/<clip_id>.npy          runs/<id>/best.pt, metrics.json
        ▼                          <root>/mel/_meta.json (param_hash)  → export_esp32 → deployment.json
 <root>/labels/<session_uid>.csv                                         (int8 scale/zp → Mel 에게 다시 전달)
 <root>/audio/<clip_id>.wav (하드링크)
```

- `<root>` 하나에 여러 세션이 쌓인다. 세션 하나 = CSV 하나 = classism split 단위 하나.
- 세 단계 모두 다시 실행해도 결과가 같다(멱등).
  export는 해당 세션 파일만 교체하고, mel은 원자적으로 덮어쓰며, train은 실행마다 별도 디렉터리를 만든다.
  그래서 자동화할 때 단계별로 재실행해도 안전하다.
- **검증 내용:** sound-autodaq의 `SyntheticBackend`로 38클래스 keypress 세션을 하나 만들었다(pre/post 500 ms).
  이어서 `export_session`, meligation `code --in`, classism `dataset.load_samples` + `LogMelDataset[0]`을 차례로 실행했다.
  결과는 38/38 생성, 0 skip, 텐서 `(1,64,48) float32`이다.
  해시 비교 결과는 `5711c12b` ≠ `915acee6`이다.

---

## 2. 통합 전에 정리할 것 (블로커/리스크)

### B1. `param_hash` 불일치 — **블로커**
- 원인: 두 저장소의 `Regime`은 값은 같지만 **필드 이름과 구성이 다르다.**
  - meligation: `win_length, hop_length, pre_ms, post_ms` (float)
  - classism: `win_samples, hop_samples, clip_ms, onset_pre_ms, normalization, dtype`
- 그 결과 sha256 입력이 달라져 해시가 다르게 나온다. meligation 코드 주석에도
  "classism 정의에 맞춰 `canonical_dict()`만 고치면 된다"고 적혀 있다.
- 해결:
  1. **classism `src/melio.py::Regime`을 해시의 단일 원본으로 정한다.** meligation은 `canonical_dict()`가
     classism과 똑같은 dict를 내도록 수정하고, 해시가 같은지 테스트로 고정한다.
  2. **classism 학습·평가에 해시 검증을 추가한다.** `mel/_meta.json`(또는 샤드 헤더)의 `param_hash`가
     `REGIME.param_hash()`와 다르면 학습을 거부한다. 지금은 checkpoint에 자기 해시를 기록만 한다.

### B2. meligation을 import할 수 없다
- 지금은 확장자 없는 단일 파일 `code`다. CLI(`python code --in`)로는 돌아가지만
  import할 수 없고, 버전도 고정할 수 없다.
- 요청할 것: 파일명을 `mel_build.py`(또는 `meligation/` 패키지)로 바꾸고 태그를 달아 달라고 한다.
  §3의 Phase 2(학습 중 실시간 추출·증강)와 온디바이스 레퍼런스 비교에 필요하다.

### B3. onset 정렬 (정확도 리스크, 실측 필요)
- 학습할 때는 **키보드 훅 타임스탬프**에서 자르지만, 배포할 때는 **음향 온셋 검출**로 자른다. 기준이 다르다.
- 창이 onset 앞 5 ms밖에 안 된다. 그래서 훅 시각이 실제 소리보다 몇 ms만 늦어도 누름(press) 순간이 잘려 나간다.
  Windows 훅 지연은 하드웨어에서 아직 재지 않았다.
- 녹음 파일은 onset 앞뒤로 500 ms씩 여유가 있다. 그래서 다음 두 가지를 할 수 있다.
  1. 파일럿 세션에서 "훅 시각 대비 음향 온셋 오프셋" 분포를 리포트한다.
  2. meligation에 **onset 재정렬** 단계를 둔다. `wav_onset_s` ±30 ms 안에서 **런타임과 같은 온셋 검출기**로
     시각을 다시 잡는다. 이렇게 하면 학습과 배포의 컷 기준이 같아진다.

### B4. 데이터 양과 split 조건
- classism의 session split(val 0.1 / test 0.2)은 **세션이 최소 3개** 있어야 돌아간다.
  participant split도 최소 3명이 필요하다.
- `configs/S01.yaml` 기준으로 세션 하나는 38클래스 × 500회 = **19,000 trial**이다.
  trial 하나가 약 2.5~3 s(카운트다운 1 s + 반응 대기 + post 0.5 s + 간격 0.5 s)라서
  **세션 하나에 13~16시간** 걸린다.
  세션 길이와 세션 수를 정하는 계획이 필요하다(§6 열린 질문).

---

## 3. 수만 개 소리 파일을 어떻게 입력할까 (핵심)

### 3.1 규모 계산
| 항목 | 값 |
|---|---|
| 세션당 클립 | 19,000 (38 × 500) |
| WAV 1개 | 1 s × 48 kHz × 16 bit ≈ 96 KB → **세션당 약 1.8 GB** |
| 로그멜 1개 (`.npy`) | 12,416 B → **세션당 약 236 MB, 파일 19,000개** |
| 10세션 | WAV 약 18 GB / mel 약 2.4 GB, **mel 파일 190,000개** |
| 추출 속도 (meligation, 단일 코어, 이 컨테이너 기준) | 클립당 약 1.0 ms → **세션당 약 20초** |

결론: 학습 데이터는 RAM(수 GB)에 다 들어가고, 추출도 빠르다.
**병목은 작은 파일 수십만 개를 다루는 비용이다.**
- 매 실행마다 파일 존재 확인과 개별 로드가 일어난다.
- Windows NTFS와 Defender 스캔 때문에 더 느려진다.
- 복사·백업도 어렵다.

### 3.2 선택지
| 방식 | 내용 | 장점 | 단점 |
|---|---|---|---|
| A. 클립별 `.npy` (현재 계약) | `mel/<clip_id>.npy` | 이미 동작, 디버깅 쉬움 | 파일 수십만 개, 첫 epoch 로딩 느림 |
| **B. 세션 샤드 `.npz`** | `mel/<session_uid>.npz` = `logmel (N,64,48)`, `clip_id (N,)`, `param_hash` | 세션=CSV=샤드 1:1, split 단위와 같음, 세션별 교체 쉬움, 파일 수 = 세션 수 | classism 로더에 약 30줄 추가 |
| C. 학습 중 WAV에서 실시간 추출 | DataLoader가 WAV를 읽고 meligation 추출기 호출 | **증강 가능**(onset 지터, gain, 잡음 합성) | B2(import) 해결이 먼저 필요, epoch마다 CPU 비용(세션당 약 20초) |
| D. HDF5 / LMDB / WebDataset | 대용량 포맷 | 수백만 클립 이상에서 유리 | 지금 규모에는 과함, 의존성만 늘어남 |

### 3.3 권장안
1. **원본 WAV는 손대지 않고 보존한다.** export는 하드링크로 하므로 복사본이 생기지 않는다.
   WAV는 재추출·증강·onset 재정렬의 원천이다.
2. **학습 입력은 B(세션 샤드)로 한다.**
   - 오케스트레이터가 meligation 출력 `mel/*.npy`를 세션별로 묶어 `mel_packed/<session_uid>.npz`를 만든다.
     헤더에는 `param_hash`, `mel_spec_version`을 포함한다.
     안정되면 이 기능을 meligation `--pack` 옵션으로 옮기도록 제안한다.
     그러면 다른 사람의 작업을 기다리지 않아도 된다.
   - classism `dataset.load_samples`는 샤드가 있으면 샤드를 읽고, 없으면 기존 클립별 `.npy`를 읽는다.
     기존 계약과 테스트는 그대로 유지된다.
   - 학습할 때는 전체를 한 번에 `np.load`해서 텐서로 메모리에 올린다. 아이템마다 파일 I/O가 없다.
3. **Phase 2에서 C(실시간 추출 + 증강)를 추가한다.**
   - onset ±2~5 ms 지터: 런타임 온셋 검출 오차에 강해지게 한다(B3 대응).
   - gain과 배경잡음 합성: far/noise 시나리오에 대응한다.
   - 증강 없는 B 결과를 베이스라인으로 두고 비교한다.
4. D는 클립이 수백만 개를 넘기 전까지 쓰지 않는다.

---

## 4. 자동화 파이프라인 설계

### 4.1 단계와 게이트
```
S0 녹음(사람)          sound-autodaq --config ...           → data/raw/<P>/<S>/<SESSION>/
S1 검증+export         --validate, --export-classism        → work/labels/*.csv, work/audio/*.wav
   G1: 검증 실패 세션은 export 거부(이미 구현됨). 미완료 세션은 "남은 trial 수" 경고
S2 Mel 추출            meligation --in work [--session X]   → work/mel/*.npy, _meta.json
   G2: _meta.param_hash == classism REGIME.param_hash()  (B1)
   G3: _skipped_<session>.csv 비율 ≤ 1% (넘으면 해당 세션 보류)
S3 패킹                npy → work/mel_packed/<session>.npz
S4 데이터 점검         classism src.dataset (coverage)      → 자모별 ≥25/참가자, 특수키 ≥25
   G4: coverage 미달이면 학습은 하되 "리포트 전용"으로 표시
   G5: 세션 ≥3 (split 가능 조건)
S5 학습                classism src.train                   → runs/<run_id>/best.pt, split.json
S6 평가+보정           classism src.evaluate                → metrics.json, calibration.json, AC-3/4/5 판정
S7 int8 export         classism src.export_esp32            → model_int8.tflite, deployment.json(scale/zp), AC-6
S8 기록·리포트         run 디렉터리에 데이터 매니페스트 + 세 저장소 커밋 + 지표 요약
```

### 4.2 오케스트레이터 (이 저장소 `kdaa-full-sample`)
- **형태:** 순수 Python CLI `python -m kdaa run`과 설정 파일 `pipeline.yaml`.
  - 학습 머신이 Windows이고 팀이 작다.
  - Snakemake/Airflow는 Windows 지원과 운영 부담 때문에 보류한다.
  - 데이터 버전 관리가 필요해지면 그때 DVC를 검토한다.
- **세 저장소 연결:** git submodule로 커밋을 고정한다(`third_party/{sound-autodaq,meligation,classism}`).
  run 기록에 세 커밋 해시를 남겨 재현성을 확보한다(NFR-4).
- **증분 실행:** 세션별 상태 파일 `work/state/<session_uid>.json`에 단계별 지문(fingerprint)을 저장한다.
  - S1 지문: `manifest.jsonl` 해시 + autodaq 커밋
  - S2 지문: S1 지문 + `param_hash` + meligation 커밋
  - 지문이 같으면 그 단계는 건너뛴다. 새 세션만 export/mel/pack 하고, 학습은 전체 샤드로 다시 한다
    (19k×N 클립 학습은 CPU에서도 수 분~수십 분 수준이다. 실측 필요).
- **run 디렉터리:** `runs/<YYYYMMDD-HHMM>_<datahash8>/`
  - `data_manifest.json`: 포함 세션, 클립 수, `param_hash`, 세 저장소 커밋
  - classism 산출물(`best.pt`, `split.json`, `metrics.json`, `calibration.json`, `eval_test/`, `export/`)
  - `report.md`: 이전 run 대비 top-1/top-5/keytype/ECE 변화, 게이트 결과
- **트리거:** 세 단계로 늘려 간다.
  1. 수동 `kdaa run`
  2. `kdaa watch data/raw`: 세션이 완료(exit 0 + validate 통과)되면 S1~S8 자동 실행
  3. 야간 스케줄
- **int8 scale/zp 전달(O-O5):** S7의 `deployment.json`을 meligation 쪽 온디바이스 설정으로 넘기는 경로를
  run 산출물로 고정한다.

### 4.3 디렉터리 (제안)
```
kdaa-full-sample/
  third_party/{sound-autodaq, meligation, classism}   # submodule, 커밋 고정
  kdaa/            # 오케스트레이터 (stages.py, mel_shim.py, pack.py, report.py, cli.py)
  pipeline.yaml    # 경로, 게이트 임계값, classism config 오버라이드
  data/raw/        # sound-autodaq 출력 (git 제외)
  work/            # labels/ audio/ mel/ mel_packed/ state/ (git 제외)
  runs/            # 학습 결과 (git 제외, 요약만 커밋 가능)
  tests/e2e/       # SyntheticBackend 세션 → S1~S5(1 epoch) 스모크, CI에서 실행
```

---

## 5. 저장소별 변경 목록

| 저장소 | 변경 | 우선 | 담당 |
|---|---|---|---|
| meligation | `canonical_dict()`를 classism `Regime`과 일치시킴 + 해시 일치 테스트 (B1) | **P0** | aworse |
| meligation | `code` → `mel_build.py`/패키지로 바꾸고 태그 (B2) | P0 | aworse |
| classism | 학습·평가 시 `mel/_meta.json`/샤드 `param_hash` 검증, 불일치면 거부 (B1) | **P0** | regx64 |
| classism | `load_samples`가 세션 샤드 `.npz`를 읽도록 확장(없으면 npy 사용) | P1 | regx64 |
| kdaa-full-sample | 오케스트레이터 S1~S8, 상태 파일, 패킹, 리포트, e2e CI | P1 | regx64 |
| meligation | onset 재정렬(런타임 검출기로 ±30 ms 재탐색) + 오프셋 리포트 (B3) | P2 | aworse |
| meligation | `--pack` 옵션(오케스트레이터 패킹을 이관) | P2 | aworse |
| classism | 실시간 추출 + 증강 DataLoader (§3.3-3) | P2 | regx64 |
| meligation | WAV 전체 대신 필요한 구간만 읽기(`wave.setpos`) — 선택 최적화 | P3 | aworse |
| sound-autodaq | 블로커 없음. 문서의 `aworse/classism` 표기를 `regx64/classism`으로 정정 | P3 | regx64 |

---

## 6. 마일스톤

- **M0 계약 정렬:** B1·B2 해결. 세 저장소 해시가 같은 것을 테스트로 고정한다.
- **M1 원커맨드 E2E:** 합성 세션 3개로 `kdaa run`을 돌려 S1~S6(1 epoch)이 통과하고, CI에 넣는다.
- **M2 파일럿 실데이터:** 실제 세션 1~3개로 B3 onset 오프셋을 측정하고, 샤드 로딩 시간과 학습 시간을 잰다.
- **M3 증분·자동 트리거:** `kdaa watch`, run 비교 리포트를 만든다.
- **M4 배포 루프:** int8 export(AC-6)를 하고, scale/zp를 meligation 온디바이스로 넘긴다. 증강 실험(Phase 2)도 한다.

## 7. 열린 질문 (결정 필요)

1. **학습 머신:** Windows + GPU 유무, 원본 WAV를 저장할 위치(세션당 약 1.8 GB). 원본을 얼마나 보관할지.
2. **세션 계획:** 세션 하나에 19,000 trial(13~16시간)을 유지할지, 짧은 세션 여러 개로 나눌지.
   split 조건상 세션은 최소 3개, 참가자 split은 최소 3명이 필요하다.
3. **meligation 담당자와 합의:** 해시 원본을 classism으로 둘지, 패킹 책임을 누가 가질지, onset 재정렬을 meligation에 넣을지.
4. **재학습 정책:** 새 세션이 들어올 때마다 다시 학습할지, N개 세션이 쌓이면 학습할지.
