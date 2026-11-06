# START HERE — [author]가 볼 것만

이 저장소는 두 달치 실험이 쌓여 있다. 폐기된 코드는 `_trash/` 로 옮겨 뒀고(삭제 아님, `_trash/README.md` 에 사유 표),
여기엔 **지금 살아 있는 것만** 적는다. 날짜는 2026-10-10 기준.

---

## 1. 보고서 (상금이 걸린 쪽 — 보고서 $5,000 > 1등 $3,000)

| 파일 | 무엇 |
|---|---|
| `report/draft.md` | 기술 리포트 본문 (영어, 워크숍 4~6쪽). 마감 **2026-11-15** |
| `report/plan_labelgen.md` | 자체 라벨 생성 계획서 (출처·라이선스·절차·비용) |
| `report/probe_pilot.md` | 프로브 파일럿 결과 |
| `JOURNAL.html` / `.en.html` | 일지 (대본 형식, 확인/예측을 배지로 구분) |

리포트의 중심 문장: **33문제로는 방법의 성패를 판정할 수 없다** (라벨 섞기 LOPO 95% 구간이 [0.13, 0.72]).
그래서 자체 라벨을 만들고 있다.

---

## 2. 지금 돌아가는 파이프라인 세 개

### (a) Kaggle 라벨 생성 — **큐 정지됨 (2026-10-10). 커널 2개만 마무리 중**
* `experiments/kaggle/queue_labelgen.sh` — 분리된 창(AIMO-QUEUE)에서 돌며 빈 슬롯이 생기면 커널을 올린다.
  5분 간격 재시도. **이 폴더 전체가 실행 중이라 건드리면 돌던 커널이 재시작될 수 있다.**
* `experiments/kaggle/genlab_v2.py` — 커널 본체. 선별(원본 k=4) → 변형(공식 3종 × k=4) → 라벨.
* 상태 보기: `tail experiments/results/labelgen_run.log`
* 제약: Kaggle **주간 GPU 30시간** · 동시 세션 2개 · 세션당 9시간.
* **대기 작업 하나**: DeepSeek `aimo-lg-run0-...` / `aimo-lg-run40-...` 두 커널이 COMPLETE 되면
  출력을 내려받아야 한다(큐가 정지됐으니 자동으로 안 받는다. Kaggle 출력은 만료되지 않으니 급하지 않다):
  ```
  .venv/Scripts/kaggle.exe kernels status [author-kaggle]/aimo-lg-run0-deepseek-r1-0528-qwen3-8b
  .venv/Scripts/kaggle.exe kernels output [author-kaggle]/aimo-lg-run0-deepseek-r1-0528-qwen3-8b       -p experiments/results/genlab/aimo-lg-run0-deepseek-r1-0528-qwen3-8b
  ```
  이 두 커널은 **cap 16384** 라벨이다(RunPod 는 32768) — `gen_cap` 으로 구분된다.

### (b) RunPod 유료 GPU — 지금 주력
* 파드 띄우기 **전에** 반드시: `.venv/Scripts/python.exe analysis/payload_symbols.py`
  (Kaggle 커널은 여러 파일을 한 네임스페이스로 붙여 돌리므로 payload 에 빠진 심볼이 생긴다 —
   안 잡으면 모델까지 올린 뒤 NameError 로 죽어 돈이 나간다)
* `experiments/runpod/launch.py` — 파드 생성 → 업로드 → 실행 → 결과 회수 → **반드시 terminate**.
  하드캡 **$95**(최악 비용 기준 사전 거부) · 수명 상한 감시 · 종료 확인.
* 누적 비용 보기: `.venv/Scripts/python.exe experiments/runpod/launch.py --ledger`
* 떠 있는 파드 전부 정리: `... launch.py --reap`
* `RUNPOD_API_KEY` 는 Windows 사용자 환경변수에서 읽는다. **값은 출력·로그·커밋되지 않는다.**

### (c) 제출물 두 개
| 폴더 | 방법 | 상태 |
|---|---|---|
| `solutions/h22/` | **라벨 정의 재현** — P(원본 안정) × (섭동 과반 오답 비율), 절대 임계값 0.5 | 현재 주력 |
| `solutions/hprobe/` | 사전등록 프로브 (#77) — forward 1회 hidden state | 가중치 대기 중 |

빌드: `.venv/Scripts/python.exe scripts/package.py h22` → `dist/h22-small.zip`
제출 전 필수: `.venv/Scripts/python.exe analysis/smoke_submission.py dist/h22-small.zip`
**업로드는 [author]만 한다.**

---

## 3. 라벨·문제 데이터가 어디 있나

### 문제지는 **파일이 아니다** — 커널이 런타임에 결정적으로 재구성한다
`experiments/kaggle/genlab_v2.py` 의 `load_pool()` 이 매번 두 곳에서 받아 `seed=0` 순열로 뽑는다:

| 출처 | 라이선스 | 후보 수 |
|---|---|---|
| `AI-MO/aimo-validation-aime` (AIME, 정수 정답) | Apache-2.0 | **90** |
| `EleutherAI/hendrycks_math` Level 5, `\boxed` 가 정수 | MIT | **724** |
| | | 합 **814** (train-main-v2 와 중복 0) |

MATH Level 5 과목별: algebra 185 · intermediate_algebra 148 · number_theory 141 · prealgebra 116 ·
counting_and_probability 53 · geometry 44 · precalculus 37.

커널은 이 814개 중 `--pool-offset` 으로 N개씩 잘라 쓴다(현재 큐: N=40, offset 0/40/80 = 모델당 120문제).
**같은 seed 라 재현된다** — 문제지를 저장해 둘 필요가 없다.

### 완성된 라벨
| 파일 | 내용 |
|---|---|
| `experiments/results/genlab/skywork_labels_merged.json` | **Skywork 88건** (robust 62 / spurious 26) · AIME 4 + Level 5 84 |
| `experiments/results/train-main-v2.parquet` | 주최측 학습셋 82행 → **문제 단위 라벨 33개** (robust 20 / spurious 13) |

라벨 한 건의 필드: `problem · gold · level · subject · base_acc · var_acc · var_detail(n_wrong/n_unfin/n_fin/n_hit)
· max_drop · label · label_fin · accepted · pidx`.
**라벨이 두 개인 이유**: `label` 은 미완주를 오답으로 세고(주라벨, #77 에서 고정), `label_fin` 은 완주한
샘플만 분모로 쓴다(대조). 둘의 일치율이 90% 다.

DeepSeek·Olmo 라벨은 아직 없다 — 커널이 돌고 있다.

---

## 4. 지금 열려 있는 질문 하나

**리더보드 상위 4명이 전부 86%(12/14)에 몰려 있다.** 논문 정의(섭동 과반 오답 다수결)를 그대로 재현하면
그 점수가 나오는가 — `solutions/h22` 가 그 질문이고, RunPod 에서 공식 33문제로 재고 있다.
우리 이전 제출들은 전부 0.43~0.57 이었다(always-true 0.50 과 구분 불가).

---

## 5. 절대 규칙 (CLAUDE.md 전문 참조)

1. 실전 입력은 `model_id` 와 문제 텍스트 **둘뿐**이다.
2. Small 트랙 ZIP 은 루트에 `small.txt` 필수, Main 은 있으면 실패.
3. 반환은 **순수 파이썬 `bool`** (`np.bool_` 면 그룹 전량 무효).
4. 미지의 `model_id` 에 `KeyError` 가 나면 그룹 전체가 날아간다.
5. 모델 로드는 배치당 1회.
6. **점수를 보고 설계를 바꾸지 않는다** (사전등록 — 결과가 반대로 나오면 반대로 적는다).
7. 다른 참가자의 제출물 내부는 열지 않는다.
8. **업로드는 [author]만.** push 는 [author]가 말할 때만.
