# _trash — 지금 파이프라인이 쓰지 않는 코드

**삭제한 것이 아니다.** 원래 경로 구조를 그대로 유지해 옮겨 뒀다(`_trash/analysis/...` 식).
되살리려면 `git mv _trash/<경로> <경로>` 한 줄이면 된다. git 히스토리도 그대로다.

## 왜 옮겼나

두 달 동안 가설을 스무 개 넘게 세우고 대부분 기각했다 — 그게 이 프로젝트의 결과물이다(리포트의
negative results). 그런데 폐기된 코드가 현재 코드와 같은 디렉터리에 섞여 있어서 파일 목록만 보고는
무엇이 지금 돌아가는 것인지 알 수 없게 됐다. 그래서 **지금 파이프라인이 import·호출·참조하는 것만**
제자리에 남기고 나머지를 여기로 옮겼다.

## 옮기는 기준 (기계적으로 적용했다)

현재 파이프라인(`experiments/kaggle/` · `experiments/runpod/` · `genlab_v2` · `variants_*` · `probe_pilot*` ·
`solutions/h22` · `solutions/hprobe` · `analysis/` 의 현행 스크립트 · `report/` · `docs/` · `scripts/`)의
**코드와 문서 전체를 한 덩어리로 읽어** 후보의 모듈명·파일명을 **단어 경계로** 검색했다.
한 번이라도 걸리면 옮기지 않았다. 참조가 0인 것만 옮겼다 — 후보 118건 중 **72건 이동 · 31건 보류**.

* `experiments/kaggle/` 는 **AIMO-QUEUE 가 실행 중**이라 손대지 않았다.
* `experiments/results/` (라벨·문제 데이터)는 후보에서 제외했다.
* `report/`·`docs/` 가 이름을 언급하는 스크립트는 전부 **보류**했다 — 리포트 숫자의 출처라서다.
* `CLAUDE.md`·`JOURNAL` 의 언급은 **역사 기록**이라 차단 조건으로 쓰지 않았다
  (그 둘은 거의 모든 스크립트를 언급한다 — 의존성으로 보면 아무것도 못 옮긴다).

* 처음 집계에서 `solutions/*/solution.py` 가 전부 '참조됨' 으로 나왔는데, 그건 `solution.py` 라는
  **일반 문자열**이 keep 쪽 문서에 있었던 오탐이었다. 그래서 제출물은 **디렉터리 이름** 단위로,
  나머지는 단어 경계로 다시 쟀다(`probe_lib` 가 `hprobe_lib` 의 부분문자열로 걸리던 것도 같은 오탐).

## 옮긴 것 (72건)

| 원래 경로 | 어떤 가설·용도였나 · 왜 폐기 |
|---|---|
| `analysis/build_compass_artifacts.py` | compass 가중치 빌드 |
| `analysis/build_h15_artifacts.py` | h15 임계값 빌드 |
| `analysis/build_h20_artifacts.py` | h20 임계값 격자 빌드 |
| `analysis/cap_simulation.py` | 토큰 상한 시뮬. '신호가 문제의 성질이 아니라 예산의 함수' 임을 보인 계산 |
| `analysis/compass_smoke.py` | compass 제출물 스모크 |
| `analysis/count_loads.py` | 업스트림 trained-probe 의 '루프 안 모델 로드' 버그 확인 |
| `analysis/cpu_throughput.py` | CPU 처리율 실측(forward 51s). GPU 확인 후 불필요 |
| `analysis/data_inventory.py` | 초기 데이터 재고 조사. docs/DATA.md 로 정리됨 |
| `analysis/explain_score.py` | 초기 점수 설명기 |
| `analysis/genlab_budget.py` | 라벨 생성 v1 예산 |
| `analysis/genlab_v2_budget.py` | 라벨 생성 v2 예산. 실측으로 대체 |
| `analysis/h14_budget.py` | H14 예산 계산기. H14 폐기와 함께 |
| `analysis/h14_eval.py` | H14 길이비. AUC 0.400 — 사전등록 부호와 반대 |
| `analysis/h14b_eval.py` | H14b 저비용 대리지표. 0.427/0.436/0.338 전부 null 안 |
| `analysis/h15_external.py` | H15 외부검증. 자체 v3 는 방향 유지, MATH137 에서 무너짐 |
| `analysis/h15_rename_density.py` | H15 기호 밀도. log(단어수) 잔차 후 null 안 |
| `analysis/h20_grid.py` | H20 설정 격자. h20 계열 종료(실전에서 2모델이 상한 포화) |
| `analysis/h20_original_only.py` | H20 원본만 풀기. train 0.850 이었지만 리더보드 0.43 |
| `analysis/h20e_coverage.py` | h20e 커버리지(혼합안). 예산 안에서는 텍스트 단독을 못 넘음 |
| `analysis/h20e_time.py` | h20e 시간표. h20e 종료와 함께 |
| `analysis/h21_combo.py` | H21 combo(사후 조합). 사전등록이 아니라 주장하지 않음 |
| `analysis/h21_eval.py` | H21a~d·H21p 평가. H21p 결합이 단독보다 나빴다 |
| `analysis/h21c_confirm.py` | H21c 확증 검정(자체 88문제). 0.594 · null 안 — 재현 실패 |
| `analysis/h21c_time.py` | h21c/h21cK 시간표 |
| `analysis/h21ck_eval.py` | H21cK K 곡선·부트스트랩. '신호 미입증' 판정의 근거 |
| `analysis/h3_pooled.py` | H3 문제 풀링. 길이만 null 밖(그것도 ▼ = 길수록 spurious) |
| `analysis/h5_relative.py` | H5 상대 친숙도(모델 간 NLL 차). 주 지표 전환용이었으나 신호 없음 |
| `analysis/infer_task_size.py` | 본선 task 규모 추정. 공지로 확정되어 불필요 |
| `analysis/leaderboard_luck.py` | 리더보드 운 계산. 0.86 세 명은 운으로 설명 가능 |
| `analysis/leakage_permutation_type.py` | permutation_type 라벨 누출 확인(28/28 완벽 분리). 업스트림이 제거함 |
| `analysis/length_confound.py` | 길이 교란 진단. MATH137 프로브가 길이를 배웠음을 밝힌 스크립트 |
| `analysis/median_split.py` | H1 배치 중앙값 분할. 어떤 점수도 null 을 못 넘음 |
| `analysis/model_prior.py` | model_id 사전확률. 검증셋이 균형이라 설계상 무력 |
| `analysis/permutation_check.py` | 셔플 null 점검기(옛 버전) |
| `analysis/perturbed_text_bias.py` | 변형 텍스트 편향 점검 |
| `analysis/sensitivity_confound.py` | 민감도 프로파일 1차(내부 표현 코사인). 측정 도구 쪽에서 기각 |
| `analysis/topic_confound.py` | 주제 교란 진단. 같은 결론 |
| `analysis/what_makes_true.py` | 초기 탐색 — 무엇이 True 를 만드나 |
| `analysis/why_fractional.py` | 초기 탐색 — 분수 정답 문제 |
| `experiments/extract_confidence_v2.py` | 자신감 특징 추출. 신호 없음 |
| `experiments/extract_train_v2.py` | train-v2 특징 추출. probe_pilot.py 로 대체 |
| `experiments/extract_trajectory.py` | 궤적 추출기. H10 계열 종료 |
| `experiments/feature_bank.py` | 특징 모음 실험 |
| `experiments/full_solve_probe.py` | 전체 풀이 프로브. 노트북으로 완주 불가 판정 |
| `experiments/genlab_local.py` | 라벨 생성 로컬판. Kaggle 경로로 대체 |
| `experiments/genlab_local_screen.py` | 라벨 생성 로컬 선별. 같은 이유 |
| `experiments/h10_compass.py` | H10 나침반(H9 가중치). 풀링 잔차 0.542, null 안 |
| `experiments/h10_extract.py` | H10 벡터 추출기. H10 계열 종료와 함께 |
| `experiments/h11_knockout.py` | H11·H11b 표면 토큰 차단. 차단 효과 자체가 ~0 이라 측정이 무효 |
| `experiments/h12_patch.py` | H12 교환 개입. H11 무효 판정과 함께 폐기 |
| `experiments/h13_subspace.py` | H13 표면 부분공간 PCA. 풀링 잔차 0.512, null 안 |
| `experiments/h16_prefill_pool.py` | H16 prefill 풀링 프로브. AUC 0.531 — 33문제로 수천 차원은 불가 |
| `experiments/h18_h19.py` | H18·H18b·H19 암기 신호 3종. 셋 다 null 안, H19 는 부호 반대 |
| `experiments/h21ck.py` | H21cK 측정(상수 변형 K개). h21c 의 0.802 가 운이었음을 보인 실험 |
| `experiments/h4_external.py` | H4 외부 검증(MATH137 -> v2). 최고 AUC 0.73 도 null 안 — 신호 없음 |
| `experiments/h5_memorization.py` | H5 암기 신호(NLL·Min-K). 길이 잔차 후 null 안 |
| `experiments/h6_internal_shake.py` | H6 내부 흔들기. 사전등록 부호와 **반대**로 나와 폐기(뒤집지 않음) |
| `experiments/h8_surface_dependence.py` | H8 표면 의존도(backward). 전부 null 안 |
| `experiments/train_confidence_v2.py` | 자신감 특징 LOPO. 신호 없음(일부는 anti-learning) |
| `experiments/train_instrument.py` | 공용 계측기 학습(inst_probe). 문제 단위 예측은 구조적으로 0.50 |
| `experiments/train_probe.py` | 초기 프로브 학습. build_hprobe_artifacts.py 로 대체 |
| `experiments/train_train_v2.py` | train-v2 프로브 LOPO. 신호 없음 |
| `experiments/train_trajectory.py` | 궤적 분류기 학습. H10 종료 |
| `solutions/h15C2` | h15C 변형 제출물 |
| `solutions/h20b` | h20 디코딩 경로 실험 제출물 |
| `solutions/h20c` | h20 시간모델 v1 제출물(DEPRECATED.md 로 폐기 선언) |
| `solutions/h20d` | h20 시간모델 v2 제출물. h20e 로 대체 |
| `solutions/inst_probe` | 사전등록 제출물 A(공용 계측기). 대조 실험용, ~0.50 확인 |
| `solutions/length_rule` | 사전등록 제출물 B(길이 규칙). 같은 목적 |
| `solutions/probe_tput` | 워커 처리율 측정용 제출물 |
| `submissions/always-false` | 상수 False 제출물(베이스라인) |
| `submissions/solution-always-false.zip` | 상수 False 제출물 zip |

## 보류한 것 (참조가 남아 제자리에 둔 31건)

`solutions/` : compass · h15A · h15B · h15C · h20 · h20e · h21c · h21cK · h21combo · probe · probe2 · shake · trajectory

`analysis/` : columns · h20m_eval · label_rule · labelgen_pilot_eval · model_confound · probe_counts · probe_field · show_variants

`experiments/` : extract_hidden · gen_labels · h10b_eval · h10b_ext · h10c_eval · h21_uncertainty · h9_essence · mini_sim · replicate_labels · sensitivity

이유는 셋 중 하나다: (a) `experiments/kaggle/` 의 체인·감시 스크립트가 부른다,
(b) `report/`·`docs/` 가 숫자의 출처로 이름을 적어 뒀다, (c) 현재 제출물이 코드를 참조한다
(예: `solutions/hprobe/hprobe_lib.py` 가 h15A 의 `n_sym` 을 텍스트 폴백으로 쓴다).

