# Discord에서만 확인되는 규칙 · 상태 (발췌, 원문 그대로)

## 2026-10-08 ~ 10-11 신규 (지휘 전달 요약 — **원문 아님**)

**이 절은 [author]가 전달한 요약이다.** 아래 항목을 설계 근거로 쓸 때는 원문 확인이 필요하다
(9/28 요약의 '시간제한 문제당' 이 공식 FAQ 8 로 뒤집힌 선례가 있다 — 지휘 #44).

### (1) 10/9 공지 — **배치 균형 가정에 기댄 방법은 신뢰 불가**
상위 제출 다수가 **'라벨 균형 분포'**(배치 내 순위 매기기 · 반반 분할)에 기대고 있고,
그건 **과적합**이라 최종 테스트에서 신뢰할 수 없다. 주최측이 **문제별 평가 도입을 검토 중**이다.
함께 상기된 9/16 공지: **문제 특징 기반 방법은 새 문제에서 랜덤 이하**.
→ 우리 기록과 맞는 방향이다(지휘 #8·#64: 문제 단위 예측은 모델별 균형 셋에서 0.50 벽).
   우리 제출 중 **중앙값/순위 분할에 기댄 것**: `inst_probe`(A) · `length_rule`(B) · `shake`(C) ·
   `h20` · `h20e` · `h21c` · `h21combo` · `h21cK` · `hprobe`.
   **절대 임계(배치 무관)인 것**: `h15A` · `h15B` · **`h15C`** · `compass`(모델별 고정 임계) ·
   `h22`(임계 0.5) · **`sub2048`**(모델별 train 점수 중앙값).

### (2) 최종 테스트 규모·시간
문제 **30~50개** · **대부분 미공개 문제** · 샘플당 시간 상한 동일(**약 170초/샘플**).
→ 9/28 요약의 '30~40개' 보다 넓어졌다. '샘플당 170초' 는 **전체 합산 3600초**(공식 FAQ 8 ·
   getting-started README · Discord 9/9 세 출처 일치)와 **다른 단위**이므로, 어느 쪽이 지배하는지
   원문 확인이 필요하다. 170s × 샘플 수로 읽으면 21샘플에서 3,570s 로 3600s 와 거의 같아진다.

### (3) 지명·제출 한도
최종 지명 **5~10개**(9/28 요약의 2~5개에서 늘었다) · **마감 전 선택** · 일일 제출 **10회** ·
시각 기준 **CEST**.

### (4) 제출 로그 — **stdout/stderr 열람 가능**
→ 2026-10-12 확인: 공개 API 로는 안 된다. `/api/submissions/<id>/` 가 인증 없이 **404**
  (`No Submission matches the given query`). **로그인한 [author]만** 받을 수 있다.

### (5) 보고서
NeurIPS 템플릿 **1~4쪽** · **보고서당 동료평가 의무**.
→ `report/draft.md` 는 워크숍 4~6쪽 가정으로 썼다 → 분량 재조정 필요.

### (6) 활성화(activation) 제공 계획 **취소** — 3 TB 규모 때문.

### (7) 자체 라벨 허용 · **누출 방지 제외 목록**
제외: **AIME 2024 / 2025 / 2026** · **AIMO2 · AIMO3 공개 문제** · **JMO** ·
`problems-public` · **`train-main-v2` 유사문제**.

### (8) 다른 참가자가 같은 함정 둘을 질문했다
**DeepSeek 토크나이저 공백 문제**와 **boxed 파서 폴백**. 주최측 **라벨 생성 코드는 미공개**.
→ 우리가 이미 돈으로 배운 둘과 같은 것이다: `Ġ`/`Ċ` 디토큰화 손상(#TOK-AUDIT, 사고 비용 $57.24)과
  `extract_answer` 가 `\[ \boxed{} \]` 를 못 뽑는 결함(2026-10-12, 규칙 ② 가 잡음).
  **우리 둘 다 측정해 뒀고 수치가 있다** — 리포트에서 '라벨 생성 함정' 절의 1차 자료다.

### 확인 결과 (지휘 (a)(b), GPU 0 · 2026-10-12)

**(a) 우리 두 제출물은 배치 순위·균형 분포에 기대지 않는다.**
* `h15C`: 점수 = −log(단어수), 임계 **−4.0840** 으로 **절대 고정**(train 두 클래스 중앙값의 중점).
  다른 문제의 점수가 한 문제의 판정에 영향을 주지 않는다. **다만 '문제 특징만 보는 방법' 이라
  9/16 공지에 정면으로 걸린다** — 리더보드 0.57 은 그 공지대로 최종셋에서 무너질 수 있다.
* `sub2048`: **모델별 절대 임계**(Skywork +3.357 · Olmo +2.511 · DeepSeek +2.707) · 그룹 크기 무관 ·
  순위 계산 없음. 폴백도 h15C(절대 임계)다. 순위 분할 경로는 `SUB2048_FALLBACK=nsym` 뿐이고 기본 꺼짐.
  **정직하게 적어 둘 한 가지**: 그 임계를 *고른 동기* 는 균형 가정이었다 — 로지스틱 자연 경계(0)가
  True 를 81% 찍어 상수 예측에 가까워지므로 **train 점수 중앙값**(True 0.495)으로 잡았다.
  예측 자체는 배치와 무관하지만, **불균형 최종셋에서는 True 비율 ~50% 고정이 최적이 아니다.**

**(b) 우리 학습 문제 출처에 제외 대상은 없다 — 다만 구조적 위험이 남는다.**
(`analysis/source_leak_audit.py` → `experiments/results/source_leak_audit.txt`)
* `AI-MO/aimo-validation-aime` 는 **90문제 · 연도 2022/2023/2024 각 30** = **2024 가 33%** 다
  (2025·2026 은 없다). **즉 이 데이터셋 자체는 제외 대상을 포함한다.**
* **그런데 우리가 라벨을 만든 문제는 전부 MATH level 5 이고 AIME 는 0개다.** 두 독립 증거:
  (ㄱ) `level` 필드가 전 행 `Level 5`(Skywork 38 · DeepSeek off40 35 · #TEMP-SENS 20 전부)
  (ㄴ) AIME 90문제 텍스트와 정규화 대조 **일치 0건**.
  이유는 `load_pool` 구조다 — AIME 를 rows 앞쪽에 쌓고 permutation 후 **정렬**하므로 AIME 가
  **pidx 앞 구간에 몰린다**. 우리 k=8 라벨은 전부 **offset 40~119** 구간이라 MATH L5 만 들어왔다.
* **구조적 위험**: `load_pool` 은 **연도를 보지 않는다**. `--offset 0` 쪽 슬라이스를 쓰거나 풀 크기를
  바꾸면 **AIME 2024 가 섞인다**. 앞으로 라벨을 더 만들려면 연도 필터를 넣어야 한다(미구현).
* MATH level 5 는 제외 목록에 없다(MIT 라이선스, 지휘 #70 에서 확인).
* **`train-main-v2` 는 완전 일치만 제거한다**(`problem not in v2`). '유사문제' 는 걸러지지 않는다.
  8-그램 자카드 최댓값이 전 묶음 **0.000** 이지만 **8-그램은 근사 중복 검출력이 약하다**(서로 다른
  문제끼리는 거의 항상 0) → 더 짧은 n-그램이나 임베딩으로 다시 재야 한다. **아직 안 했다.**

## 2026-09-28 #qa 요약 (지휘 전달, 원문 아님)
- 최종 테스트셋: 새 문제 30~40개, 모델 동일, 균형 보장 없음 → 중앙값 분할 금지, 고정 임계값으로
- 시간제한: 문제당 기준 유지되도록 조정
- are_robust 안에서 다른 캐시 모델 로드·비교 허용
- prebuilt wheel 허용 · 최종 지명 제출 2~5개
- (9/26) train-main-v2로 추가 학습데이터 생성 자유 답변 → 라벨 생성 금지 판단 보류([author] 확인 전까지 미사용)

## 2026-09-17 ~ 09-26 #qa 요약 — 전달받은 요약이라 원문 아님 (9/26)

**학습셋**
- `train-main-v2`(82행)는 **더 확장 안 함**. 예외: Skywork 몇 개만 다음 주 이후 추가 예정.
- val/test의 임계값은 학습셋의 `≤0.10 / ≥0.25`와 **같거나 더 엄격**.
- 학습 라벨은 안정성 테스트를 거쳐 신뢰도 높음. `max_drop` 연속값은 경계 실험용.
- 옛 `math-robust-final.csv`(137문제)는 LLM 생성이라 **오류 있음 — 주최측 인정**.
  → 우리가 9/14에 찾은 "domain 변형이 짧은 문제를 파괴" 아티팩트와 일치.
- `problems-public` 비공개 확정.

**방법론**
- 평가 기준: **한 방법이 여러 모델에 일반화되는가.**
- 리더보드 외에 기술 리포트의 과학적 기여에 상금 비중 큼.
- 적대적 데이터 생성 코드 공개 철회(위 9/23 항목). 실제 변형 종류는 논문보다 많음.
- **Skywork 3문제로 찾은 피처가 검증셋에서 정반대로 나온 사례 보고됨** → n=3 모델별 튜닝 금지.

**운영**
- 제출 시 Main/Small **하나만** 선택. 둘 다 고르면 점수 0 / n/a.
- 대회 웹사이트에 정리된 리더보드 생김.
- 컴퓨트 지원 신청서: 핵심 아이디어 + 용도만 쓰면 됨 (AIMO3 팀명 칸은 형식적인 듯).

## 2026-09-23 — 라벨 생성 코드는 공개 안 함 · 전문가 변형 유형이 추가됨

질문 (강민서): 논문이 약속한 "LLM 기반 적대적 학습 데이터 생성 코드베이스"는 어디 있나.

> **Štefánik:** taking into account the reviews for the proposal, we have decided to **withdraw
> from publishing the exact code that can be used for reproducing our labelling scheme**. We hope
> that this will allow us to more reliably test the generalisation of the submitted methods …
> there is nothing stopping you from creating a codebase and generating LLM-based perturbations
> for other problems, similar to the ones that we did in the proposal. Just note that **the list
> of perturbations we presented in the proposal was extended for principally new ones from our
> mathematical experts.** We will be happy to share more after the competition is over.

→ 논문 §2.2의 코드 공개 약속 **철회**. 라벨링 절차를 그대로 재현하는 건 의도적으로 막음.
→ 변형 유형이 논문의 7종(rephrase·rename·domain·distract·typos·expert 2종)보다 **더 있다.** 어떤 것인지 비공개.
   → 우리 변형(rename·typos·number…)으로 라벨을 재현해도 주최측 라벨과 **정의가 다를 수 있음**. mini_sim 결과 해석 시 주의.
→ 직접 변형을 만들어 학습 데이터로 쓰는 건 여전히 허용(9/5 답변과 일관).

## 2026-09-22 공지 — 리더보드 초기화 · 시그니처 변경 · 학습셋 공개

> @everyone The competition leaderboard has now been updated with the new leaderboard and
> validation set labels **balanced per problem and per model**. We've cleared the leaderboard
> and will try to rerun your existing solutions to repopulate it. … we encourage you to manually
> resubmit your best solutions.
> Note that the submission interface newly exposes reasoning_effort: str, so the new full
> submissions signature is:
> `def are_robust(model_id: str, problems: list[str], reasoning_effort: list[str]) -> list[bool]`
> We also release the new complementary train set …

→ **공지의 시그니처는 코드와 다르다.** `ingestion.py`(커밋 7e05b26)는
  `are_robust(model_id, reasoning_effort=effort, problems=problems)` 로 **키워드** 호출하고,
  `reasoning_effort`는 배치당 **문자열 하나**다 (배치 = (model_id, effort) 쌍). README도 그렇게 적혀 있다.
  파라미터 **이름**이 `reasoning_effort`여야 3-인자 경로를 탄다. 없으면 옛 2-인자 경로(effort 무시).
→ README는 값이 `"default" | "low" | "medium"`이라 하지만 train-main-v2엔 **`"high"`도 있다** (gpt-oss-120b 18행).
→ 학습셋 `aimo-interp/train-main-v2` (9/17): 82행 · 5모델 · 36문제 · 컬럼 `model_id, reasoning_effort, problem, is_robust, max_drop`.
→ 새 task id: Main **36237** (N=21) · Small **36241** (N=14). 옛 35766/35767 점수는 사라짐.

공식 README·논문에 없고 Discord 답변에만 있는 것들. 날짜는 Discord 표시 기준(KST).
주최측 발언자: Michal Štefánik, Michal Spiegel, PMMon, JosefKuchar.

---

## 런타임에 변형본 만들어 라벨 절차를 재현하는 방법 — 권장 안 함 (2026-08-03)

질문 (runfme): 제출물이 hidden-test 문제의 변형본을 생성하고, 해당 model_id를 돌려서,
원본과 변형본 답의 일치도로 robust를 예측해도 되는가.

> **Štefánik:** We were hoping to publish the perturbation_types to maintain transparency
> of the labelling, but it is true that this also makes the label hacking much easier.
> **This kind of methods is not what we want to incentivize by the competition.**
> For now I can say that we will patch this scenario either by making the
> perturbation_type not available in private val/final test evaluation, or disallowing
> it explicitly by the rules (checked against the technical reports and open-sourced
> implementation, both required for eligibility for prizes).

→ 9/14 업스트림이 `permutation_type`을 cases.jsonl에서 제거한 것이 이 "patch".
→ 상금 자격은 기술 리포트 + 공개 구현으로 심사됨. 런타임 라벨 재현은 걸릴 수 있음.

## 학습 데이터 규칙 삭제 — 직접 만든 변형·라벨은 뭐든 써도 됨 (2026-09-05)

질문 (runfme): 직접 생성한 변형본과 그로 만든 robust 라벨을 검증·학습에 써도 되는가.

> **Štefánik:** you can use those for anything. Note that **we have removed the rule on
> the training data from the official rules** listed on the website, hoping to
> incentivize whitebox and light-weight methods with some constraints on the evaluation
> time. Just be careful not to overfit the existing validation sets as the final scoring
> will be done on the new problems.

→ 논문 §2.2 규칙 1("같은 유형 변형에 라벨 금지")은 **폐지**.
→ `experiments/replicate_labels.py`를 **학습 데이터 생성용**으로 쓰는 건 허용.
   같은 절차를 `are_robust` 안에서 돌리는 건 위 8/3 답변대로 권장 안 함.

## 검증셋은 model_id만으로 못 맞히게 균형 잡음 (2026-08-29)

> **Štefánik:** we need the val set to be approximately balanced (i.e. **unpredictive of
> label solely based on the model_id**), we are now running 1000s of evaluations testing
> different models and finding robust=True labels.

→ 모델별 사전확률(1층)은 설계상 무력화됨. 본선 리더보드에서 상수 예측이 0.52/0.53인 것과 일치.
→ <10B 모델에서 robust=True 찾기가 어렵다고 명시. Small 트랙 라벨이 그래서 늦어짐.

## 공개 학습 데이터셋 — 아직 없음 (2026-09-14 기준)

> **PMMon (08-31):** we are currently working on a training dataset that we will hopefully
> publish soon.

> **MS_Tom (09-13):** In the csv dataset provided on GitHub, there is only 1 model. …
> the proposal mentions that both train and val samples are available on hugging face
> but in reality it is only Val samples.

> **Spiegel (09-14):** we will publish the public training dataset in the following days
> with a more diverse set of models

→ 현재 HF `aimo-interp/*` 6개는 전부 5월 자료. 본선 5모델 중 라벨 있는 건 DeepSeek 8B뿐.

## `aimo-interp/problems-public` — 공개 안 함 (2026-09-07)

> **Štefánik:** we do not plan to make this collection public as its problems overlay
> with the private validation set. It is called "public" because it includes previously
> published problems, not because it was intended to be public.

→ CLAUDE.md 미해결 항목 "401, 아직 비공개"는 **영구**. 심볼릭 체인은 안 온다.

## 제출 한도 (2026-09-09)

> **JosefKuchar:** we don't have any zip size limit, but … there is just **15 gb user
> quota for your submissions combined**. eval time limit is 1 hour

→ 27 MB짜리 trained-probe ZIP 기준 500번 여유. 모델 가중치를 ZIP에 넣기 시작하면 금방 참.

## 리더보드 정렬 버그 (2026-09-09)

> **KKLUV:** The submission 919754 (Small: 0.53) above the 0.79 submissions and above all
> submissions with a Main score. It looks like a missing Main value may be sorting ahead
> of numeric scores.
> **JosefKuchar:** we are aware of this, not really sure we'll be able to fix this

→ 웹 리더보드 순서는 믿지 말 것. `analysis/leaderboard.py`가 task_id로 갈라 읽는 이유.

## 컴퓨트 신청 양식 — AIMO 3 팀명 필수 (2026-08-23, 미답변)

> **katoro8989:** I did not participate in AIMO 3, but the application form requires an
> AIMO 3 team name. Which form should I use, and what should I enter in that field?

→ 답변 없음. Fields 신청 시 같은 문제에 부딪힐 것.
