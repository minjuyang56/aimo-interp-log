# 데이터 컬럼 15개, 하나씩

원본 데이터셋 [`aimo-interp/val-sample`](https://huggingface.co/datasets/aimo-interp/val-sample)에는
컬럼이 **15개**입니다. 그런데 `scripts/import_hf_dataset.py`가 만드는 `cases.jsonl`에는
그중 일부만 남습니다. 그래서 원본을 직접 받아 전수 조사했습니다.

재현: `uv run python analysis/columns.py`

---

## 먼저, 가장 중요한 구별

```
실전에서 받는 것   2개  ←  model_id, original_problem
나머지            13개  ←  정답을 만드는 과정의 기록
```

채점기가 넘겨주는 건 딱 두 개입니다:

```python
# components/ingestion_program/ingestion.py:158-160
problems = [
    case["problem"]["original_problem"] for _, case in batch
]
# 그리고 are_robust(model_id, problems) — 여기 model_id 하나 더.
```

**나머지 13개는 입력으로 쓸 수 없습니다.** 실전에 존재하지 않으니까요.

**하지만 학습 타깃으로는 쓸 수 있습니다.** 이 구별이 전부입니다.
입력에 넣으면 누출이고, 타깃에 넣으면 정당한 학습입니다.

특히 `absolute_accuracy_decay` — 이진 `True`/`False`보다 연속값이 정보가 훨씬 많습니다.
저장소의 `uncertainty-profiling` 예제가 분류기가 아니라 **회귀모델**을 쓰는 이유가 이겁니다.

---

## 묶음 1 — 식별자 (누구/무엇인가)

| 컬럼 | 내용 | 이 표본에서 |
|---|---|---|
| `model_id` | 판정 대상 모델 | **7종** · 실전에서 받는 유일한 메타 정보 |
| `dataset_id` | 문제의 출처 문제집 | **전부 `aimo-2025-reference`** — 이 표본에선 정보가 0 |
| `problem_id` | 문제 고유 번호 | **8개뿐** — 28행이 8문제를 공유 |
| `original_problem` | 원본 문제 텍스트 | 실전에서 받는 유일한 내용 |

`problem_id`가 8개뿐이라는 게 중요합니다. 28행처럼 보이지만 **독립 문제는 8개**이고,
모델마다 받은 문제가 다릅니다. 그래서 "모델이 강해서"인지 "쉬운 문제를 받아서"인지
분리가 안 됩니다. → `analysis/model_confound.py`

---

## 묶음 2 — 어떻게 비틀었나

| 컬럼 | 내용 |
|---|---|
| `permutation_type` | 어떤 방식으로 흔들었는지 — `rename`, `domain`, `distract`, `rephrase`, `typos`, `expert_perturbations`, `expert_no_solution` |
| `permutation_source` | 그 변형본을 누가 만들었는지 |

### ⚠️ `permutation_type`은 라벨 누출입니다

| permutation_type | robust | 전체 |
|---|---|---|
| 단일 유형 하나만 | 0 | 19 |
| 7~12개 유형이 나열됨 | 9 | 9 |

**28/28 완벽 분리.** 문제의 속성이 아니라 **라벨을 만든 절차가 남긴 지문**입니다 —
robust 판정을 받으려면 모든 유형을 통과해야 하니 전부 나열되고,
spurious는 처음 무너뜨린 유형 하나만 적히고 거기서 끝나거든요.

채점기가 이 필드를 안 넘겨주므로 실전에선 못 씁니다. 다만 `cases.jsonl`에는 들어 있으니
**로컬 특징 추출 코드가 실수로 집어넣지 않도록** 주의해야 합니다.

### `permutation_source` 실제 값

| 값 | 횟수 | 정체 |
|---|---|---|
| `gpt-5.2-2025-12-11:low` | 53 | 변형본을 생성한 모델 |
| `gemini-3.1-pro-preview:low` | 16 | **변형본을 생성한 또 다른 모델** |
| `expert_no_solution` | 19 | 사람이 만듦 (유형 이름이 그대로 들어감) |
| `expert_perturbations` | 9 | 사람이 만듦 |

두 가지 짚을 점:

1. 자동 생성 모델이 **하나가 아니라 둘**입니다. gpt-5.2와 gemini-3.1-pro 양쪽이 변형본을 만듭니다.
2. 사람이 만든 경우엔 사람 이름 대신 **유형 이름이 그대로** 들어갑니다. 즉 이 필드는
   "모델이면 모델 ID, 사람이면 유형 이름"이라는 다소 어색한 스키마입니다.

---

## 묶음 3 — 실제 성적 (라벨의 뿌리)

| 컬럼 | 내용 | 예시 |
|---|---|---|
| `base_accuracy` | 원본 문제 정답률 | 10번 시켜 6번 맞으면 `0.6` |
| `permuted_accuracy` | 변형본 전체 정답률 | 비튼 문제를 시켰을 때 |
| `absolute_accuracy_decay` | `base − permuted` | `0.6 → 0.0`이면 `0.6` |
| `relative_accuracy_decay` | 같은 걸 비율로 | `0.6 → 0.3`이면 절대 `0.3`, 상대 `50%` |
| `n_base_predictions` | 원본을 몇 번 시켰나 | **항상 10** |
| `n_permuted_predictions` | 변형본을 총 몇 번 시켰나 | **10 ~ 120** ← 여기가 비대칭 |

### `absolute_accuracy_decay`가 진짜 타깃입니다

이 표본의 분포:

```
robust  = True   →  decay 정확히 0.0    (9건 전부, 예외 없음)
robust  = False  →  decay 0.5 ~ 1.0     (19건 전부)
```

**0.0과 0.5 사이가 텅 비어 있습니다.** 이건 우연이 아니라 설계입니다.

라벨 규칙은 논문 §1.4에 있습니다: 변형본 중 **과반이 무너지면 False**, 아니면 True.
그리고 애매한 케이스("not clear-cut")는 평가셋에서 **의도적으로 뺐다**고 명시돼 있습니다.
675행 학습 데이터에서도 같습니다 — `OVERALL` 행 141개 전부 `robust == (decay == 0.0)`이고
False 쪽 최소 decay가 0.5입니다 (`analysis/label_rule.py`).

그러니 **임계값을 찾을 필요가 없습니다.** 중간값이 존재하지 않게 만들어진 데이터고,
본선도 같은 절차로 만들어지니 마찬가지일 겁니다.

### ⚠️ 검증 예산이 비대칭입니다

| 라벨 | 원본 실행 | 변형 실행 |
|---|---|---|
| spurious (19건) | 10 | **정확히 10** |
| robust (9건) | 10 | **67 ~ 120** |

robust 선언은 모든 유형을 통과해야 하고, spurious는 처음 무너진 지점에서 멈춥니다.
조기 종료 설계죠. 합리적이지만 결과적으로:

> **`True` 라벨이 `False` 라벨보다 10배 이상 비싸게 검증됐습니다.**
> `True`는 거의 확실하고, `False`는 "적어도 한 군데선 무너졌다"는 약한 진술에 가깝습니다.

학습할 때 두 클래스를 같은 무게로 다루면 안 됩니다.
`n_permuted_predictions`를 샘플 가중치로 쓰는 게 자연스럽습니다.
→ `analysis/probe_counts.py`

---

## 묶음 4 — 종합 결과

| 컬럼 | 내용 |
|---|---|
| `n_detrimental_permutations` | 몇 개의 변형본에서 무너졌나. robust면 `0` |
| `permutations_causing_decay` | **무너뜨린 변형본들의 전문 + 각각의 정답률** |
| `model_is_robust` | 최종 이진 라벨 |

### `permutations_causing_decay`가 금광입니다

이름이 헷갈리는데, 뜻은 이렇습니다:

```
permutations        변형본들
causing             ~를 일으킨
decay               (정확도) 하락
```

19건 전부에서 검증했습니다 (`analysis/probe_field.py`):

- 리스트 길이 == `n_detrimental_permutations` (19/19)
- 모든 항목의 `variant_accuracy` < `base_accuracy` (19/19)

즉 **원본보다 정답률이 떨어진 변형본만 골라 담은 리스트**입니다. 항목 하나의 구조:

```json
{
  "question": "변형된 문제 전문",
  "variant_accuracy": 0.0,
  "n_predictions": 1,
  "permutation_type": "rephrase",
  "permutation_source": "gpt-5.2-2025-12-11:low"
}
```

**원본과 무너진 변형본을 나란히 볼 수 있는 유일한 필드**입니다.
"이 모델이 정확히 어떤 종류의 변화에 무너지는가"를 문장 단위로 관찰할 수 있습니다.
→ `analysis/show_variants.py`

### 이름에 대한 사족

`permutation`은 원래 **순열 · 재배열**입니다. 여기서 하는 일은 **perturbation(섭동)**이고요.
저장소도 두 단어를 섞어 씁니다 — README는 "perturbation categories"라 부르는데
컬럼은 `permutation_type`이고, 그 컬럼의 **값 하나가 `expert_perturbations`**입니다.
한 줄 안에서 어긋납니다. 초기엔 `rename`, `domain`처럼 표면 요소를 갈아끼우는 변형만
있었다가 이름이 굳은 것으로 보입니다.

---

## 한 장 요약

```
                      실전 입력   학습 타깃   비고
model_id                 ✅         ·        약한 사전확률로만 (표본 1~7건)
original_problem         ✅         ·        유일한 내용 정보
─────────────────────────────────────────────────────────
dataset_id               ❌         ·        이 표본에선 값이 하나뿐
problem_id               ❌         ·        8개뿐 — 교락 원인
permutation_type         ❌         ❌       28/28 누출. 타깃으로도 부적절
permutation_source       ❌         ·        생성 모델 2종 + 사람
base_accuracy            ❌         ✅       난이도 보정용
permuted_accuracy        ❌         ✅
absolute_accuracy_decay  ❌         ✅✅     회귀 타깃으로 최우선
relative_accuracy_decay  ❌         ✅
n_base_predictions       ❌         ·        항상 10
n_permuted_predictions   ❌         ✅       샘플 가중치로 사용
n_detrimental_perms      ❌         ✅
permutations_causing_..  ❌         ✅✅     원본↔변형본 쌍. 특징 설계의 재료
model_is_robust          ❌         ✅       최종 라벨
```

`permutation_type`만 타깃으로도 ❌인 이유: 라벨과 동어반복이라
그걸 예측하도록 학습하면 아무것도 배우지 못합니다.
