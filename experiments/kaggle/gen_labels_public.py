"""공개 수학문제 자체 라벨 생성 파이프라인 — **준비만, 실행 금지** (지휘 #26-3, 주최측 답변 대기).

목적: 33문제 검정력 병목을 깨기 위해 공개 문제(MATH · AIME) 수백 개에 주최측과 같은 규칙으로 라벨을 붙인다.
      모델 4종 각각에 대해 → 모델 간 라벨 차이(상호작용)도 생긴다.
규칙 적합성: 사용 여부는 주최측 답변 후 결정. 최종 검증셋과 같은 변형 유형 우려가 있어 변형은 rename/typos/distract/reorder(정답 불변)만.

절차 (Kaggle, 모델당 커널 하나, gen_labels.py 를 인라인)
  1. 문제: MATH(HF EleutherAI/hendrycks_math, level 3–5, 정답이 정수인 것) N_MATH개 + AIME(AI-MO/aimo-validation-aime) 전부 → 셔플, seed 0
  2. 원문제 k=8회(temperature 0.6) → 의사정답 = 최빈 답. base_acc < 0.5 인 문제는 제외(주최측: 원본을 맞히는 게 전제)
  3. 변형 4종 × k=8회 → acc_v → my_max_drop = base_acc − min_v acc_v
  4. 라벨: robust ⟺ my_max_drop ≤ 0.10, spurious ⟺ ≥ 0.25, 사이는 None (train-main-v2 규칙)
  5. 출력: /kaggle/working/gen_public/<모델별칭>/generations.jsonl · summary.csv (문제, gold, base_acc, acc_v…, my_max_drop, label)
비용 추정: 문제 300 × (1 + 4) 텍스트 × 8샘플 = 12,000 생성 × ~2048토큰. T4×2 fp16 배치 16에서 7B ≈ 25 tok/s/시퀀스 → 약 17시간/모델 → 커널 12시간 한도 초과.
  → 1차는 문제 120개·k=6 (≈ 4시간/모델) 로 잡고, 정답 정수 문제만. 실행 결정 후 이 상수만 바꾼다.

Kaggle 헤더 (kaggle_run.py 방식으로 인라인):
  pip install transformers==5.13.0 …; sys.argv = ["gen_labels.py", "--source", "public", "--n-math", "100", "--n-aime", "30", "--k", "6", "--max-new-tokens", "2048", "--batch", "16", "--model", "<model_id>"]
gen_labels.py 에 필요한 확장(아직 안 함, 실행 결정 후): --source public (아래 load_public), --model 인자(MODEL 상수 대체), summary 에 label 컬럼.
"""

import io
import json
import re
import urllib.request

import numpy as np

MATH_URL = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main/{subject}/test-00000-of-00001.parquet"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "intermediate_algebra", "number_theory", "prealgebra", "precalculus", "geometry"]
AIME_URL = "https://huggingface.co/datasets/AI-MO/aimo-validation-aime/resolve/main/data/train-00000-of-00001.parquet"
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=180).read()


def load_public(n_math=100, n_aime=30, seed=0, levels=("Level 3", "Level 4", "Level 5")):
    """정답이 정수인 MATH(level 3–5) n_math개 + AIME n_aime개. 반환: [{"pidx", "problem", "gold", "source"}]"""
    import pandas as pd
    rng = np.random.default_rng(seed); rows = []
    for subj in MATH_SUBJECTS:
        df = pd.read_parquet(io.BytesIO(_get(MATH_URL.format(subject=subj))))
        for r in df.itertuples():
            if r.level not in levels:
                continue
            m = BOXED.search(r.solution)
            if not m or not re.fullmatch(r"-?\d+", m.group(1).strip()):
                continue
            rows.append({"problem": r.problem.strip(), "gold": int(m.group(1)), "source": f"MATH/{subj}/{r.level}"})
    rng.shuffle(rows); math_rows = rows[:n_math]
    df = pd.read_parquet(io.BytesIO(_get(AIME_URL)))
    aime_rows = [{"problem": r.problem.strip(), "gold": int(r.answer), "source": f"AIME/{r.url.split('/')[-1] if hasattr(r, 'url') else ''}"} for r in df.itertuples()]
    rng.shuffle(aime_rows); aime_rows = aime_rows[:n_aime]
    out = math_rows + aime_rows
    for i, r in enumerate(out):
        r["pidx"] = i
    return out


def label_from_drop(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    return True if d <= 0.10 else (False if d >= 0.25 else None)


if __name__ == "__main__":
    # 준비 확인용: 문제만 세어 본다 (모델 실행 없음)
    rows = load_public(20, 5)
    print(len(rows), "문제 ·", sum(r["source"].startswith("MATH") for r in rows), "MATH ·", sum(r["source"].startswith("AIME") for r in rows), "AIME")
    print(json.dumps(rows[0], ensure_ascii=False)[:300])
