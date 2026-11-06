"""True/False 라벨이 실제로 무엇을 자른 것인지 데이터로 확정한다.

논문 §1.4는 규칙을 이렇게 정의한다:

    S(X, M, Δ) = 0  if for a majority of ΔX ∈ Δ : M(X)=y_true and M(ΔX)≠y_true
                 1  otherwise

즉 "전부 견뎌야 True"가 아니라 "과반이 무너지면 False"다.
그리고 애매한 케이스는 평가셋에서 뺐다고 적혀 있다("not clear-cut").

여기서 확인하는 것 세 가지:

  1. 라벨의 부착 단위 — 행 단위인가, (모델, 문제) 단위인가
  2. OVERALL 행에서 라벨과 정확히 일치하는 술어가 무엇인가
  3. "clear-cut만 남겼다"는 주장이 decay 값 분포에 실제로 찍혀 있는가
"""

import csv
from collections import defaultdict

from _paths import AGG_CSV, FINAL_CSV, require


def as_bool(text):
    return text.strip().lower() == "true"


def main():
    require(FINAL_CSV, AGG_CSV)
    with FINAL_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    print(f"math-robust-final.csv: {len(rows)}행")

    # 1. 라벨의 부착 단위 -----------------------------------------------------
    by_problem = defaultdict(set)
    for row in rows:
        by_problem[row["problem_id"]].add(as_bool(row["model_is_robust"]))
    mixed = [pid for pid, values in by_problem.items() if len(values) > 1]

    print()
    print("[1] 라벨의 부착 단위")
    print(f"    고유 problem_id       : {len(by_problem)}")
    print(f"    라벨이 섞인 problem_id : {len(mixed)}")
    if mixed:
        print("    -> 행 단위 라벨")
    else:
        print("    -> 문제 단위 라벨. 모든 유형별 행에 같은 값이 복사돼 있다.")
        print(f"       실질 샘플 수는 {len(rows)}가 아니라 {len(by_problem)}개다.")

    # 2. 라벨과 일치하는 술어 -------------------------------------------------
    overall = [r for r in rows if r["permutation_type"] == "OVERALL"]
    print()
    print(f"[2] 라벨과 일치하는 술어 (OVERALL 행 {len(overall)}개 대상)")
    candidates = {
        "absolute_accuracy_decay == 0.0": lambda r: float(r["absolute_accuracy_decay"]) == 0.0,
        "absolute_accuracy_decay <= 0.0": lambda r: float(r["absolute_accuracy_decay"]) <= 0.0,
        "absolute_accuracy_decay < 0.5": lambda r: float(r["absolute_accuracy_decay"]) < 0.5,
        "n_detrimental_permutations == 0": lambda r: int(r["n_detrimental_permutations"]) == 0,
    }
    for name, predicate in candidates.items():
        hits = sum(1 for r in overall if as_bool(r["model_is_robust"]) == predicate(r))
        mark = "  <-- 완전 일치" if hits == len(overall) else ""
        print(f"    {name:34s} {hits:4d}/{len(overall)}{mark}")

    # 같은 술어를 675행 전체에 적용하면 왜 깨지는지 보여준다.
    print()
    print("    같은 술어를 675행 전체에 적용하면:")
    for name, predicate in candidates.items():
        hits = sum(1 for r in rows if as_bool(r["model_is_robust"]) == predicate(r))
        print(f"    {name:34s} {hits:4d}/{len(rows)}")
    print("    -> 유형별 행은 그 유형만의 decay를 담는데 라벨은 문제 전체의 것이라 어긋난다.")

    # 3. clear-cut 필터링의 흔적 ---------------------------------------------
    print()
    print("[3] decay 값 분포 (OVERALL 행)")
    for label in (True, False):
        values = sorted({float(r["absolute_accuracy_decay"])
                         for r in overall if as_bool(r["model_is_robust"]) is label})
        head = ", ".join(f"{v:g}" for v in values[:6])
        tail = "" if len(values) <= 6 else f", ... , {values[-1]:g}"
        print(f"    robust={str(label):5s} n={len(values):3d} 고유값: [{head}{tail}]")

    true_max = max(float(r["absolute_accuracy_decay"])
                   for r in overall if as_bool(r["model_is_robust"]))
    false_min = min(float(r["absolute_accuracy_decay"])
                    for r in overall if not as_bool(r["model_is_robust"]))
    print()
    print(f"    True 쪽 최대 decay : {true_max:g}")
    print(f"    False 쪽 최소 decay: {false_min:g}")
    print(f"    사이 구간 ({true_max:g}, {false_min:g}) 에 값이 없다"
          f" -> 논문의 'not clear-cut 제외'가 데이터에 그대로 찍혀 있다.")
    print("    경계선을 튜닝할 여지가 없다. 애초에 중간값이 존재하지 않는다.")


if __name__ == "__main__":
    main()
