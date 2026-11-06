"""True와 False가 각각 무엇인지 실제 행으로 보여준다.

라벨은 (모델, 문제) 쌍에 붙는다. 같은 문제라도 모델이 다르면 라벨이 다를 수 있다.
"""

import io
import json
import textwrap
import urllib.request

import pandas as pd

URL = (
    "https://huggingface.co/datasets/aimo-interp/val-sample/resolve/"
    "1ae454ec1fad9727084eda8f9f3c9ae2239b21de/data/validation-00000-of-00001.parquet"
)


def wrap(text: str, width: int = 88, indent: str = "      ") -> str:
    return textwrap.fill(text.strip(), width=width, initial_indent=indent,
                         subsequent_indent=indent)


def show(row, title: str) -> None:
    print(title)
    print(f"      모델          {row.model_id}")
    print(f"      문제 번호      {row.problem_id}")
    print(f"      원본 정답률    {row.base_accuracy}   ({row.n_base_predictions}회 시행)")
    print(f"      변형 정답률    {row.permuted_accuracy}   ({row.n_permuted_predictions}회 시행)")
    print(f"      하락폭        {row.absolute_accuracy_decay}")
    print(f"      무너진 변형본  {row.n_detrimental_permutations}개")
    print(f"      라벨          {'True (robust)' if row.model_is_robust else 'False (spurious)'}")
    print()


def main() -> None:
    frame = pd.read_parquet(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))

    print("=" * 96)
    print("규칙")
    print("=" * 96)
    print("  원본 문제를 여러 번 풀린 정답률과, 문제를 비튼 변형본을 풀린 정답률을 비교한다.")
    print("  안 떨어지면 True(robust), 떨어지면 False(spurious).")
    print()

    true_row = frame[frame.model_is_robust].iloc[0]
    show(true_row, "── True 예시 ──" + "─" * 78)
    print(wrap(true_row.original_problem))
    print()
    print("      → 7가지 방식으로 비튼 변형본 전부를 통과했다. 하락 0.")
    print()

    false_row = frame[
        (~frame.model_is_robust) & (frame.n_detrimental_permutations >= 1)
    ].sort_values("absolute_accuracy_decay", ascending=False).iloc[0]
    show(false_row, "── False 예시 ──" + "─" * 77)
    print(wrap(false_row.original_problem))
    print()

    broken = json.loads(false_row.permutations_causing_decay)[0]
    print(f"      ↓ 이렇게 비틀었더니 정답률 {broken['variant_accuracy']} "
          f"({broken['permutation_type']}, 생성: {broken['permutation_source']})")
    print()
    print(wrap(broken["question"]))
    print()

    print("=" * 96)
    print("같은 문제라도 모델이 다르면 라벨이 갈린다")
    print("=" * 96)
    shared = (
        frame.groupby("problem_id")
        .filter(lambda group: group.model_is_robust.nunique() > 1)
        .sort_values(["problem_id", "model_is_robust"])
    )
    for problem_id, group in shared.groupby("problem_id"):
        print(f"\n  문제 {problem_id}")
        for _, row in group.iterrows():
            mark = "True " if row.model_is_robust else "False"
            print(f"    {mark}  {row.model_id:<34} "
                  f"{row.base_accuracy} → {row.permuted_accuracy}")


if __name__ == "__main__":
    main()
