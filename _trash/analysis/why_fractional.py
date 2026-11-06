"""정답률이 왜 0/1이 아니라 0.4 같은 소수인지 확인한다.

같은 문제를 여러 번 풀린다. accuracy = 맞힌 횟수 / 시행 횟수.
accuracy × n_predictions 이 정수로 떨어지는지 검산해 보면 바로 드러난다.
"""

import io
import urllib.request

import pandas as pd

URL = (
    "https://huggingface.co/datasets/aimo-interp/val-sample/resolve/"
    "1ae454ec1fad9727084eda8f9f3c9ae2239b21de/data/validation-00000-of-00001.parquet"
)


def main() -> None:
    frame = pd.read_parquet(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))

    print("원본 문제 쪽 — accuracy × 시행횟수 = 맞힌 횟수")
    print()
    print("{:>10} {:>8} {:>10}   {}".format("accuracy", "시행", "맞힌 수", "해석"))
    combos = (
        frame.groupby(["base_accuracy", "n_base_predictions"])
        .size()
        .reset_index(name="rows")
        .sort_values("base_accuracy")
    )
    for _, row in combos.iterrows():
        hits = row.base_accuracy * row.n_base_predictions
        print("{:>10} {:>8} {:>10.0f}   {:.0f}번 중 {:.0f}번 정답   ({}개 행)".format(
            row.base_accuracy, row.n_base_predictions, hits,
            row.n_base_predictions, hits, row.rows))

    print()
    print("변형 문제 쪽 — 시행 횟수가 훨씬 많다")
    print()
    print("{:>10} {:>8} {:>10}".format("accuracy", "시행", "맞힌 수"))
    combos = (
        frame.groupby(["permuted_accuracy", "n_permuted_predictions"])
        .size()
        .reset_index(name="rows")
        .sort_values(["n_permuted_predictions", "permuted_accuracy"])
    )
    for _, row in combos.iterrows():
        hits = row.permuted_accuracy * row.n_permuted_predictions
        print("{:>10} {:>8} {:>10.0f}".format(
            row.permuted_accuracy, row.n_permuted_predictions, hits))

    print()
    print("검산: accuracy × 시행횟수가 전부 정수인가?")
    for column, count_column in [
        ("base_accuracy", "n_base_predictions"),
        ("permuted_accuracy", "n_permuted_predictions"),
    ]:
        products = frame[column] * frame[count_column]
        whole = (products - products.round()).abs() < 1e-6
        print(f"  {column:20s} {whole.sum()}/{len(frame)} 행에서 정수")


if __name__ == "__main__":
    main()
