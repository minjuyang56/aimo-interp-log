"""모델별 robust 사전확률 — 원시 비율 vs 평활 추정.

원시 비율(sum/count)은 표본이 1~7건뿐이라 0.00 / 1.00 같은 극단값이 나온다.
전체 평균을 사전분포로 둔 베타-이항 평활을 적용해 비교한다.
"""

import io
import urllib.request

import pandas as pd

URL = (
    "https://huggingface.co/datasets/aimo-interp/val-sample/resolve/"
    "1ae454ec1fad9727084eda8f9f3c9ae2239b21de/data/validation-00000-of-00001.parquet"
)

PRIOR_STRENGTH = 10.0  # 가상 관측 수. 클수록 전체 평균 쪽으로 끌어당긴다.


def main() -> None:
    frame = pd.read_parquet(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))

    overall = frame.model_is_robust.mean()
    table = frame.groupby("model_id").model_is_robust.agg(["sum", "count"])
    table["원시비율"] = table["sum"] / table["count"]
    table["평활추정"] = (table["sum"] + overall * PRIOR_STRENGTH) / (
        table["count"] + PRIOR_STRENGTH
    )
    table["원시판정"] = table["원시비율"] > 0.5
    table["평활판정"] = table["평활추정"] > 0.5

    pd.set_option("display.width", 160)
    print(f"전체 robust 비율 (사전분포 중심): {overall:.3f}  =  "
          f"{int(frame.model_is_robust.sum())} / {len(frame)}")
    print(f"사전분포 강도: {PRIOR_STRENGTH} 가상 관측")
    print()
    print(table.round(3).to_string())
    print()

    for name, column in [("원시비율", "원시판정"), ("평활추정", "평활판정")]:
        predicted = frame.model_id.map(table[column])
        correct = int((predicted == frame.model_is_robust).sum())
        print(f"{name} 기준 인샘플 정확도: {correct}/{len(frame)} = "
              f"{correct / len(frame):.3f}")


if __name__ == "__main__":
    main()
