import io
import urllib.request

import pandas as pd

URL = (
    "https://huggingface.co/datasets/aimo-interp/val-sample/resolve/"
    "1ae454ec1fad9727084eda8f9f3c9ae2239b21de/data/validation-00000-of-00001.parquet"
)

frame = pd.read_parquet(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))

print("고유 문제 수:", frame.problem_id.nunique(), "/ 행 수:", len(frame))
print()

print("=== 모델 × 문제 교차표 (T=robust, F=spurious) ===")
table = frame.pivot_table(
    index="model_id",
    columns="problem_id",
    values="model_is_robust",
    aggfunc=lambda values: "T" if values.iloc[0] else "F",
)
print(table.fillna("·").to_string())
print()

print("=== 문제별로 몇 개 모델이 평가됐나 ===")
per_problem = frame.groupby("problem_id").agg(
    모델수=("model_id", "nunique"),
    robust=("model_is_robust", "sum"),
)
print(per_problem.sort_values("모델수", ascending=False).to_string())
print()

print("=== permutation_type 종류별 robust 비율 ===")
frame["단일유형"] = frame.permutation_type.apply(
    lambda value: value if not value.startswith("[") else "복수유형"
)
print(frame.groupby("단일유형").model_is_robust.agg(["sum", "count"]).to_string())
