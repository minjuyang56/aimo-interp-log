import io
import json
import urllib.request
from collections import Counter

import pandas as pd

URL = (
    "https://huggingface.co/datasets/aimo-interp/val-sample/resolve/"
    "1ae454ec1fad9727084eda8f9f3c9ae2239b21de/data/validation-00000-of-00001.parquet"
)

frame = pd.read_parquet(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))

keys = Counter()
len_matches_count = 0
all_below_base = 0
rows_with_entries = 0

for _, row in frame.iterrows():
    entries = json.loads(row.permutations_causing_decay)
    if not entries:
        continue
    rows_with_entries += 1
    for entry in entries:
        keys.update(entry.keys())
    if len(entries) == row.n_detrimental_permutations:
        len_matches_count += 1
    if all(entry["variant_accuracy"] < row.base_accuracy for entry in entries):
        all_below_base += 1

print("항목이 있는 행:", rows_with_entries)
print("리스트 길이 == n_detrimental_permutations:", len_matches_count)
print("모든 항목이 base_accuracy 미만:", all_below_base)
print()
print("항목의 키:", dict(keys))
print()

row = frame.iloc[9]
entries = json.loads(row.permutations_causing_decay)
print("샘플 행 9 —", row.model_id)
print("base", row.base_accuracy, "permuted", row.permuted_accuracy,
      "n_detrimental", row.n_detrimental_permutations, "n_permuted_pred", row.n_permuted_predictions)
print("항목 수:", len(entries))
print()
for entry in entries:
    print("  {:22s} src={:28s} acc={} n={}".format(
        entry["permutation_type"],
        entry["permutation_source"][:26],
        entry["variant_accuracy"],
        entry["n_predictions"],
    ))
