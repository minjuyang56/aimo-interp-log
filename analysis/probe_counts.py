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

print("{:>3} {:>6} {:>6} {:>7} {:>6} {:>6}  {}".format(
    "행", "n_base", "n_perm", "detrim", "항목", "고유문", "항목별 n_predictions"))
for index, row in frame.iterrows():
    entries = json.loads(row.permutations_causing_decay)
    if not entries:
        continue
    unique_questions = len({entry["question"] for entry in entries})
    npred = Counter(entry["n_predictions"] for entry in entries)
    print("{:>3} {:>6} {:>6} {:>7} {:>6} {:>6}  {}".format(
        index,
        row.n_base_predictions,
        row.n_permuted_predictions,
        row.n_detrimental_permutations,
        len(entries),
        unique_questions,
        dict(npred),
    ))
