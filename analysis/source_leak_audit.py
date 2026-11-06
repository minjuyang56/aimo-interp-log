"""자체 라벨 문제 출처에 **누출 방지 제외 대상**이 섞였는지 감사. GPU 0. (지휘 디스코드 10/8~10/11 (7))

주최측이 제외하라고 한 것: **AIME 2024 / 2025 / 2026** · AIMO2·AIMO3 공개 문제 · JMO ·
`problems-public` · `train-main-v2` **유사문제**.

우리 문제풀은 `genlab_v2.load_pool` 이 만든다 — `AI-MO/aimo-validation-aime` + MATH level 5,
그리고 `train-main-v2` 와 **완전 일치**하는 문제만 제거한다(`problem not in v2`).
이 감사가 묻는 것 둘:
  1. `aimo-validation-aime` 의 연도 분포와, **우리가 실제로 라벨을 만든 문제 중 AIME 2024 가 몇 개인가**
  2. `train-main-v2` 와의 '유사문제' 는 제거되지 않았다 → 근사 중복(토큰 자카드)으로 얼마나 가까운지

사용:  .venv/Scripts/python.exe analysis/source_leak_audit.py
"""

import argparse
import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AIME_URL = ("https://huggingface.co/datasets/AI-MO/aimo-validation-aime/"
            "resolve/main/data/train-00000-of-00001.parquet")
PROBE = ROOT / "experiments" / "results" / "probe"
RUNPOD = ROOT / "experiments" / "results" / "runpod" / "labelgen32k"
WORD = re.compile(r"[A-Za-z]+|\d+")

# 라벨을 만든 묶음 전부 (별칭, screen/labels json 경로)
SETS = [
    ("Skywork off40 k=8 (#77 · sub2048 학습)", RUNPOD / "pj7vh1qllei51i/out/Skywork-OR1-Math-7B_off40_v2_labels.json"),
    ("Olmo off40 k=8 (#77 · sub2048 학습)", PROBE / "Olmo-3-7B-Think_own_k8.npz"),
    ("DeepSeek off40 k=8 새 라벨 (#DS-RELABEL)", RUNPOD / "mbpc29h3v7um64/out/DeepSeek-R1-0528-Qwen3-8B_off40_v2_labels.json"),
    ("DeepSeek off80 k=8 새 라벨 (#DS-RELABEL)", RUNPOD / "9svqsw8dzrtjsv/out/DeepSeek-R1-0528-Qwen3-8B_off80_v2_labels.json"),
    ("Olmo off40 20문제 T=1.0 (#TEMP-SENS)", RUNPOD / "i6hxd8z437eas4/out/Olmo-3-7B-Think_off40_v2_screen.json"),
]


def norm(t):
    return " ".join(WORD.findall(str(t).lower()))


def shingles(t, k=8):
    w = norm(t).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def rows_of(p):
    """(pidx, problem) 목록. npz 와 json 둘 다 받는다."""
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        return [(int(q), str(x)) for q, x in zip(d["pidx"], d["problems"])]
    if not p.is_file():
        return []
    r = json.loads(p.read_text(encoding="utf-8"))
    return [(int(x["pidx"]), x["problem"]) for x in r
            if (p.name.endswith("_screen.json") or isinstance(x.get("label"), bool))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/results/source_leak_audit.txt"))
    a = ap.parse_args()
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    import pandas as pd
    b = urllib.request.urlopen(urllib.request.Request(AIME_URL, headers={"User-Agent": "Mozilla/5.0"}),
                               timeout=180).read()
    aime = pd.read_parquet(io.BytesIO(b))
    aime["year"] = aime["url"].astype(str).str.extract(r"(19[89]\d|20[0-2]\d)")[0]
    say("자체 라벨 문제 출처 감사 — 누출 방지 제외 대상이 섞였나 (지휘 디스코드 (7))")
    say("")
    say(f"## `AI-MO/aimo-validation-aime`: {len(aime)}문제 · 연도 "
        f"{dict(sorted(aime['year'].value_counts().items()))}")
    say(f"   → **AIME 2024 가 {int((aime['year'] == '2024').sum())}문제(데이터셋의 "
        f"{(aime['year'] == '2024').mean():.0%})** 들어 있다. 2025·2026 은 없다.")
    say("")
    by_norm = {norm(p): y for p, y in zip(aime["problem"], aime["year"])}

    v2p = ROOT / "experiments/results/train-main-v2.parquet"
    v2 = []
    if v2p.is_file():
        v2 = [str(x) for x in pd.read_parquet(v2p)["problem"].unique()]
    v2sh = [shingles(x) for x in v2]
    say(f"## `train-main-v2` 고유 문제 {len(v2)}개 (완전 일치 제거는 코드에 있다: `problem not in v2`)")
    say("")

    say(f"{'묶음':42s} {'문제':>4s} {'AIME':>5s} {'2024':>5s} {'2022':>5s} {'2023':>5s} {'v2 최대자카드':>12s}")
    tot24 = []
    for name, p in SETS:
        rs = rows_of(p)
        if not rs:
            say(f"{name:42s} (파일 없음: {p.name})")
            continue
        yrs = [by_norm.get(norm(t)) for _, t in rs]
        n_aime = sum(1 for y in yrs if y)
        c = {y: yrs.count(y) for y in ("2022", "2023", "2024")}
        mx = 0.0
        for _, t in rs:
            s = shingles(t)
            for o in v2sh:
                if s and o:
                    j = len(s & o) / len(s | o)
                    mx = max(mx, j)
        say(f"{name:42s} {len(rs):>4d} {n_aime:>5d} {c['2024']:>5d} {c['2022']:>5d} {c['2023']:>5d} "
            f"{mx:>12.3f}")
        if c["2024"]:
            tot24.append((name, c["2024"], len(rs),
                          [q for (q, t), y in zip(rs, yrs) if y == "2024"]))
    say("")
    if tot24:
        say("## **제외 대상 포함 — AIME 2024 문제가 들어 있다**")
        for name, k, n, pidx in tot24:
            say(f"   {name}: {k}/{n} 문제 · pidx {pidx}")
    else:
        say("## AIME 2024 포함 0건")
    say("")
    say("## 읽는 법")
    say("   `v2 최대자카드` = train-main-v2 문제와의 8-그램 자카드 최댓값. 1.0 에 가까우면 사실상")
    say("   같은 문제다. 코드가 **완전 일치만** 제거하므로 '유사문제' 는 이 값으로 봐야 한다.")
    say("   AIME 2024 는 주최측이 제외하라고 한 대상이고, 우리 풀은 그 연도를 걸러내지 않는다")
    say("   (`load_pool` 이 연도를 보지 않는다). 몇 건인지가 위 표의 `2024` 열이다.")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
