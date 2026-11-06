"""센 섭동을 넣어도 과반 규칙이 켜질 수 있나 — 유형당 붕괴율로 역산한다. (지휘 #DOMAIN-FIX)

왜 이걸 재는가. '유형 수를 3 -> 6 으로 늘리면 과반이 성립한다' 는 기대는 **임계도 같이 올라간다**는
사실을 빼먹는다. 과반은 '유형의 절반 초과' 이므로 3유형은 2개, 6유형은 4개가 깨져야 한다.
그래서 결정적인 양은 유형 수가 아니라 **유형당 붕괴 확률 p** 다. p 를 실측하고,
주최측의 spurious 27% 를 내려면 p 가 얼마여야 하는지 이항분포로 역산한다.
"""
import glob
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rows(pat):
    """라벨 파일 경로는 **명시**한다 — glob 패턴을 쓰다 실제 파일명과 안 맞아
    '없음' 으로 조용히 넘어간 적이 있다(파일럿2 평가기 버그와 같은 실수)."""
    out = []
    for f in sorted(glob.glob(str(ROOT / pat), recursive=True)):
        for r in json.loads(Path(f).read_text(encoding="utf-8")):
            if r.get("label") is None and r.get("label_major") is None:
                continue
            if r.get("n_types"):
                out.append(r)
    return out


def pois(sets):
    print(f"{'라벨셋':<28}{'k':>3}{'문제':>6}{'유형수':>7}{'깨진유형':>9}{'유형당 p':>10}")
    allb = allt = 0
    for tag, pat in sets:
        rs = rows(pat)
        if not rs:
            print(f"{tag:<28}{'-':>3}{'없음':>6}")
            continue
        nt = sum(r["n_types"] for r in rs)
        nb = sum(r["n_broken_types"] for r in rs)
        k = rs[0].get("gen_k", "?")
        print(f"{tag:<28}{k:>3}{len(rs):>6}{nt:>7}{nb:>9}{nb / max(1, nt):>9.1%}")
        if k == 8:
            allb += nb
            allt += nt
    return allb / max(1, allt)


p = pois([
    ("Skywork 0-99 (k4)", "experiments/results/genlab/skywork_labels_merged.json"),
    ("Skywork 40-79 (k8)", "experiments/results/runpod/labelgen32k/*/out/Skywork*off40*labels*.json"),
    ("Olmo 40-79 (k8)", "experiments/results/runpod/labelgen32k/ofclt769pgvn6u/out/Olmo*labels*.json"),
    ("DeepSeek 40-79 (k8)",
     "experiments/results/runpod/labelgen32k/luvw8yu3mkxsza/out/DeepSeek*v3_labels*.json"),
    ("DeepSeek 80-119 (k4)", "experiments/results/runpod/labelgen32k/*/out/DeepSeek*off80*labels*.json"),
])
print(f"\nk=8 셋 합산 유형당 붕괴율 p = {p:.1%}")

print("\n유형 수 T, 유형당 붕괴율 p 일 때 과반(> T/2 개 붕괴) 확률 = spurious 비율:")
print(f"{'p':>7} " + " ".join(f"{'T=' + str(t):>8}" for t in (3, 4, 6, 8)))
for pp in (p, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
    cells = []
    for T in (3, 4, 6, 8):
        need = T // 2 + 1
        q = sum(comb(T, i) * pp ** i * (1 - pp) ** (T - i) for i in range(need, T + 1))
        cells.append(f"{q:>7.1%}")
    mark = "  <- 실측 p" if abs(pp - p) < 1e-9 else ""
    print(f"{pp:>6.0%}  " + " ".join(cells) + mark)

print("\n읽는 법: 주최측은 과반 규칙으로 spurious **27%** 를 얻는다(train-main-v2 19/71).")
print("실측 p 로는 유형을 6개, 8개로 늘려도 그 근처에 못 간다 — 임계가 같이 올라가기 때문이다.")
print("27% 를 내려면 유형당 붕괴율이 40~50% 대여야 한다. 즉 부족한 것은 **유형 수가 아니라 섭동의 세기**다.")

# ---- 독립 가정은 참이 아니다. 관측 분포와 이항 예측을 맞대 본다.
#      문제마다 '잘 깨지는 문제' 가 있으면 붕괴가 문제 안에서 상관되고, 그러면 과반이
#      독립 예측보다 **자주** 켜진다. 그 방향과 크기를 숫자로 적어 둔다.
print("\n관측 vs 독립 이항 예측 (깨진 유형 수 분포 · 과반 발생 수):")
print(f"{'라벨셋':<28}{'관측 분포':>24}{'관측 과반':>10}{'독립 예측':>10}")
for tag, pat in (("Skywork 40-79 (k8)",
                  "experiments/results/runpod/labelgen32k/*/out/Skywork*off40*labels*.json"),
                 ("Olmo 40-79 (k8)",
                  "experiments/results/runpod/labelgen32k/ofclt769pgvn6u/out/Olmo*labels*.json"),
                 ("DeepSeek 40-79 (k8)",
                  "experiments/results/runpod/labelgen32k/luvw8yu3mkxsza/out/DeepSeek*v3_labels*.json")):
    rs = rows(pat)
    dist = {}
    for r in rs:
        dist[r["n_broken_types"]] = dist.get(r["n_broken_types"], 0) + 1
    nt = sum(r["n_types"] for r in rs)
    nb = sum(r["n_broken_types"] for r in rs)
    pp = nb / max(1, nt)
    obs = sum(1 for r in rs if r["n_broken_types"] > r["n_types"] / 2)
    exp = sum(sum(comb(r["n_types"], i) * pp ** i * (1 - pp) ** (r["n_types"] - i)
                  for i in range(r["n_types"] // 2 + 1, r["n_types"] + 1)) for r in rs)
    print(f"{tag:<28}{str(dict(sorted(dist.items()))):>24}{obs:>10}{exp:>10.1f}")
print("→ Olmo 는 관측 과반이 독립 예측보다 많다 = 붕괴가 **문제 안에서 상관**된다"
      "(잘 깨지는 문제가 따로 있다).")
print("  그래도 10% 수준이라 주최측 27% 와는 멀다. 상관을 감안해도 결론은 바뀌지 않는다.")
