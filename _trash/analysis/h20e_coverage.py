"""최종셋 규모(35~40문제)에서 cap 8192 이상을 사는 두 가지 방법. GPU 0. (지휘 #63)

문제: 호출당 예산 825s 에서 cap 8192 이상을 쓰면 **문제 수가 제한된다**. 격자(H14 데이터)에서 신호는
cap 8192 이상에만 있고(k1 0.731 / 0.781, k2 0.769 / 0.840), cap 4096~6144 는 임계값이 cap 과 같아
(= 전부 절단) AUC 0.600~0.662 로 사실상 정보가 없다.

이 스크립트가 재는 것
  A) **커버리지**: (k, cap) 마다 한 호출에 들어가는 **최대 문제 수**. 시간 모델 ms(B,L) + 모델별 KV 한도
     (Olmo 는 MHA 로 토큰당 512 KB 라 그룹이 쪼개져 더 적게 들어간다) 둘 다 반영.
  B) **혼합안**: m 개만 생성하고 나머지는 텍스트 규칙(n_sym)으로 채울 때의 AUC.
     선택 규칙은 **라벨을 안 보는 고정 규칙** 3종(무작위 아님):
       R1 애매한 것 먼저  — |n_sym − train 중앙값| 이 작은 m 개 (싼 신호가 흐린 데만 비싸게 재는 cascade)
       R2 의심스러운 것 먼저 — n_sym 큰 m 개 (텍스트 규칙이 spurious 라 지목한 쪽을 생성으로 확인)
       R3 입력 순서 앞 m 개 — 대조군 (채점기 순서는 임의이므로 '무작위에 가까운' 기준선)
     판정 방식: **각 군을 따로 중앙값 분할**한다. 생성군은 긴 쪽 절반을 False, 비생성군은 n_sym 큰 쪽 절반을 False.
     두 군을 한 점수로 합치지 않는다 — 토큰 수와 n_sym 은 단위가 달라 순위를 이어 붙이면 군 간 순서가 뒤섞인다
     (처음 그렇게 짰다가 같은 m 에서 0.51~0.88 로 요동치는 걸 보고 바꿨다).
     출력이 bool 벡터이므로 지표는 AUC 가 아니라 **정확도**다(채점기와 같은 기준).
  검문소: 문제 단위 permutation null 1000. **데이터는 Skywork 1모델뿐**(H14 저장분)이라 모델 간 전이는
  H20-M(지휘 #63) 결과로 따로 봐야 한다. B 는 같은 33문제를 재사용하는 **탐색**이며 사전등록 지표가 아니다.

사용:  .venv/Scripts/python.exe analysis/h20e_coverage.py [--perms 1000] [--budget 825]
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solutions" / "h20e"))
import h20e_lib as L                                     # noqa: E402

CAPS = (4096, 6144, 8192, 12288, 16384)
GPU_GB = 96.0
MODELS = [("Qwen3.5-4B", 9.3, 128.0), ("Skywork-7B", 15.2, 56.0),
          ("Olmo-3-7B-Think", 14.6, 512.0), ("DeepSeek-8B", 16.4, 144.0)]


def slots_for(weights_gb, kv_kb):
    return max(1, int(max(0.0, GPU_GB - weights_gb) * 1024 * 1024 * L.KV_SAFE / kv_kb))


def call_seconds(n, k, cap, slots):
    """n 문제를 k 샘플·cap 으로 한 호출에 돌릴 때의 예상 초 (VRAM 한도면 그룹 분할)."""
    g = n if slots is None else max(1, min(n, int(slots // (cap * k))))
    ngr = -(-n // g)
    return sum(L.seconds_for(min(g, n - i * g) * k, cap, 1.0) for i in range(ngr)), ngr


def max_problems(k, cap, budget, slots):
    m = 0
    for cand in range(1, 201):
        sec, _ = call_seconds(cand, k, cap, slots)
        if sec <= budget:
            m = cand
        else:
            break
    return m


def load_h14():
    hits = sorted(glob.glob(str(ROOT / "experiments/results/h14/**/*_h14.json"), recursive=True))
    rec = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    return [{"y": r["y"], "toks": r["by_kind"]["orig"]["tokens"], "problem": r["problem"]}
            for r in rec if r["by_kind"]["orig"]["tokens"]]


def pct_rank(x):
    """작은 값이 0, 큰 값이 1 인 백분위 (동점은 평균 순위)."""
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):                               # 동점 평균
        m = x == v
        r[m] = r[m].mean()
    return r / max(1, len(x) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=825.0, help="호출당 예산(초). 3300/4 = 825")
    a = ap.parse_args()
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ---------- A) 커버리지 ----------
    out(f"A) 한 호출(예산 {a.budget:.0f}s)에 들어가는 **최대 문제 수** · 괄호는 그 설정의 train AUC(격자)")
    out(f"   {'모델':16s} " + "  ".join(f"k{k}·c{c}" for k in (1, 2) for c in (8192, 12288, 16384)))
    for name, w, kv in MODELS:
        slots = slots_for(w, kv)
        cells = []
        for k in (1, 2):
            for c in (8192, 12288, 16384):
                cells.append(f"{max_problems(k, c, a.budget, slots):9d}")
        out(f"   {name:16s} " + "  ".join(cells))
    out(f"   {'train AUC':16s} " + "  ".join(f"{L.grid_auc(k, c):9.3f}" for k in (1, 2) for c in (8192, 12288, 16384)))
    out()
    out("   읽는 법: 최종셋이 모델당 35문제라면 cap 8192·k1 로도 20문제 안쪽만 생성할 수 있다(Olmo 는 그룹 분할로 더 적다).")
    out("   cap 12288 은 8~9문제, 16384 는 5~6문제 — 즉 **전수 생성으로는 신호 구간을 못 산다**.")
    out()

    # ---------- B) 혼합안 ----------
    rows = load_h14()
    y = np.array([r["y"] for r in rows])
    toks1 = np.array([r["toks"][0] for r in rows], float)          # k=1 = 첫 샘플
    toks2 = np.array([np.mean(r["toks"][:2]) for r in rows], float)
    nsym = np.array([float(L.n_sym(r["problem"])) for r in rows])
    words = np.array([max(1, len(re.findall(r"[A-Za-z]+", r["problem"]))) for r in rows], float)
    n = len(y)
    rng = np.random.default_rng(0)
    PY = np.array([rng.permutation(y) for _ in range(a.perms)], float)   # (perms, n) 라벨 섞기
    n1, n0 = float(y.sum()), float((1 - y).sum())

    def band(score):
        """AUC 와 null 95%. null 은 Mann-Whitney 를 **순위 행렬곱**으로 한 번에 계산한다
        (셀마다 sklearn 을 1000번 부르면 수 분씩 걸린다)."""
        r = pct_rank(score) * (len(score) - 1) + 1                        # 1..n 평균 순위
        auc = (float(r[y == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
        nl = (PY @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return auc, lo, hi, ("▲" if auc > hi else ("▼" if auc < lo else " "))

    def two_tier(gen, tk, nsym):
        """혼합 판정: **각 군을 따로 중앙값 분할**한다(h20e 가 실제로 하는 일).
        생성군은 긴 쪽 절반 False, 비생성군은 n_sym 큰 쪽 절반 False.
        군을 하나의 점수로 합치지 않는다 — 토큰 수와 n_sym 은 단위가 달라 순위를 이어 붙이면 섞인다."""
        pred = np.ones(n, bool)
        for mask, val in ((gen, tk), (~gen, nsym)):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            o = sorted(idx, key=lambda i: (-float(val[i]), -i))
            for i in o[:len(idx) // 2]:
                pred[i] = False
        return pred

    def acc_band(pred):
        """예측 벡터의 정확도와 null 95%(라벨 섞기). h20e 의 출력은 bool 벡터이므로 AUC 가 아니라 정확도를 본다."""
        acc = float((pred == (y == 1)).mean())
        nl = (PY == pred[None, :].astype(float)).mean(axis=1)
        lo, hi = np.percentile(nl, [2.5, 97.5])
        return acc, lo, hi, ("▲" if acc > hi else ("▼" if acc < lo else " "))

    thr_mid = float(np.median(nsym))
    rules = {
        "R1 애매한 것 먼저": np.argsort(np.abs(nsym - thr_mid), kind="mergesort"),
        "R2 n_sym 큰 것 먼저": np.argsort(-nsym, kind="mergesort"),
        "R3 입력 순서": np.arange(n),
    }

    half = np.zeros(n, bool); half[np.arange(n) % 2 == 0] = True   # True 비율 ~50% 인 아무 예측
    _, nlo, nhi, _ = acc_band(half)
    out(f"B) 혼합안: m 개만 생성(cap 별) + 나머지는 n_sym 순위 · 문제 {n} (robust {int(y.sum())}/spurious {int(n - y.sum())}) "
        f"· null {a.perms}회 · **Skywork 데이터 1모델**")
    out(f"   null 95% (True 비율 50% 예측): [{nlo:.3f}, {nhi:.3f}] · 상수 예측(전부 True) {y.mean():.3f}")
    out("   판정 = 각 군 따로 중앙값 분할 · 숫자는 **정확도**(채점기 기준) · null 은 라벨 섞기 1000회")
    for k, toks in ((1, toks1), (2, toks2)):
        for cap in (8192, 12288):
            # 이 cap·k 에서 Skywork 가 한 호출에 생성할 수 있는 문제 수
            fit = max_problems(k, cap, a.budget, slots_for(15.2, 56.0))
            out()
            out(f"   k={k} · cap {cap} (train 전수 AUC {L.grid_auc(k, cap):.3f} · 예산상 Skywork 최대 {fit}문제)")
            out(f"     {'규칙':20s} " + "  ".join(f"m={m:<2d}" for m in (0, 5, 10, 15, 20, 25, 33)))
            tk = np.minimum(toks, cap)
            for label, order in rules.items():
                cells = []
                for m in (0, 5, 10, 15, 20, 25, 33):
                    gen = np.zeros(n, bool)
                    gen[order[:m]] = True
                    pred = two_tier(gen, tk, nsym)
                    acc, lo, hi, flag = acc_band(pred)
                    cells.append(f"{acc:.3f}{flag}")
                out(f"     {label:20s} " + "  ".join(f"{c:7s}" for c in cells))
            sec, ngr = call_seconds(fit, k, cap, slots_for(15.2, 56.0))
            out(f"     시간: m={fit} 이면 호출당 {sec:.0f}s (그룹 {ngr}) → 4호출 {4 * sec / 60:.1f}분 · "
                f"m=0 은 0s(텍스트만)")
    out()
    out("   m=0 열은 n_sym 단독 분할(= h15A 계열), m=33 열은 전수 생성이다. 상수 예측(전부 True)의 정확도는 "
        f"{y.mean():.3f} 다.")
    out("   같은 33문제를 반복 사용하는 탐색이므로 null 을 넘은 칸도 '설계 후보' 수준으로만 읽을 것.")
    # ---------- C) 예산을 '문제' 대신 '모델'에 몰아주기 ----------
    out()
    out("C) 같은 예산을 **모델에 몰아주는** 안 (문제는 전수, 모델만 줄인다)")
    out("   3600s 중 3300s 를 쓴다고 보고, 문제 35개를 한 모델에서 전수 생성하는 비용:")
    for k in (1, 2):
        for cap in (8192, 12288):
            secs = {}
            for name, w, kv in MODELS:
                sec, ngr = call_seconds(35, k, cap, slots_for(w, kv))
                secs[name] = sec
            # 싼 모델부터 담아 몇 개가 들어가는지 센다 (가장 비싼 모델 기준으로 나누면 과소평가된다)
            order = sorted(secs.items(), key=lambda kv: kv[1])
            tot, picked = 0.0, []
            for nm, sc in order:
                if tot + sc <= 3300:
                    tot += sc; picked.append(nm.split("-")[0])
            out(f"     k{k}·c{cap}: 모델당 " + " · ".join(f"{n.split('-')[0]} {s:.0f}s" for n, s in secs.items())
                + f" → 3300s 안에 **{len(picked)}개 모델**({', '.join(picked) or '없음'}) · 합 {tot:.0f}s"
                + f" · train AUC {L.grid_auc(k, cap):.3f}")
    out()
    out("   이 안의 기대값은 '모델별 성분이 실재하는가'에 달려 있다 — H20-M(지휘 #63)의 모델 간 순위상관이 그 답이다.")
    out("   상관이 높으면(문제 효과 지배) 두 모델만 제대로 재도 나머지 두 모델에서 잃는 것이 그대로 남아 이득이 없다.")
    (ROOT / "experiments" / "results" / "h20e_coverage_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
