"""H-FIN 자연종료 규칙 — 저장 데이터로 $0 계산 (지휘 #ENDGAME / H-FIN Step 1).

가설([author]): **robust ⟺ 그 모델의 시도가 토큰 상한 전에 스스로 끝난다.**
    score_fin = (cap 안에서 자연종료한 샘플 수) / k      큼 = robust
    판정 = score_fin >= 0.5   (절대 임계. EconLearn 의 '3 중 2' 는 0.667 이다)

**이것은 탐색이다 — 사전등록이 아니다.** 이유를 먼저 적는다:
  우리는 이 양을 **이미 봤다**. H20b(원본 완주율)가 같은 양이고 CLAUDE.md 에
  `AUC 0.671` (cap 16384) · `상한 8192 로 다시 재면 0.779 ▲` 로 기록돼 있다.
  그래서 '사전등록 1회' 의 자격이 없다. 새로 재는 것은 **판정 규칙**이다 —
  절대 임계 0.5 의 정확도는 한 번도 계산한 적이 없다(우리는 AUC 와 중앙값 분할만 봤다).

데이터: `by_kind['orig']` 에 샘플별 `tokens` 와 집계 `n_fin` 이 들어 있다.
  저장 cap 을 cap_rec 라 하면 `tokens[i] < cap_rec` ⟺ 그 샘플이 자연종료했다.
  따라서 **cap_sim <= cap_rec 에 대해 오프라인 시뮬이 정확하다**:
      자연종료(cap_sim) ⟺ tokens[i] < cap_rec  AND  tokens[i] <= cap_sim
  → [author]의 Step 2(k=3 cap 4096 Kaggle 실행)의 cap 축은 **GPU 없이 재현된다**.
    재현 안 되는 축은 k(샘플 수)뿐이다.

검증 게이트: 저장된 `n_fin` 과 `sum(tokens[i] < cap_rec)` 가 어긋나면 멈춘다.

사용: .venv/Scripts/python.exe analysis/hfin_eval.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from ttp_eval import auc, loo_acc, null_band  # noqa: E402  — 같은 함수를 import

RES = ROOT / "experiments/results"
OUT = RES / "hfin_report.txt"
L = []

#  (모델, 라벨, 파일, 저장 cap, k)
SRC = [
    ("Skywork", "H14", RES / "h14/h14/Skywork-OR1-Math-7B_h14.json", 16384, 2),
    ("Skywork", "h20m", RES / "h20m/aimo-h20m-skywork-or1-math-7b/h14/Skywork-OR1-Math-7B_h14.json", 12288, 1),
    ("Olmo", "h20m", RES / "h20m/aimo-h20m-olmo-3-7b-think/h14/Olmo-3-7B-Think_h14.json", 12288, 1),
    ("DeepSeek", "h20m", RES / "h20m/aimo-h20m-deepseek-r1-0528-qwen3-8b/h14/DeepSeek-R1-0528-Qwen3-8B_h14.json", 12288, 1),
]
CAPS = [2048, 4096, 8192, 12288, 16384]
#  CLAUDE.md 에 기록된 공식 33 기준선 (같은 라벨·같은 n)
CONST_ACC = 0.6061      # '전부 robust' 상수
LEN_ACC = 0.6364        # 길이 h15C, LOO 교정 임계


def out(s=""):
    print(s, flush=True)
    L.append(s)


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = []
    for r in rows:
        o = r["by_kind"].get("orig")
        if not o:
            continue
        recs.append(dict(pidx=int(r["pidx"]), y=int(r["y"]),
                         tokens=[int(t) for t in o["tokens"]],
                         n_fin=int(o["n_fin"]), problem=r["problem"]))
    return recs


def gate(recs, cap_rec, tag):
    """저장 n_fin == sum(tokens < cap_rec) 인지 확인. 어긋나면 멈춘다."""
    bad = [(r["pidx"], r["n_fin"], sum(1 for t in r["tokens"] if t < cap_rec)) for r in recs
           if r["n_fin"] != sum(1 for t in r["tokens"] if t < cap_rec)]
    out("   검증 게이트 %-6s n_fin == #(tokens < cap_rec) : 행 %d · 불일치 **%d**"
        % (tag, len(recs), len(bad)))
    if bad:
        for b in bad[:5]:
            out("      pidx %d: 저장 %d vs 계산 %d" % b)
        raise SystemExit("검증 게이트 실패 — 자연종료 판정을 토큰 수로 복원할 수 없다")


def score(recs, cap_rec, cap_sim):
    """score_fin = cap_sim 안에서 자연종료한 비율. spur = 1 − score_fin."""
    sc = []
    for r in recs:
        fin = sum(1 for t in r["tokens"] if t < cap_rec and t <= cap_sim)
        sc.append(fin / len(r["tokens"]))
    return np.array(sc, float)


def main():
    out("H-FIN 자연종료 규칙 — 저장 데이터 오프라인 평가 (**탐색**, GPU 0, $0)")
    out("   가설: robust ⟺ 시도가 cap 전에 스스로 끝난다 · 판정 score_fin >= 0.5 (절대 임계)")
    out("   **사전등록 아님**: 같은 양을 H20b 로 이미 봤다 (AUC 0.671 @16384 · 0.779 @8192)")
    out("   기준선(공식 33, CLAUDE.md): 상수 '전부 robust' %.4f · 길이 h15C LOO %.4f" % (CONST_ACC, LEN_ACC))
    out()

    for model, tagsrc, path, cap_rec, k in SRC:
        if not path.is_file():
            out("== %-9s %-5s **파일 없음** %s" % (model, tagsrc, path))
            continue
        recs = load(path)
        y = np.array([r["y"] for r in recs], int)
        out("== %-9s %-5s  n=%d (robust %d / spurious %d) · 저장 cap %d · k=%d"
            % (model, tagsrc, len(recs), int(y.sum()), int((y == 0).sum()), cap_rec, k))
        gate(recs, cap_rec, tagsrc)
        out("   %-7s %7s %7s  %8s %8s  %8s %-18s %s"
            % ("cap_sim", "robust", "spur", "임계0.5", "LOO교정", "AUC", "AUC null 95%", "값칸"))
        for cs in CAPS:
            if cs > cap_rec:
                continue
            s = score(recs, cap_rec, cs)
            spur = 1.0 - s
            #  절대 임계 0.5: score_fin >= 0.5 → robust
            pred = s >= 0.5
            acc_abs = float((pred == (y == 1)).mean())
            acc_loo, _ = loo_acc(spur, y)
            a = auc(spur, y)
            (nl, nu), (al, au) = null_band(spur, y, reps=1000, seed=0)
            mr = float(s[y == 1].mean()) if (y == 1).any() else float("nan")
            ms = float(s[y == 0].mean()) if (y == 0).any() else float("nan")
            star = " ▲" if a > au else ""
            out("   %-7d %7.3f %7.3f  %8.4f %8.4f  %8.3f [%.2f, %.2f]%-6s %d칸"
                % (cs, mr, ms, acc_abs, acc_loo, a, al, au, star, len(set(s.tolist()))))
        out("      (robust/spur 열 = 그 군의 평균 score_fin · 값칸 = 서로 다른 점수 개수)")
        out()

    out("## 읽는 법 — 무엇을 재현했고 무엇을 못 했나")
    out("   * **cap 축은 GPU 없이 재현된다**: 저장된 토큰 수로 더 낮은 cap 의 자연종료를 정확히 센다.")
    out("     [author]의 Step 2(k=3 · cap 4096)에서 cap 4096 칸이 위 표에 이미 들어 있다.")
    out("   * **k 축은 재현 안 된다**: h20m 은 k=1 이라 score_fin 이 {0,1} 2칸뿐이고,")
    out("     H14 Skywork 만 k=2 로 {0,0.5,1} 3칸이다. EconLearn 의 '3 중 2'(임계 0.667)는")
    out("     k=3 이 필요하므로 **우리 저장 데이터로는 그 규칙 자체를 못 쓴다**.")
    out("   * **Qwen3.5-4B 는 데이터가 없다** — h20m 에서 vLLM 이 아키텍처를 못 읽어 제외됐다.")
    out("     즉 Small 4모델 중 3모델이고, 모델평균은 그 3개 평균이다.")
    out("   * **공개 82행 LOTO 는 측정 불가**: train-main-v2 에는 `is_robust`·`max_drop` 만 있고")
    out("     생성 산출물(토큰 수·종료 플래그)이 없다. 그 셋에서 자연종료를 재려면 생성이 필요하다.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))


if __name__ == "__main__":
    main()
