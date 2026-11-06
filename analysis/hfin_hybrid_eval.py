"""H-FIN 하이브리드 — 모델별 라우팅의 공식 33 점수 (지휘 #FINAL 1, **사후·탐색 표기**).

라우팅(지휘 지정): **Olmo·Skywork 는 H-FIN**(자연종료 비율 >= 0.5) ·
                   **DeepSeek·Qwen3.5 는 h15C**(-log 단어수 >= 고정 임계).

**이것은 사후 선택이다 — 사전등록이 아니다.** 라우팅은 H-FIN 모델별 결과를 본 뒤
'게이트를 통과한 두 모델만 H-FIN' 으로 고른 것이고, CLAUDE.md·리포트에 EXPLORATORY 로 적는다.

게이트(지휘 지정): 모델평균 정확도가 **0.65 초과** 그리고 **자기 셔플 null 95% 상단 초과**.
둘 다 넘지 못하면 새 제출물 없음(zip 작업 중단).

데이터: h20m 저장 산출물(k=1 · 저장 cap 12288)에서 cap 12288 자연종료를 센다.
  Skywork·Olmo 를 **같은 출처·같은 k** 로 맞춘다 — H14 Skywork 는 k=2 라 섞지 않는다.
h15C 는 텍스트만 쓰므로 DeepSeek·Qwen3.5 두 칸이 **같은 예측**이다(모델 무관 규칙).

사용: .venv/Scripts/python.exe analysis/hfin_hybrid_eval.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "solutions" / "sub2048"))

from sub2048_lib import H15C_THR, log_words  # noqa: E402  — 출하된 제출물의 식을 그대로 import

RES = ROOT / "experiments/results"
OUT = RES / "hfin_hybrid_report.txt"
L = []

H20M = {
    "Skywork": RES / "h20m/aimo-h20m-skywork-or1-math-7b/h14/Skywork-OR1-Math-7B_h14.json",
    "Olmo": RES / "h20m/aimo-h20m-olmo-3-7b-think/h14/Olmo-3-7B-Think_h14.json",
    "DeepSeek": RES / "h20m/aimo-h20m-deepseek-r1-0528-qwen3-8b/h14/DeepSeek-R1-0528-Qwen3-8B_h14.json",
}
CAP_REC = 12288
CAP_SIM = 12288
CONST_ACC = 0.6061
LEN_ACC_LOO = 0.6364


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
        recs.append(dict(pidx=int(r["pidx"]), y=int(r["y"]), problem=r["problem"],
                         tokens=[int(t) for t in o["tokens"]], n_fin=int(o["n_fin"])))
    return recs


def hfin_pred(recs):
    """score_fin = cap_sim 안에서 자연종료한 비율. robust iff >= 0.5."""
    sc = np.array([sum(1 for t in r["tokens"] if t < CAP_REC and t <= CAP_SIM) / len(r["tokens"])
                   for r in recs], float)
    return sc >= 0.5, sc


def h15c_pred(texts):
    return np.array([(-log_words(t)) >= H15C_THR for t in texts], bool)


def main():
    out("H-FIN 하이브리드 — 모델별 라우팅의 공식 33 점수  (**사후 선택 · EXPLORATORY**)")
    out("   라우팅: Olmo·Skywork = H-FIN(자연종료>=0.5) · DeepSeek·Qwen3.5 = h15C(길이 고정임계)")
    out("   게이트: 모델평균 정확도 > 0.65  그리고  > 자기 셔플 null 95% 상단")
    out("   기준선: 상수 '전부 robust' %.4f · 길이 h15C(LOO 교정) %.4f" % (CONST_ACC, LEN_ACC_LOO))
    out()

    base = load(H20M["Skywork"])
    key = [r["problem"] for r in base]
    y = np.array([r["y"] for r in base], int)
    out("공식 33 기준판: n=%d (robust %d / spurious %d)" % (len(y), int(y.sum()), int((y == 0).sum())))

    #  정렬 게이트 — 세 파일의 문제 텍스트·라벨이 같은 순서인지 확인한다
    for m, p in H20M.items():
        recs = load(p)
        same_txt = sum(1 for a, b in zip(key, [r["problem"] for r in recs]) if a == b)
        same_y = int((np.array([r["y"] for r in recs], int) == y).sum())
        out("   정렬 게이트 %-9s 행 %d · 문제텍스트 일치 %d/%d · 라벨 일치 %d/%d"
            % (m, len(recs), same_txt, len(key), same_y, len(y)))
        if same_txt != len(key) or same_y != len(y):
            raise SystemExit("정렬 게이트 실패 — 모델 간 문제/라벨 순서가 다르다")
    out()

    preds, scores = {}, {}
    for m in ("Skywork", "Olmo"):
        p, s = hfin_pred(load(H20M[m]))
        preds[m], scores[m] = p, s
    pl = h15c_pred(key)
    preds["DeepSeek"], preds["Qwen3.5"] = pl, pl

    out("%-10s %-8s %8s %9s %9s" % ("모델", "규칙", "정확도", "True비율", "서로다른값"))
    accs = []
    for m in ("Skywork", "Olmo", "DeepSeek", "Qwen3.5"):
        rule = "H-FIN" if m in scores else "h15C"
        a = float((preds[m] == (y == 1)).mean())
        nv = len(set(scores[m].tolist())) if m in scores else len(set((-np.array([log_words(t) for t in key])).tolist()))
        accs.append(a)
        out("%-10s %-8s %8.4f %9.3f %9d" % (m, rule, a, float(preds[m].mean()), nv))
    hyb = float(np.mean(accs))
    out()
    out("**하이브리드 모델평균 정확도 = %.4f**" % hyb)

    #  셔플 null: 예측 고정 · 라벨만 섞어 같은 모델평균 정확도를 1000회
    rng = np.random.default_rng(0)
    P = np.stack([preds[m] for m in ("Skywork", "Olmo", "DeepSeek", "Qwen3.5")])
    yb = (y == 1)
    nulls = np.empty(1000, float)
    for i in range(1000):
        ys = rng.permutation(yb)
        nulls[i] = float((P == ys[None, :]).mean(axis=1).mean())
    lo, up = float(np.quantile(nulls, 0.025)), float(np.quantile(nulls, 0.975))
    out("   셔플 null 95%% [%.4f, %.4f] (1000회 · 예측 고정, 라벨만 섞음)" % (lo, up))
    out()

    g1 = hyb > 0.65
    g2 = hyb > up
    out("게이트 1  정확도 %.4f > 0.65        : %s" % (hyb, "통과" if g1 else "**미달**"))
    out("게이트 2  정확도 %.4f > null 상단 %.4f : %s" % (hyb, up, "통과" if g2 else "**미달**"))
    out()
    if g1 and g2:
        out("판정: **통과** → dist/hfin-small.zip 빌드로 간다")
    else:
        out("판정: **미달** → 사전등록 규칙대로 **새 제출물 없음. zip 작업 중단.**")
    out()
    out("참고 — 개별 H-FIN 값(hfin_report.txt 와 같은 계산):")
    for m in ("Skywork", "Olmo"):
        s = scores[m]
        out("   %-9s score_fin robust평균 %.3f / spur평균 %.3f · 서로다른값 %d"
            % (m, float(s[y == 1].mean()), float(s[y == 0].mean()), len(set(s.tolist()))))
    out("   h15C 단독(고정임계) 정확도 %.4f · True비율 %.3f"
        % (float((pl == yb).mean()), float(pl.mean())))
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))


if __name__ == "__main__":
    main()
