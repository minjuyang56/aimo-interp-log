"""#SUB-GEN2048 포맷·수치 스모크 — GPU 0 · 모델 로드 없음. (지휘 #SUB-GEN2048 (2))

묻는 것 다섯. 전부 **계측**이고 주장이 아니다:
 (1) 제출물의 `probe_score` 가 학습 때 sklearn 파이프라인 점수를 **재현**하는가
     — 재현되지 않으면 가중치가 무의미해진다. 이 검사를 통과하지 않으면 제출물이 아니다.
 (2) 모델별 임계 `thr` 로 판정했을 때 train True 비율
 (3) h15C 폴백이 **h15C 제출물과 같은 판정**을 내는가 (복제 검증)
 (4) 가중치 없는 `model_id`(Qwen3.5 · gpt-oss · 미지 · 빈 문자열)가 **모델을 열지 않고** 폴백으로 가는가
 (5) `debpe` 가드가 Ġ/Ċ 를 걷어내고 정상 텍스트는 건드리지 않는가

사용:  .venv/Scripts/python.exe analysis/sub2048_smoke.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "experiments" / "results" / "probe"
TEXTS = [
    "What is the value of $255^2 - 245^2$?",
    "Find the number of positive divisors of 2024.",
    "Let $a$, $b$, $c$ be the roots of $x^3-2x^2+3x-4=0$. " * 9,
    "In triangle $ABC$, $AB=13$, $BC=14$, $CA=15$. Find the area. " * 3,
    " ".join(["word"] * 59),
    " ".join(["word"] * 60),
    "x",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/results/sub2048_smoke.txt"))
    a = ap.parse_args()
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    tmp = Path(os.environ.get("TEMP", "/tmp"))
    os.environ["SUB2048_CACHE"] = str(tmp / "sub2048_smoke_cache.json")
    os.environ["SUB2048_STATE"] = str(tmp / "sub2048_smoke_state.json")
    for f in (os.environ["SUB2048_CACHE"], os.environ["SUB2048_STATE"]):
        Path(f).unlink(missing_ok=True)

    sys.path.insert(0, str(ROOT / "solutions" / "sub2048"))
    import sub2048_lib as S

    art = S.art()
    say(f"#SUB-GEN2048 스모크 (GPU 0 · 모델 로드 없음)")
    say(f"아티팩트: 창 {art['win']} · 층 규칙 {art['layer_rule']} · PCA {art['npc']} · "
        f"가중치 있는 모델 {len(art['per_model'])}개")
    say("")

    # ---------- (1)(2)
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    say("## (1)(2) 가중치 적용식 재현 · 임계")
    ok12 = True
    for mid, w in art["per_model"].items():
        d = np.load(PROBE / w["source"], allow_pickle=True)
        X = d[f"h{w['layer']}"].astype(np.float64)
        y = d["y"].astype(int)
        sc = StandardScaler().fit(X)
        Z = sc.transform(X)
        pca = PCA(n_components=min(art["npc"], X.shape[0] - 1, X.shape[1]), random_state=0).fit(Z)
        m = LogisticRegression(C=w["C"], class_weight="balanced", max_iter=5000).fit(pca.transform(Z), y)
        ref = m.decision_function(pca.transform(Z))
        got = np.array([S.probe_score(w, x) for x in X])
        md = float(np.abs(ref - got).max())
        pred = got > w["thr"]
        say(f"   {w['alias']:28s} 층 {w['layer']:>2d} · n {len(y):>3d} · 차원 {X.shape[1]} · "
            f"**최대절대차 {md:.2e}**")
        say(f"   {'':28s} thr {w['thr']:+.3f} → train True {pred.mean():.3f} · "
            f"참고 임계 0 → {float((got > 0).mean()):.3f} (라벨 robust {y.mean():.3f})")
        ok12 &= md < 1e-9
    say(f"   판정: {'통과 — 적용식이 학습 파이프라인과 동일' if ok12 else '!! 실패'}")
    say("")

    # ---------- (3)
    say("## (3) h15C 폴백 복제 검증")
    sys.path.insert(0, str(ROOT / "solutions" / "h15C"))
    import h15_lib

    ref3 = h15_lib.decide(TEXTS, "C")
    got3 = S.h15c_decide(TEXTS)
    say(f"   문제 단어수   {[len(t.split()) for t in TEXTS]}")
    say(f"   h15C 제출물   {['T' if x else 'F' for x in ref3]}")
    say(f"   sub2048 폴백  {['T' if x else 'F' for x in got3]}")
    ok3 = ref3 == got3
    say(f"   판정: {'통과 — 같은 판정' if ok3 else '!! 실패 — 폴백이 h15C 와 다르다'}")
    say("")

    # ---------- (4)
    say("## (4) 가중치 없는 model_id → 모델을 열지 않고 폴백")
    ok4 = True
    for mid in ("Qwen/Qwen3.5-4B", "openai/gpt-oss-120b", "some/unknown-model", ""):
        out = S.decide(TEXTS[:4], model_id=mid)
        pure = all(type(x) is bool for x in out)
        same = out == S.h15c_decide(TEXTS[:4])
        say(f"   {(mid or '(빈 문자열)'):26s} → {['T' if x else 'F' for x in out]} · "
            f"순수 bool {pure} · h15C 와 동일 {same}")
        ok4 &= pure and len(out) == 4 and same
    say(f"   판정: {'통과' if ok4 else '!! 실패'}")
    say("")

    # ---------- (5)
    say("## (5) debpe 가드 (#TOK-AUDIT 재발 방지)")
    # `Ġ` 는 **공백**으로 복원된다 → `\boxed{ 90}`. 그게 vLLM 손상 전의 원문이고,
    # `extract_answer` 가 strip 해서 90 으로 읽는다(#TOK-AUDIT 의 `same_answer("Ġ90", 90)` 단위검사와 같은 지점).
    bad = "The answer is \\boxed{\u012090}\u010aDone."
    want = "The answer is \\boxed{ 90}\nDone."
    good = "The answer is \\boxed{90}\nDone."
    f_bad, f_good = S.debpe(bad), S.debpe(good)
    ok5 = (f_bad == want) and (f_good == good)
    say(f"   손상 입력 → {f_bad!r}  (기대 {want!r})")
    say(f"   정상 입력 → 변화 {'없음' if f_good == good else '있음(!!)'}")
    say(f"   판정: {'통과 — 손상은 고치고 정상은 건드리지 않는다' if ok5 else '!! 실패'}")
    say("")

    allok = ok12 and ok3 and ok4 and ok5
    say(f"# 전체: {'통과' if allok else '!! 실패'}")
    say("")
    say("검사하지 않은 것(여기서는 불가): 워커에서의 실제 생성·prefill 특징이 학습 npz 와")
    say("같은 값인지. 그건 GPU 가 필요하다 — 로컬 8 GB 에서 2048토큰 생성은 시간이 과하고,")
    say("RunPod 견적은 보고에만 적었다(지휘 (3): 띄우지 말 것).")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
