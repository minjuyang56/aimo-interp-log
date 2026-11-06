"""H21c 확증 검정 — 새 88문제 Skywork 라벨에서 drop(K=4, v2) 이 재현되는가. GPU 0. (지휘 #76)

사전등록 (측정 전 기록, CLAUDE.md "H21c 확증 검정")
  지표 = **drop 평균, K=4, v2 규칙** (h21cK 제출본과 동일). **부호: 클수록 robust.** 뒤집지 않는다.
  주지표 라벨 = **`label`**(미완주=오답). `label_fin`(완주만)은 대조로 병기.
  검문소 = 셔플 null **1000** · log(단어수) 잔차 AUC · 부트스트랩 95% CI(문제 재추출 1000).
  질문 = 공식 33문제의 **Skywork 0.741** 이 88문제에서 재현되는가.
  보조(탐색, 사전등록 아님) = drop + 프로브 PCA 결합 로지스틱(문제 단위 GroupKFold).

사용:  .venv/Scripts/python.exe analysis/h21c_confirm.py [--perms 1000] [--boot 1000]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OWN = ROOT / "experiments" / "results" / "h21ck" / "Skywork-OR1-Math-7B_own.json"
OFF = ROOT / "experiments" / "results" / "h21ck" / "Skywork-OR1-Math-7B.json"
PROBE = ROOT / "experiments" / "results" / "probe" / "Skywork-OR1-Math-7B_own.npz"
K = 4


def drop_k(rec, k=K, rule="v2"):
    v = [x for lab, x in rec["drops"].items() if lab.startswith(rule + ":")][:k]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rec = json.loads(OWN.read_text(encoding="utf-8"))
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    d = np.array([drop_k(r) if drop_k(r) is not None else np.nan for r in rec], float)
    ok = ~np.isnan(d)
    y = np.array([r["y"] for r in rec])[ok]
    yf = np.array([-1 if r.get("y_fin") is None else r["y_fin"] for r in rec])[ok]
    words = np.array([len(r["problem"].split()) for r in rec], float)[ok]
    dd = d[ok]
    lw = np.log(np.maximum(1, words))
    rng = np.random.default_rng(0)

    def resid(x):
        X = np.column_stack([np.ones(len(x)), lw])
        b, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ b

    def report(tag, yy, score, primary):
        if len(set(yy)) < 2:
            out(f"| {tag} | 한 클래스 | — | — | — | — |")
            return None
        auc = roc_auc_score(yy, score)
        nl = [roc_auc_score(rng.permutation(yy), score) for _ in range(a.perms)]
        lo, hi = np.percentile(nl, [2.5, 97.5])
        bt = []
        for _ in range(a.boot):
            i = rng.integers(0, len(yy), len(yy))
            if len(set(yy[i])) == 2:
                bt.append(roc_auc_score(yy[i], score[i]))
        blo, bhi = np.percentile(bt, [2.5, 97.5]) if bt else (np.nan, np.nan)
        ra = roc_auc_score(yy, resid(score))
        bacc = balanced_accuracy_score(yy, (score >= np.median(score)).astype(int))
        flag = "▲" if auc > hi else ("▼" if auc < lo else "")
        star = "*" if primary else ""
        out(f"| {tag}{star} | {int(yy.sum())}/{int((1 - yy).sum())} | **{auc:.3f}**{flag} | {bacc:.3f} | "
            f"[{lo:.2f}, {hi:.2f}] | [{blo:.2f}, {bhi:.2f}] | {ra:.3f} |")
        return auc

    out("## H21c 확증 검정 — 자체 88문제 (Skywork, K=4 v2, drop 평균)")
    out()
    out(f"변형이 생성된 문제 {int(ok.sum())}/{len(rec)} · 셔플 null {a.perms} · 부트스트랩 {a.boot}")
    out()
    out("| 라벨 | robust/spurious | AUC | balanced acc | null95% | 부트스트랩 95% CI | 길이잔차 AUC |")
    out("|---|---|---|---|---|---|---|")
    auc_label = report("label (미완주=오답)", y, dd, True)
    mask = yf >= 0
    auc_fin = report("label_fin (완주만)", yf[mask], dd[mask], False) if mask.any() else None
    out("   * = 사전등록 주지표")
    out()

    # 공식 33문제 비교 (같은 스크립트가 만든 파일)
    if OFF.is_file():
        ro = json.loads(OFF.read_text(encoding="utf-8"))
        do = np.array([drop_k(r) if drop_k(r) is not None else np.nan for r in ro], float)
        m2 = ~np.isnan(do)
        yo = np.array([r["y"] for r in ro])[m2]
        auc_off = roc_auc_score(yo, do[m2])
        out(f"**공식 33문제 같은 설정**: AUC **{auc_off:.3f}** (n={int(m2.sum())}, robust {int(yo.sum())}/spurious {int((1 - yo).sum())}) "
            f"— H21cK 표의 Skywork K=4 값과 같아야 한다(0.658).")
        if auc_label is not None:
            out(f"**재현 여부**: 자체 88문제 {auc_label:.3f} vs 공식 33문제 {auc_off:.3f} · 차이 {auc_label - auc_off:+.3f}")
        out()

    # ---- 보조(탐색): drop + 프로브 PCA 결합 ----
    out("### 보조(탐색, 사전등록 아님) — drop + 프로브 PCA 결합 로지스틱")
    out()
    if not PROBE.is_file():
        out("프로브 npz 없음 → 생략")
    else:
        z = np.load(PROBE, allow_pickle=True)
        layers = [int(x) for x in z["layers"]]
        pid_probe = list(z["pidx"])
        idx = [pid_probe.index(r["pidx"]) for r in rec]        # 같은 문제끼리 맞춘다
        idx = np.array(idx)[ok]
        out("| 특징 | AUC | balanced acc | null95% |")
        out("|---|---|---|---|")
        for name, build in (("drop 단독", lambda l: dd.reshape(-1, 1)),
                            ("프로브 단독", lambda l: z[f"h{l}"][idx]),
                            ("drop + 프로브", lambda l: np.column_stack([dd, z[f"h{l}"][idx]]))):
            best = None
            for l in layers:
                X = build(l)
                skf = StratifiedKFold(5, shuffle=True, random_state=0)
                s = np.zeros(len(y))
                for tr, te in skf.split(X, y):
                    sc = StandardScaler().fit(X[tr])
                    Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
                    if X.shape[1] > 24:                         # 폴드 학습분으로만 PCA
                        pca = PCA(n_components=min(24, len(tr) - 1), random_state=0).fit(Ztr)
                        Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
                    m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit(Ztr, y[tr])
                    s[te] = m.decision_function(Zte)
                au = roc_auc_score(y, s)
                if best is None or au > best[0]:
                    best = (au, s, l)
                if name == "drop 단독":
                    break                                       # 층과 무관
            au, s, l = best
            nl = [roc_auc_score(rng.permutation(y), s) for _ in range(a.perms)]
            lo, hi = np.percentile(nl, [2.5, 97.5])
            bacc = balanced_accuracy_score(y, (s >= np.median(s)).astype(int))
            tag = name + (f" (층 {l})" if name != "drop 단독" else "")
            out(f"| {tag} | {au:.3f}{'▲' if au > hi else ''} | {bacc:.3f} | [{lo:.2f}, {hi:.2f}] |")
        out()
        out("층을 AUC 로 고른 값이라 **낙관 편향**이 있다(폴드 내부 선택이 아니다) — 조합 가치만 보는 탐색이다.")
    (ROOT / "experiments" / "results" / "h21c_confirm_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n저장: experiments/results/h21c_confirm_report.txt")


if __name__ == "__main__":
    main()
