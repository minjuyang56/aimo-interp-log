"""hprobe 제출 아티팩트 — 모델별 프로브 가중치를 JSON 으로 굽는다. GPU 0. (지휘 #78-1)

사전등록된 최종 프로토콜(CLAUDE.md "최종 프로브 평가 프로토콜", 커밋 e749b2e)과 **같은 파이프라인**을 쓴다:
  특징   = 문제 텍스트 forward 1회 hidden state, 문제 본문 토큰 구간 평균 풀링 (experiments/probe_pilot.py)
  층 후보 = hidden_states 인덱스 L//4 · L//2 · 3L//4 · L — **내부 CV 로만** 고른다(여기서는 train 전체의
            StratifiedKFold 4 폴드 밖 AUC 로 고르고, 고른 설정으로 train 전체에 다시 적합한다 = 최종 모델 적합 단계)
  축소   = StandardScaler → PCA 24성분
  분류기 = 로지스틱 L2, class_weight="balanced", C ∈ {0.003, 0.01, 0.03, 0.1, 0.3}

저장 형식(제출물이 numpy 만으로 복원할 수 있게 전부 평문 숫자):
  per_model[model_id] = {layer, C, mean, scale, pca_mean, comps(24×D), coef(24), intercept, train_auc, n, n_robust}
점수 = ((x − mean)/scale − pca_mean) @ comps.T @ coef + intercept.  **높을수록 robust.**

지금은 Skywork 88라벨만 있다. DeepSeek·Olmo 라벨이 오면 `--alias` 를 추가해 같은 명령을 다시 돌리면
**가중치만 갈린다** — 제출물 코드는 손대지 않는다.

사용:  .venv/Scripts/python.exe analysis/build_hprobe_artifacts.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "experiments" / "results" / "probe"
OUT = ROOT / "solutions" / "hprobe" / "artifacts" / "hprobe.json"
CS = (0.003, 0.01, 0.03, 0.1, 0.3)
NPC = 24
ALIAS2ID = {                                   # npz 별칭 → 실전에서 그대로 넘어오는 model_id
    "Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
    "DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think",
    "Qwen3_5-4B": "Qwen/Qwen3.5-4B",
}


def pipe(Xtr, ytr, C):
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr)
    pca = PCA(n_components=min(NPC, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=0).fit(Ztr)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(pca.transform(Ztr), ytr)
    return sc, pca, m


def apply(sc, pca, m, X):
    return m.decision_function(pca.transform(sc.transform(X)))


def pick(feats, y, layers):
    """층·C 를 **폴드 밖 점수의 AUC** 로 고른다(라벨을 직접 보고 고르는 게 아니라 CV 로)."""
    best, key = -1, None
    skf = StratifiedKFold(4, shuffle=True, random_state=0)
    for l in layers:
        for C in CS:
            s = np.zeros(len(y))
            for tr, te in skf.split(feats[l], y):
                sc, pca, m = pipe(feats[l][tr], y[tr], C)
                s[te] = apply(sc, pca, m, feats[l][te])
            a = roc_auc_score(y, s)
            if a > best:
                best, key = a, (l, C)
    return key, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", action="append", default=None,
                    help="npz 별칭. 여러 번 줄 수 있다. 기본은 _own.npz 가 있는 전부")
    ap.add_argument("--label", default="y", choices=["y", "y_fin"],
                    help="주라벨 y(=label, 미완주=오답). 사전등록대로 기본값을 쓴다")
    a = ap.parse_args()
    aliases = a.alias or sorted(p.name[:-len("_own.npz")] for p in PROBE.glob("*_own.npz"))
    art = {"npc": NPC, "label": a.label, "per_model": {}, "notes": []}
    for al in aliases:
        f = PROBE / f"{al}_own.npz"
        if not f.is_file():
            print(f"  건너뜀: {f.name} 없음")
            continue
        z = np.load(f, allow_pickle=True)
        layers = [int(x) for x in z["layers"]]
        feats = {l: z[f"h{l}"].astype(np.float64) for l in layers}
        y = z[a.label].astype(int)
        (l, C), auc = pick(feats, y, layers)
        sc, pca, m = pipe(feats[l], y, C)
        s_in = apply(sc, pca, m, feats[l])
        mid = ALIAS2ID.get(al, al)
        art["per_model"][mid] = {
            "alias": al, "layer": l, "layers": layers, "C": C, "dim": int(feats[l].shape[1]),
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "pca_mean": pca.mean_.tolist(), "comps": pca.components_.tolist(),
            "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
            "cv_auc": float(auc), "train_auc": float(roc_auc_score(y, s_in)),
            "train_bacc": float(balanced_accuracy_score(y, (s_in >= np.median(s_in)).astype(int))),
            "n": int(len(y)), "n_robust": int(y.sum()),
        }
        print(f"  {mid:44s} 층 {l:3d} C {C:<6} n {len(y):3d}(robust {int(y.sum())}) "
              f"CV AUC {auc:.3f} · 적합 후 train AUC {roc_auc_score(y, s_in):.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art), encoding="utf-8")
    print(f"\n저장 {OUT.relative_to(ROOT)} · {OUT.stat().st_size / 1e6:.2f} MB · 모델 {len(art['per_model'])}개")
    print("CV AUC 는 '이 가중치가 train 안에서 어느 정도였나' 이고, **사전등록 평가가 아니다**")
    print("(사전등록 평가는 nested CV 1회 — analysis/probe_pilot_eval.py).")


if __name__ == "__main__":
    main()
