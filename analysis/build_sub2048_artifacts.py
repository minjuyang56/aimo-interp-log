"""#SUB-GEN2048 제출 아티팩트 — 창 2048 '풀이 도중 벡터' 프로브 가중치를 JSON 으로 굽는다. GPU 0.

(지휘 #SUB-GEN2048, 2026-10-12)

## 왜 창 2048 인가
#DS-RELABEL 재평가에서 **창 2048 만** 두 판(교집합판·보충판) 모두 통과 기준 1·2 를 넘었다
(모델평균 0.871 / 0.872 vs null 상단 0.75 / 0.79 · 기준선 셋 전부보다 높음). 사전등록 주창 512 는
두 판 모두 미달이라 **판정은 '신호 미입증'** 이고, 2048 은 '사전등록 반복이 필요한 유일한 미해결
후보' 로 기록됐다. 리더보드 제출은 그 반복을 **비공개 검증셋**에서 하는 것에 해당한다.

## 학습 설계 (제출물과 반드시 같아야 하는 것)
특징 = 그 모델 자신의 greedy 풀이를 프롬프트 뒤에 붙여 **prefill forward 1회** →
**풀이 토큰 앞 2048개 평균 풀링** (`experiments/probe_gen.py` 와 같은 경로·같은 프롬프트·
같은 토크나이즈 순서: `tok(chat_prompt).ids + tok(sol, add_special_tokens=False).ids[:2048]`).

**층은 `L//2` 하나로 고정한다** (Skywork 14 · Olmo 16 · DeepSeek 18).
지휘가 '사전등록 층 중 미리 정한 것 하나' 를 요구했고, L//2 는 이 프로젝트가 줄곧 쓴 기본 층이다
(H10-B · H16 · H6 전부 L//2). **공개해 둘 오염 하나**: #GEN-PROBE 보고서에서 DeepSeek 의 폴드
내부 CV 가 세 폴드 전부 18(= L//2)을 골랐다는 것을 이미 봤다 — 따라서 이 선택이 완전히
무오염은 아니다. 다만 층을 결과로 고른 것이 아니라 **프로젝트 기본값**을 쓴 것이고, 대안
(폴드별로 다른 층)은 제출물에서 쓸 수 없다(가중치가 하나여야 한다).

축소·분류기 = StandardScaler → PCA 24 → 로지스틱 L2 `class_weight="balanced"`,
C ∈ {0.003, 0.01, 0.03, 0.1, 0.3} 은 **층을 고정했으므로 C 만** 내부 CV(폴드 밖 AUC)로 고른다.

학습 데이터 = **우리 깨끗한 라벨 전체** (지휘 지정):
  Skywork `_gen2048.npz` 30행 · Olmo `_gen2048.npz` 31행 ·
  DeepSeek **`_gen2048_supp.npz` 32행** — 보충판이 #DS-RELABEL 새 라벨의 전체다
  (교집합판 21행은 greedy 커버리지로 잘린 부분집합이라 '전체' 가 아니다).
  Qwen3.5 는 라벨이 0개라 가중치를 만들지 않는다 → 제출물에서 텍스트 폴백으로 간다.

저장 형식(제출물이 numpy 만으로 복원): per_model[model_id] =
  {layer, C, mean, scale, pca_mean, comps(24×D), coef(24), intercept, train_auc, cv_auc,
   n, n_robust, thr_note, train_pred_true_rate}
점수 = ((x − mean)/scale − pca_mean) @ comps.T @ coef + intercept.  **높을수록 robust.**
판정은 **모델별 절대 임계값 `thr` = train 점수 중앙값**. 고른 근거는 코드 주석에 계측과 함께 적었다.

사용:  .venv/Scripts/python.exe analysis/build_sub2048_artifacts.py
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
OUT = ROOT / "solutions" / "sub2048" / "artifacts" / "sub2048.json"
CS = (0.003, 0.01, 0.03, 0.1, 0.3)
NPC = 24
WIN = 2048

# 별칭 → (npz 파일명, 실전 model_id). DeepSeek 만 보충판을 쓴다(깨끗한 라벨 전체).
SETS = [
    ("Skywork-OR1-Math-7B", f"Skywork-OR1-Math-7B_gen{WIN}.npz", "Skywork/Skywork-OR1-Math-7B"),
    ("Olmo-3-7B-Think", f"Olmo-3-7B-Think_gen{WIN}.npz", "allenai/Olmo-3-7B-Think"),
    ("DeepSeek-R1-0528-Qwen3-8B", f"DeepSeek-R1-0528-Qwen3-8B_gen{WIN}_supp.npz",
     "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"),
]


def pipe(X, y, C):
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    pca = PCA(n_components=min(NPC, X.shape[0] - 1, X.shape[1]), random_state=0).fit(Z)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(pca.transform(Z), y)
    return sc, pca, m


def apply(sc, pca, m, X):
    return m.decision_function(pca.transform(sc.transform(X)))


def pick_C(X, y):
    """C 만 폴드 밖 AUC 로 고른다(층은 L//2 로 미리 고정했다).
    소수 클래스가 작아 n_splits 를 그 수로 제한한다 — DeepSeek 는 spurious 3건이다."""
    k = int(min(4, min(np.bincount(y))))
    if k < 2:
        return CS[0], float("nan"), 0
    skf = StratifiedKFold(k, shuffle=True, random_state=0)
    best, bC, bs = -1.0, CS[0], None
    for C in CS:
        s = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            sc, pca, m = pipe(X[tr], y[tr], C)
            s[te] = apply(sc, pca, m, X[te])
        a = roc_auc_score(y, s)
        if a > best:
            best, bC, bs = a, C, s
    return bC, best, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", default=str(ROOT / "experiments/results/sub2048_artifacts.txt"))
    a = ap.parse_args()
    L = []

    def say(s=""):
        print(s, flush=True)
        L.append(s)

    say(f"#SUB-GEN2048 가중치 — 창 {WIN} · 층 L//2 고정 · PCA {NPC} · LR balanced · C 내부 CV")
    say("")
    art = {"win": WIN, "npc": NPC, "layer_rule": "L//2", "per_model": {}}
    for alias, fn, mid in SETS:
        p = PROBE / fn
        if not p.is_file():
            say(f"!! {fn} 없음 — {alias} 건너뜀")
            continue
        d = np.load(p, allow_pickle=True)
        layers = d["layers"].astype(int).tolist()
        lay = sorted(layers)[1]                     # [L//4, L//2, 3L//4, L] 의 두 번째 = L//2
        X = d[f"h{lay}"].astype(np.float64)
        y = d["y"].astype(int)
        C, cv, k = pick_C(X, y)
        sc, pca, m = pipe(X, y, C)
        s_in = apply(sc, pca, m, X)
        auc = roc_auc_score(y, s_in)
        # **임계는 train 점수의 중앙값으로 모델별 고정한다** (계측으로 고른 것, 리더보드 점수는 안 봤다).
        # 왜 LR 의 자연 경계(0)가 아닌가: train 라벨이 robust 쪽으로 쏠려 있어(26/4 · 21/10 · 29/3)
        # 임계 0 은 True 를 **평균 81%** 찍는다 — 상수 예측(always-true 0.50)에 가까워 균형 셋에서
        # 분해능이 거의 없다. 두 클래스 중앙값의 중점(h15 관행)도 train 이 완전 분리돼 0.12 로 같은 값이 된다.
        # train 점수 중앙값이면 True 비율이 **0.495** 로 균형 셋 분해능이 최대이고, 절대 임계라
        # 호출 그룹 크기(리더보드는 모델당 3~5문제)에 무관하다 — 중앙값 분할의 약점이 없다.
        # train 정확도(임계 0 에서 1.00)로 고르지 않은 이유: 30문제·3,584차원이 PCA 24 에서도
        # 완전 분리되므로 그 1.00 은 과적합의 증거이고 임계 선택 근거가 못 된다.
        thr = float(np.median(s_in))
        pred = (s_in > thr).astype(int)
        say(f"## {alias}  ({fn})")
        say(f"   층 {lay} (후보 {layers} 중 L//2) · 차원 {X.shape[1]} · n {len(y)} "
            f"(robust {int(y.sum())} / spurious {int((1 - y).sum())})")
        say(f"   C {C} (내부 CV {k}폴드 밖 AUC **{cv:.3f}**) · train 적합 AUC {auc:.3f} · "
            f"balanced acc {balanced_accuracy_score(y, pred):.3f}")
        pred0 = (s_in > 0).astype(int)
        say(f"   임계 **{thr:+.3f}**(train 점수 중앙값) → train True 비율 **{pred.mean():.3f}** · "
            f"acc {(pred == y).mean():.3f}  |  참고 임계 0 → True {pred0.mean():.3f} · "
            f"acc {(pred0 == y).mean():.3f} (라벨 robust 비율 {y.mean():.3f})")
        say(f"   점수 범위 [{s_in.min():+.2f}, {s_in.max():+.2f}]")
        art["per_model"][mid] = {
            "alias": alias, "layer": int(lay), "C": float(C),
            "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "pca_mean": pca.mean_.tolist(), "comps": pca.components_.tolist(),
            "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
            "cv_auc": float(cv), "train_auc": float(auc), "n": int(len(y)),
            "n_robust": int(y.sum()), "train_pred_true_rate": float(pred.mean()),
            "n_pca": int(pca.n_components_), "source": fn,
            "thr": thr, "thr_rule": "train 점수 중앙값 (모델별 고정)",
            "train_true_rate_thr0": float(pred0.mean()),
        }
        say("")

    say("## 가중치 없는 모델 → 텍스트 폴백 (h15C: log 단어수 단독, 임계 train 고정)")
    say("   Qwen/Qwen3.5-4B (라벨 0개) · openai/gpt-oss-120b · 미지 model_id")
    say("")
    say("## 읽는 법 (중요)")
    say("   내부 CV AUC 는 **train 추정치**이고 소수 클래스가 3~10건이라 신뢰구간이 매우 넓다.")
    say("   #DS-RELABEL 의 창 2048 모델평균 0.872(보충판)는 **폴드 내부에서 층까지 고른** 값이고,")
    say("   여기는 층을 L//2 로 고정했으므로 그 숫자와 같지 않다. 이 표는 '가중치가 비어 있지")
    say("   않다' 를 확인하는 용도이며, 판정은 리더보드가 한다.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(art), encoding="utf-8")
    Path(a.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    say("")
    say(f"→ {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB) · 모델 {len(art['per_model'])}개")
    say(f"→ {a.report}")
    Path(a.report).write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
