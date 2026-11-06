"""4bit nf4 vs bf16 **전정밀도** — 특징이 얼마나 다르고, 그 차이가 판정을 바꾸나. GPU 0(추출은 별도). (지휘 #VAR-CHK)

## 왜
#77 의 모든 특징은 `probe_pilot.py --quant 4bit`(nf4, compute bf16)로 뽑았다. 감사 (1)에서 그 사실을
적었고 지시 (b)가 'bf16 전정밀도로 재추출' 을 탐색 항목으로 요구했다. **추측으로 답하지 않는다**:
같은 모델·같은 라벨·같은 코드로 정밀도만 바꿔 뽑은 두 특징을 맞댄다.

두 층위로 본다 — 둘이 다른 질문이기 때문이다.
  1. **특징 자체가 다른가** — 층별 코사인·상대 L2 오차. 1 에 가까우면 양자화가 표현을 거의 안 건드린다.
  2. **판정이 달라지나** — 같은 nested CV 로 AUC 를 내서 비교. 1 이 높아도 2 가 다를 수 있다
     (신호가 작은 잔차에 있을 수 있으므로). 반대로 1 이 낮아도 2 가 같으면 양자화는 무해하다.

bf16 이 로컬에서 가능했던 이유: 8 GB VRAM 에 7B bf16 은 안 들어가지만 `device_map="auto"` 가
남는 층을 RAM 으로 내린다(실측 Skywork 30문제 467초, $0). RunPod 를 쓰기 전에 답이 나온다.

사용: python analysis/probe_precision_check.py --a <4bit npz> --b <bf16 npz>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from probe_final_eval import auc_of, bacc_of, make_plan, nested_scores, resid  # noqa: E402

OUT = ROOT / "experiments" / "results" / "probe_precision_check.txt"

lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="4bit npz")
    ap.add_argument("--b", required=True, help="bf16 npz")
    ap.add_argument("--label", default="Skywork k=8")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    za, zb = np.load(a.a, allow_pickle=True), np.load(a.b, allow_pickle=True)
    layers = [int(x) for x in za["layers"]]
    #  행 순서를 가정하지 않는다 — pidx 로 맞춘다(같은 라벨 파일이라도 순서가 같다고 믿지 않는다).
    pa, pb = list(za["pidx"]), list(zb["pidx"])
    common = [p for p in pa if p in set(pb)]
    ia = [pa.index(p) for p in common]
    ib = [pb.index(p) for p in common]
    y = np.asarray(za["y"], int)[ia]
    words = np.asarray(za["words"], float)[ia]
    out(f"정밀도 비교 — {a.label} · 공통 {len(common)}행 (4bit {len(pa)} · bf16 {len(pb)}) · "
        f"robust {int(y.sum())} / spurious {int((1 - y).sum())}")
    out("**탐색 표기.** #77 판정(미입증)을 고치지 않는다.")
    out()
    if not np.array_equal(np.asarray(za["y"], int)[ia], np.asarray(zb["y"], int)[ib]):
        raise SystemExit("두 파일의 라벨이 다르다 — 비교 불가")

    # ---- 1) 특징 자체
    out("## 1. 특징이 얼마나 다른가 (행별로 재고 중앙값·최솟값)")
    out(f"   {'층':>4}{'코사인 중앙':>12}{'코사인 최소':>12}{'상대L2 중앙':>12}{'노름비 중앙':>12}")
    for l in layers:
        A, B = za[f"h{l}"][ia].astype(np.float64), zb[f"h{l}"][ib].astype(np.float64)
        cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
        rel = np.linalg.norm(A - B, axis=1) / np.linalg.norm(B, axis=1)
        nr = np.linalg.norm(A, axis=1) / np.linalg.norm(B, axis=1)
        out(f"   {l:>4}{np.median(cos):>12.4f}{cos.min():>12.4f}{np.median(rel):>12.4f}{np.median(nr):>12.4f}")
    out()
    #  행 사이 구조가 보존되나 — 분류기가 보는 것은 절대 위치가 아니라 **행 간 차이**다.
    out("## 2. 행 사이 구조가 보존되나 (분류기가 실제로 쓰는 것)")
    out("   같은 층에서 행쌍 거리 행렬의 상관 — 1 이면 기하가 그대로다")
    for l in layers:
        A, B = za[f"h{l}"][ia].astype(np.float64), zb[f"h{l}"][ib].astype(np.float64)
        da = np.linalg.norm(A[:, None] - A[None], axis=2)
        db = np.linalg.norm(B[:, None] - B[None], axis=2)
        iu = np.triu_indices(len(A), 1)
        out(f"   층 {l:>3}: 피어슨 {np.corrcoef(da[iu], db[iu])[0, 1]:.4f} · "
            f"스피어만 {np.corrcoef(np.argsort(np.argsort(da[iu])), np.argsort(np.argsort(db[iu])))[0, 1]:.4f}")
    out()

    # ---- 3) 판정이 달라지나
    out("## 3. 판정이 달라지나 (같은 nested CV · 층·C 폴드 내부 선택 · PCA 24)")
    res = {}
    for name, z, idx in (("4bit", za, ia), ("bf16", zb, ib)):
        fs = {l: z[f"h{l}"][idx] for l in layers}
        plan = make_plan(fs, y, layers, a.folds, 0)
        if plan is None:
            out(f"   {name}: 폴드 생성 불가")
            continue
        s, picks = nested_scores(y, layers, plan)
        res[name] = (auc_of(y, s), auc_of(y, resid(s, np.log(np.maximum(1, words)))),
                     bacc_of(y, s), [p[0] for p in picks], s)
        out(f"   {name:<5} AUC **{res[name][0]:.3f}** · 길이잔차 {res[name][1]:.3f} · "
            f"bacc {res[name][2]:.3f} · 고른 층 {res[name][3]}")
    if len(res) == 2:
        d = res["bf16"][0] - res["4bit"][0]
        rho = np.corrcoef(np.argsort(np.argsort(res["4bit"][4])),
                          np.argsort(np.argsort(res["bf16"][4])))[0, 1]
        out(f"   차이 bf16 − 4bit = **{d:+.3f}** · 폴드밖 점수 순위상관 **{rho:+.3f}**")
    out()
    out("## 읽는 법")
    out("   1·2 가 1 에 가깝고 3 의 차이가 설계·부분집합 흔들림(±0.1~0.2)보다 작으면,")
    out("   **4bit 는 #77 이 신호를 못 본 이유가 아니다** → bf16 재추출에 돈을 쓸 근거가 없다.")
    out("   반대로 3 의 차이가 크면 다른 두 모델도 bf16 으로 다시 뽑아야 한다(RunPod 비용 보고 후).")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {OUT}")


if __name__ == "__main__":
    main()
