"""sub2048 의 **모델별 칸막이를 빼 보는** 제거 실험 (지휘 2026-11-06 · $0 · 저장된 공식 33 특징만).

sub2048 에는 '칸막이' 가 두 겹 있다. 어느 쪽을 빼는지가 다른 실험이므로 **둘 다** 낸다:

  AB1 **계측기 칸막이 제거** — 모델마다 자기 프로브를 쓰는 대신 **한 모델의 프로브로 33문제를
       전부 채점**하고 그 문제단위 예측을 모든 모델에 그대로 뿌린다(= 공용 계측기, h20f 와 같은 꼴).
       이 경우 모델평균 정확도 = 그 하나의 정확도다(모든 모델이 같은 답을 받으므로).
       계측기를 어느 모델로 두느냐가 자유 파라미터라 **세 모델 전부** 적는다(고르지 않는다).

  AB2 **임계 칸막이 제거** — 프로브는 모델별로 두고 **임계만 하나로 공유**한다.
       공유 임계 = 출하된 세 임계의 평균. 테스트 라벨·테스트 점수를 쓰지 않으므로
       '칸막이만 뺀' 변화가 되고, 중앙값 분할처럼 균형 가정에 기대지도 않는다.

**이건 사전등록이 아니라 EXPLORATORY 다.** 공식 33 의 주지표를 이미 본 뒤에 만든 비교이고,
같은 테스트셋을 여러 번 본 축적도 그대로 있다. 리포트 표에도 그렇게 적는다.

기존 평가기에서 **import 로 가져온다**(공식을 다시 적으면 어긋난다):
  `org_sub2048_transfer`: ART · FEAT · MODELS · auc · probe_score
  `sub2048_plus_eval`:    pred_loo_prior · acc_of · band · out 규약

사용: .venv/Scripts/python.exe analysis/sub2048_ablate.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

from org_sub2048_transfer import ART, FEAT, MODELS, auc as auc_rob, probe_score
from sub2048_plus_eval import acc_of, band, pred_loo_prior

OUT = ROOT / "experiments/results/sub2048_ablate.txt"
PERMS = 1000
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    from h21_uncertainty import problems
    probs = problems()
    ptext = {r["problem"]: r for r in probs}
    art = json.loads(ART.read_text(encoding="utf-8"))

    #  모델별 점수·라벨·문제텍스트를 한 번에 읽는다
    S, Y, PT, THR = {}, {}, {}, {}
    for mid in MODELS:
        alias = mid.split("/")[-1].replace(".", "_")
        p = FEAT / f"{alias}_org33.npz"
        if not p.is_file():
            continue
        d = np.load(p, allow_pickle=True)
        am = art["per_model"][mid]
        if int(d["layer"]) != int(am["layer"]):
            raise SystemExit(f"층 불일치 {alias}: {int(d['layer'])} vs {am['layer']}")
        pt = [str(x) for x in d["problems"]]
        S[alias] = probe_score(am, d["X"])
        Y[alias] = np.array([ptext[t]["y"] for t in pt], int)
        PT[alias] = pt
        THR[alias] = float(am["thr"])

    aliases = list(S)
    out("sub2048 모델별 칸막이 제거 실험 — EXPLORATORY (사전등록 아님)")
    out(f"   공식 train-main-v2 문제단위 라벨 {len(probs)}문제 "
        f"(robust {sum(r['y'] for r in probs)} / spurious {len(probs)-sum(r['y'] for r in probs)})")
    out(f"   가중치 {ART.relative_to(ROOT).as_posix()} 그대로 · 창 {art['win']} · 재적합 없음 · 모델 {len(aliases)}종")
    out(f"   출하 임계: " + " · ".join(f"{a} {THR[a]:+.4f}" for a in aliases))
    out()

    #  문제 순서가 모델 간 같아야 '공용 계측기' 를 뿌릴 수 있다 — 계측으로 확인한다
    base = PT[aliases[0]]
    for a in aliases[1:]:
        if PT[a] != base:
            raise SystemExit(f"문제 순서 불일치: {a}")
    y = Y[aliases[0]]
    out(f"게이트: 세 모델의 문제 순서·라벨 동일 — 확인 (n={len(y)})")

    #  길이 대조군과 상수 (같은 판정 절차로)
    lw = np.array([math.log(max(1, len(t.split()))) for t in base])
    rows = []

    # ── 기준 (칸막이 그대로) ─────────────────────────────────────────────
    accs, aucs, trs = [], [], []
    for a in aliases:
        pr = S[a] >= THR[a]
        accs.append(acc_of(pr, Y[a])); aucs.append(auc_rob(Y[a], S[a])); trs.append(float(np.mean(pr)))
    out()
    out("## 기준 — 칸막이 그대로 (모델별 프로브 + 모델별 출하 임계)")
    out(f"   모델평균 정확도 {np.mean(accs):.4f} · AUC {np.mean(aucs):.4f} · True 비율 {np.mean(trs):.3f}")
    out("   " + " · ".join(f"{a} {x:.4f}" for a, x in zip(aliases, accs)))
    base_acc_permodel = float(np.mean(accs))

    # ── AB1 계측기 칸막이 제거 ──────────────────────────────────────────
    out()
    out("## AB1 — 계측기 칸막이 제거 (한 모델 프로브로 전부 채점 → 모든 모델에 같은 답)")
    ab1 = {}
    for inst in aliases:
        pr = S[inst] >= THR[inst]
        acc = acc_of(pr, y)                       # 모든 모델이 같은 예측 → 모델평균 = 이 값
        up = band(lambda s, yy, p=pr: p, S[inst], y, perms=PERMS)[1]
        ab1[inst] = (acc, auc_rob(y, S[inst]), float(np.mean(pr)), up)
        out(f"   계측기 {inst:28s} 정확도 {acc:.4f} · AUC {auc_rob(y, S[inst]):.4f} "
            f"· True {np.mean(pr):.3f} · null상단 {up:.4f}")
    out(f"   → 계측기 선택에 따른 폭 {min(v[0] for v in ab1.values()):.4f}~{max(v[0] for v in ab1.values()):.4f}"
        f" (계측기 선택이 자유 파라미터다 — 고르지 않는다)")

    # ── AB2 임계 칸막이 제거 ────────────────────────────────────────────
    thr_shared = float(np.mean([THR[a] for a in aliases]))
    out()
    out(f"## AB2 — 임계 칸막이 제거 (프로브는 모델별, 임계 하나 = 출하 세 임계의 평균 {thr_shared:+.4f})")
    accs2, trs2, ups2 = [], [], []
    for a in aliases:
        pr = S[a] >= thr_shared
        accs2.append(acc_of(pr, Y[a])); trs2.append(float(np.mean(pr)))
        ups2.append(band(lambda s, yy, p=pr: p, S[a], Y[a], perms=PERMS)[1])
        out(f"   {a:28s} 정확도 {acc_of(pr, Y[a]):.4f} · True {np.mean(pr):.3f}")
    out(f"   모델평균 정확도 {np.mean(accs2):.4f} · True 비율 {np.mean(trs2):.3f} · null상단평균 {np.mean(ups2):.4f}")
    out("   (AUC 는 임계와 무관하므로 기준과 같다)")

    # ── 참조 기준선 ─────────────────────────────────────────────────────
    out()
    out("## 참조 기준선 (같은 절차)")
    accs3 = [acc_of(pred_loo_prior(S[a], Y[a]), Y[a]) for a in aliases]
    out(f"   모델별 프로브 + LOO 사전확률 교정   정확도 {np.mean(accs3):.4f}")
    acc_len = acc_of(pred_loo_prior(-lw, y), y)
    out(f"   길이 단독 h15C (LOO 교정)          정확도 {acc_len:.4f}")
    out(f"   상수 '전부 robust'                정확도 {np.mean(y):.4f}")

    out()
    out("## 읽는 법")
    out(f"   칸막이 그대로 {base_acc_permodel:.4f} vs 계측기 칸막이 제거 "
        f"{np.mean([v[0] for v in ab1.values()]):.4f}(계측기 평균) vs 임계 칸막이 제거 {np.mean(accs2):.4f}")
    out("   어느 칸도 상수·길이를 넘지 못하면 '칸막이가 점수의 원인이 아니다' 로 적는다.")
    out("   EXPLORATORY: 공식 33 주지표를 본 뒤 만든 비교다. 판정에 쓰지 않고 표에 그대로 적는다.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
