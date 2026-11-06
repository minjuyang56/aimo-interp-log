"""ttp_eval / ttp_r2_eval 의 지표·LOO·null 이 **실제로 신호를 가르는지** 합성 데이터로 검사한다."""
import sys, json, tempfile, pathlib
import numpy as np
ROOT = pathlib.Path(r"C:\side\aimo-interp-log")
sys.path.insert(0, str(ROOT/"analysis")); sys.path.insert(0, str(ROOT/"experiments"))
from ttp_eval import loo_acc, auc, null_band, thr_prior
from ttp_r2_eval import majority, r2_scores
from h21_uncertainty import problems
y = np.array([r["y"] for r in problems()], int); n = len(y)
rng = np.random.default_rng(0)

print(f"라벨 robust {y.sum()}/{n} · 상수 {y.mean():.4f}")
# 1) 완벽한 신호: spur = 1-y (robust=0, spurious=1)
perfect = 1.0 - y + rng.normal(0, 1e-6, n)
a1,_ = loo_acc(perfect, y); print(f"1. 완벽 신호  정확도 {a1:.4f} · AUC {auc(perfect,y):.4f}  (기대 1.0/1.0)")
# 2) 무정보: 상수
const = np.zeros(n)
a2,_ = loo_acc(const, y); print(f"2. 상수 점수  정확도 {a2:.4f} · AUC {auc(const,y):.4f}  (동점 → 입력순 상위 k)")
# 3) 난수
a3 = [loo_acc(rng.random(n), y)[0] for _ in range(200)]
print(f"3. 난수 200회 정확도 중앙 {np.median(a3):.4f} · 95% [{np.percentile(a3,2.5):.3f},{np.percentile(a3,97.5):.3f}]")
# 4) null 밴드가 난수 분포와 맞는가 (점수 고정·라벨 셔플)
s = rng.random(n); (la,ha),(lu,hu) = null_band(s, y, 400, 1)
print(f"4. null(라벨셔플) 정확도 95% [{la:.3f},{ha:.3f}] · AUC [{lu:.3f},{hu:.3f}]  (난수와 비슷해야 한다)")
# 5) 거친 격자(R2 의 N=3 해상도 모사): 점수가 네 칸뿐일 때
coarse = np.round((1.0-y) * rng.choice([0,1/3,2/3,1], n, p=[.1,.2,.3,.4]), 4)
a5,_ = loo_acc(coarse, y); print(f"5. 4칸 격자   정확도 {a5:.4f} · 서로 다른 값 {len(set(coarse))}")
# 6) majority 결정성·동점 처리
print(f"6. majority([1,1,2])={majority([1,1,2])} · ([1,2,3],fallback=2)={majority([1,2,3],2)} · "
      f"([1,2,3])={majority([1,2,3])} · ([None,None])={majority([None,None])}")
# 7) r2_scores: 합성 samp/rows
samp, rows = [], []
for i in range(n):
    gold = 100+i
    rows.append(dict(pidx=i, kind="orig", val={"512": gold}))
    for s_ in range(3):
        samp.append(dict(pidx=i, kind="orig", sdx=s_, val={"512": gold}))
        for t in ("rename","rename2","typos0","distract0"):
            # robust 문제는 전부 gold 유지, spurious 는 rename 에서 깨진다
            v = gold if (y[i]==1 or t!="rename") else gold+1
            samp.append(dict(pidx=i, kind=t, sdx=s_, val={"512": v}))
sc = r2_scores(samp, rows, "512")
md = np.array([sc[i]["md"] for i in range(n)], float)
a7,_ = loo_acc(md, y)
print(f"7. r2_scores 합성(완벽) 정확도 {a7:.4f} · AUC {auc(md,y):.4f} · md 값 {sorted(set(md))}")
# 8) 같은 합성에서 라벨만 섞으면 떨어져야 한다
ys = rng.permutation(y); a8,_ = loo_acc(md, ys)
print(f"8. 같은 점수 · 라벨 셔플 정확도 {a8:.4f}  (7 보다 많이 낮아야 검사가 유효하다)")
