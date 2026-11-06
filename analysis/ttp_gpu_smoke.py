"""#TT-PERTURB 로컬 GPU 스모크 — 실제 생성 경로를 작은 cap 으로 전부 밟는다. 비용 $0.

`ast.parse` 를 통과하고도 파드에서 1분에 죽은 적이 있다(#SAMP-FEAT, $0.43). 그래서 **진짜 모델**로
`cot_then_answer_multi` → 체크포인트 절단 → 강제 즉답 → `score_one` → `decide` 까지 밟는다.
로컬 8 GB 라 Qwen3.5-4B 를 4bit 로 열고 cap 을 아주 작게 둔다 — **측정값이 아니라 경로 검사**다.

검사:
1. 체크포인트별 즉답이 전부 나오는가(= 강제 즉답이 작동하는가) · 정수 답 추출률
2. ckpt 0(CoT 없음)과 큰 ckpt 의 답이 **다를 수 있는가**(같기만 하면 절단이 안 먹은 것)
3. `token_slots` 가 양수를 주는가 · `plan` 이 cap 을 고르는가
4. `decide` 가 끝까지 돌아 순수 bool 을 주는가
5. 재호출 캐시(모델 재로드 안 함)

사용: .venv/Scripts/python.exe analysis/ttp_gpu_smoke.py
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solutions/ttp"))
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("TTP_QUANT", "4bit")
REPORT = ROOT / "experiments/results/ttp_gpu_smoke.txt"
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    import ttp_lib as T
    import solution as S
    from h21_uncertainty import problems
    txt = [r["problem"] for r in problems()][:2]
    MID = os.environ.get("TTP_SMOKE_MODEL", "Qwen/Qwen3.5-4B")
    CAP = int(os.environ.get("TTP_SMOKE_CAP", 48))
    CKPTS = [0, 16, CAP]
    ok = True
    out(f"#TT-PERTURB GPU 스모크 — {MID} · 4bit · cap {CAP} · 체크포인트 {CKPTS}")

    t0 = time.time()
    tok, model = T.get_model(MID)
    if model is None:
        out("!! 모델 로드 실패 — 경로 검사 불가")
        REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
        return False
    out(f"   로드 {time.time() - t0:.0f}s · 토큰당 KV {T.kv_kb_per_token(model):.0f} KB")

    # 3. 슬롯·계획
    sl = T.token_slots(model, CAP)
    g = sl >= 1
    ok &= g
    out(f"3. token_slots({CAP}) = {sl}  {'OK' if g else '**FAIL**'}")
    pl = T.plan(6, 600.0, model)
    out(f"3. plan(6시퀀스, 600s) = {pl}  {'OK' if pl else '(폴백 — 예산 부족 판정)'}")

    # 1·2. 생성 경로
    flat = [txt[0]] + [v for _, v in T.perturbations(txt[0])][:2]
    t1 = time.time()
    cots, res = T.cot_then_answer_multi(tok, model, flat, CAP, max(1, min(sl, 3)),
                                        deadline=time.time() + 1200, ckpts=CKPTS)
    out(f"1. 생성 {len(flat)} 시퀀스 · {time.time() - t1:.0f}s · CoT 길이 {[len(c) for c in cots]}")
    for ck in CKPTS:
        vals = [v for _, v in res[ck]]
        texts = [a[:20].replace("\n", " ") for a, _ in res[ck]]
        got = sum(1 for v in vals if v is not None)
        out(f"   ckpt {ck:>4}: 추출 {got}/{len(vals)} · 답 {vals} · 즉답원문 {texts}")
    g = all(len(res[ck]) == len(flat) for ck in CKPTS)
    ok &= g
    out(f"1. 체크포인트별 반환 길이 일치 : {g}  {'OK' if g else '**FAIL**'}")
    diff = any(res[CKPTS[0]][i][1] != res[CKPTS[-1]][i][1] for i in range(len(flat)))
    out(f"2. ckpt {CKPTS[0]} vs {CKPTS[-1]} 답이 갈리는 행 있음 : {diff} "
        f"(False 면 절단이 안 먹었을 수 있다 — cap 이 너무 작아 CoT 가 비었는지 확인)")

    # score_one
    o = res[CAP][0][1]
    ps = [res[CAP][i][1] for i in range(1, len(flat))]
    out(f"   score_one(원본 {o}, 섭동 {ps}) = {T.score_one(o, ps)}  (섭동 2개라 K_MIN 미달 → None 정상)")

    # 4·5. decide
    t2 = time.time()
    T.SOFT = 1e9
    r = S.are_robust(MID, "default", txt)
    g = isinstance(r, list) and len(r) == len(txt) and all(type(x) is bool for x in r)
    ok &= g
    out(f"4. are_robust → {r} · 순수bool {g} · {time.time() - t2:.0f}s  {'OK' if g else '**FAIL**'}")
    t3 = time.time()
    _ = T.get_model(MID)
    g = time.time() - t3 < 1.0
    ok &= g
    out(f"5. 재호출 캐시 {time.time() - t3:.2f}s (<1s)  {'OK' if g else '**FAIL**'}")

    out()
    out("## " + ("**경로 전부 통과**" if ok else "**실패 — 파드를 띄우지 말 것**"))
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
