"""#R5-TRANSFER (a) 재료 집계 — 방향을 학습할 행이 실제로 몇 개이고 정답률 변이가 얼마인가. $0.

**포함 규칙을 코드에 박는다**(사전등록과 같은 내용):
* Ġ 디토큰화 손상 실행 제외 — DeepSeek 의 손상 slice 는 `base_acc` 평균이 약 0.50 이고
  깨끗한 재생성본은 약 0.85 다(#TOK-AUDIT 의 오판율 0.293 과 같은 크기의 갭). pod id 로 가른다.
* `i6hxd8z437eas4` 제외 — #TEMP-SENS 의 **T=1.0** 실행이라 디코딩이 다르다.
* 채택 여부와 **무관하게** screen 의 전 문제를 쓴다 — 그게 이 라운드의 요점이다(채택분만 쓰면
  `base_acc ≥ 0.5` 필터 때문에 정답률 변이가 선별로 사라진다: 실측 표준편차 0.07~0.17).

사용: .venv/Scripts/python.exe analysis/r5_rows.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WS = re.compile(r"\s+")

# 모델 → 쓰는 pod id (깨끗 · T=0.6)
INCLUDE = {
    "Skywork-OR1-Math-7B": ["pj7vh1qllei51i"],
    "Olmo-3-7B-Think": ["sprwmaodvz095s", "ofclt769pgvn6u", "nbgd17ti5ydns8", "daxtmpmss5lf1a"],
    "DeepSeek-R1-0528-Qwen3-8B": ["mbpc29h3v7um64", "9svqsw8dzrtjsv"],
}
EXCLUDE_WHY = {
    "67elv2je7wzefw": "DeepSeek Ġ 손상 (base_acc 0.497)",
    "1f7jul0dgk7o9s": "DeepSeek Ġ 손상 (0.512)",
    "luvw8yu3mkxsza": "DeepSeek Ġ 손상 (0.497)",
    "x04ghnc52b6dll": "DeepSeek Ġ 손상 k=4 (0.544)",
    "i6hxd8z437eas4": "Olmo T=1.0 (디코딩 다름)",
}
OUT = ROOT / "experiments/results/r5_rows.json"
REPORT = ROOT / "experiments/results/r5_rows.txt"
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def norm(t):
    return WS.sub(" ", str(t)).strip()


def main():
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    from h21_uncertainty import problems
    off33 = {norm(r["problem"]) for r in problems()}

    out("#R5-TRANSFER (a) 재료 — screen 전 문제(채택·탈락 모두)로 정답률 방향을 학습할 행")
    out(f"   제외한 pod: " + " · ".join(f"{k}({v})" for k, v in EXCLUDE_WHY.items()))
    out()
    store = {}
    out(f"   {'모델':<28}{'slice':<34}{'행':>4}{'k':>3}{'평균':>7}{'sd':>7}")
    for model, pods in INCLUDE.items():
        seen, rows = {}, []
        for pod in pods:
            for f in sorted((ROOT / "experiments/results/runpod").rglob(f"{pod}/out/*_screen.json")):
                o = json.loads(f.read_text(encoding="utf-8"))
                rs = o if isinstance(o, list) else o.get("rows", [])
                rs = [r for r in rs if model.split("/")[-1] in f.name and "base_acc" in r]
                if not rs:
                    continue
                ba = np.array([r["base_acc"] for r in rs], float)
                # base_acc 격자로 k 를 추정한다 (1/8 단위면 k=8, 1/4 면 k=4)
                g = sorted({round(v * 8) % 2 for v in ba})
                k = 8 if g != [0] else 4
                out(f"   {model[:26]:<28}{(pod + ' ' + f.name[:20]):<34}{len(rs):>4}{k:>3}"
                    f"{ba.mean():>7.3f}{ba.std():>7.3f}")
                for r in rs:
                    t = norm(r["problem"])
                    if t in off33:          # 누출 방지 — 설계상 0 이어야 한다
                        continue
                    if t in seen:           # 같은 문제가 여러 slice 에 나오면 첫 것만
                        continue
                    seen[t] = 1
                    rows.append(dict(problem=r["problem"], base_acc=float(r["base_acc"]),
                                     pidx=r.get("pidx"), pod=pod, k=k))
        ba = np.array([r["base_acc"] for r in rows], float)
        store[model] = rows
        out(f"   {model[:26]:<28}{'**고유 문제 합계**':<34}{len(rows):>4}{'':>3}"
            f"{ba.mean():>7.3f}{ba.std():>7.3f}")
        out(f"      base_acc 분포 {dict(sorted(Counter(np.round(ba, 3).tolist()).items()))}")
        out(f"      0 인 문제 {int((ba == 0).sum())} · 0.5 미만 {int((ba < 0.5).sum())} · "
            f"서로 다른 값 {len(set(np.round(ba, 4)))} · 공식 33 과 겹침 **0**(규칙상 제외)")
        out()
    OUT.write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    out(f"   저장 {OUT.relative_to(ROOT).as_posix()}")
    out()
    out("   비교: **채택분만** 쓸 때(특징이 이미 있는 행) 표준편차는 Skywork 0.120 · Olmo 0.167 ·")
    out("   DeepSeek 0.073 이고 DeepSeek 의 greedy 정답은 32/32 로 **변이가 0** 이었다.")
    out("   screen 전 문제를 쓰면 행도 늘고 변이도 커진다 — 그게 (a) 를 실행 가능하게 만든다.")
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n저장 {REPORT}")


if __name__ == "__main__":
    main()
