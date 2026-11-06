"""#GEN-PROBE 보고서에서 **DeepSeek 제외** 모델평균을 다시 계산한다. GPU 0 · 탐색 표기.

왜: #TOK-AUDIT 으로 DeepSeek 라벨이 디토큰화 손상 위에서 집계됐음이 확정됐다(복원 불가).
그 팔을 뺀 재계산은 '결론이 손상에 의존하는가' 를 보는 로버스트니스 점검이지
사전등록 분석이 아니다 — 사전등록은 3모델 · 주창 512 다. 숫자는 보고서 파일에서 읽어
손으로 옮겨 적지 않는다(여러 번 베끼다 어긋난 적이 있다).

사용: python analysis/probe_gen_nods.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "experiments" / "results"
FILES = [("창 256", "probe_gen_report_w256.txt"),
         ("창 512 (주창)", "probe_gen_report.txt"),
         ("창 2048", "probe_gen_report_w2048.txt")]
BASE = [("기준선1 base_acc(k=8)", r"기준선1 base_acc\(k=8\): AUC ([\d.]+)"),
        ("기준선2 greedy 정답여부", r"기준선2 greedy 정답여부: AUC ([\d.]+)"),
        ("기준선3 greedy 토큰수", r"기준선3 greedy 토큰수: AUC ([\d.]+)")]
lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


def mean(v):
    return sum(v) / len(v)


out("#GEN-PROBE — DeepSeek 제외 재계산 (**탐색**: 사전등록은 3모델·주창 512)")
out("근거: #TOK-AUDIT — DeepSeek 라벨이 Ġ/Ċ 손상 위에서 집계됐고 텍스트가 없어 복원 불가.")
out()
for label, fn in FILES:
    txt = (R / fn).read_text(encoding="utf-8")
    blocks = txt.split("## ")
    got = {}
    for b in blocks:
        for m in ("Skywork", "Olmo", "DeepSeek"):
            if b.startswith(m + " ·"):
                auc = float(re.search(r"AUC \*\*([\d.]+)\*\*", b).group(1))
                nu = float(re.search(r"null 95% \[[\d.]+, ([\d.]+)\]", b).group(1))
                rz = float(re.search(r"길이잔차 AUC ([\d.]+) · null \[[\d.]+, ([\d.]+)\]", b).group(1))
                rzn = float(re.search(r"길이잔차 AUC [\d.]+ · null \[[\d.]+, ([\d.]+)\]", b).group(1))
                got[m] = (auc, nu, rz, rzn)
    keep = ["Skywork", "Olmo"]
    a = mean([got[m][0] for m in keep])
    n = mean([got[m][1] for m in keep])
    rz = mean([got[m][2] for m in keep])
    rzn = mean([got[m][3] for m in keep])
    a3 = mean([got[m][0] for m in got])
    n3 = mean([got[m][1] for m in got])
    bs = {}
    for bl, pat in BASE:
        vals = [float(v) for v in re.findall(pat, txt)]
        bs[bl] = (mean(vals), mean(vals[:2]) if len(vals) >= 2 else float("nan"))
    out(f"## {label}")
    out(f"   3모델(사전등록)  AUC {a3:.3f} vs null상단 {n3:.2f} → "
        f"{'넘음' if a3 > n3 else '미달'}")
    out(f"   DS 제외 2모델    AUC {a:.3f} vs null상단 {n:.2f} → "
        f"{'넘음' if a > n else '미달'}   (길이잔차 {rz:.3f} vs {rzn:.2f})")
    out(f"   모델별: " + " · ".join(f"{m} {got[m][0]:.3f}[{got[m][1]:.2f}]" for m in got))
    out(f"   기준선 DS제외: " + " · ".join(f"{k} {v[1]:.3f}" for k, v in bs.items()))
    out(f"   → DS 제외 기준2(기준선 전부보다 높음): "
        f"{'통과' if all(a > v[1] for v in bs.values()) else '미달'}")
    out()
out("## 읽는 법")
out("   주창 512(사전등록 주지표)는 DS 를 빼도 미달이다 → #GEN-PROBE '신호 미입증' 은 손상과 무관하다.")
out("   창 256·2048 에서는 DS 를 빼면 기준1 을 넘는다. 그러나 창 3개 × 모델집합 2개를 본 뒤의")
out("   조합이고 모델집합 선택이 감사 결과를 본 뒤에 생겼으므로 **주장하지 않는다**.")
out("   Skywork 는 spurious 4건이라 null 폭이 0.6 이상이다 — #77 의 0.885 와 같은 구조다.")
(R / "probe_gen_nods.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\n저장 {R / 'probe_gen_nods.txt'}")
