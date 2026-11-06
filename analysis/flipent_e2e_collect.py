"""#FLIPENT-ZIP e2e 증거 수집 — `.log` 는 `.gitignore:4`(`experiments/results/*.log*`) 로 커밋되지
않으므로, 로컬 e2e 가 찍은 **계측 줄만 기계적으로 뽑아** 커밋되는 `.txt` 로 남긴다.

손으로 옮겨 적지 않는다(이 저장소에서 숫자를 베껴 적다 어긋난 적이 여러 번 있다) — grep 과 같은
규칙으로 로그에서 긁어 온다. 로그가 지워지면 이 파일이 유일한 기록이 되므로 **출처 파일명과
크기도 같이 적는다.**

사용: .venv/Scripts/python.exe analysis/flipent_e2e_collect.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments/results"
OUT = RES / "flipent_smoke_e2e.txt"
KEEP = re.compile(r"로드|생성 계획|ent_front|판정 |경로 |생성 경로|OOM|로드 실패|\[OK\]|\[FAIL\]|판정:")
LOGS = ["flipent_smoke_e2e.log",                               # 1차: 한 프로세스에 4모델
        "flipent_smoke_e2e_Skywork-OR1-Math-7B.log",           # 2차: 개별 프로세스
        "flipent_smoke_e2e_DeepSeek-R1-0528-Qwen3-8B.log"]
L = ["#FLIPENT-ZIP 로컬 e2e 계측 줄 (로그에서 기계적으로 추출 · 손으로 옮겨 적지 않았다)",
     "   추출 규칙: /%s/ 에 걸리는 줄만" % KEEP.pattern, ""]

for name in LOGS:
    p = RES / name
    if not p.is_file():
        L += ["## %s — **없음**" % name, ""]
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    hit = [l.rstrip() for l in txt.splitlines() if KEEP.search(l)]
    L += ["## %s (%d bytes · 걸린 줄 %d)" % (name, p.stat().st_size, len(hit))]
    L += ["   " + l for l in hit]
    L += [""]

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("저장 %s (%d bytes)" % (OUT, OUT.stat().st_size))
