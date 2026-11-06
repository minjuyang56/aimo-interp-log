"""저장소 경로 해석 — 다른 analysis 스크립트가 공통으로 쓴다.

로컬 클론 위치가 다르면 환경변수로 덮어쓸 수 있다:
    AIMO_GETTING_STARTED=... AIMO_BASELINES=... python analysis/label_rule.py
"""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SIDE = _HERE.parent.parent  # aimo-interp-log/analysis -> aimo-interp-log -> C:\side

GETTING_STARTED = Path(os.environ.get("AIMO_GETTING_STARTED", _SIDE / "getting-started"))
BASELINES = Path(os.environ.get("AIMO_BASELINES", _SIDE / "baselines"))

CASES = GETTING_STARTED / "data" / "val-sample" / "input" / "cases.jsonl"
LABELS = GETTING_STARTED / "data" / "val-sample" / "reference" / "labels.jsonl"
METADATA = GETTING_STARTED / "data" / "val-sample" / "metadata.json"
FINAL_CSV = BASELINES / "data" / "math-robust-final.csv"
AGG_CSV = BASELINES / "data" / "math-robust-agg.csv"
DIST = GETTING_STARTED / "dist"


def require(*paths):
    """없는 파일이 있으면 무엇을 클론해야 하는지 알려주고 종료한다."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print("필요한 파일이 없습니다:")
        for p in missing:
            print("   ", p)
        print()
        print("클론:")
        print("    git clone https://github.com/aimo-interp/getting-started")
        print("    git clone https://github.com/aimo-interp/baselines")
        raise SystemExit(1)
