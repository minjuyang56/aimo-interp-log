"""compass 제출물 로컬 스모크 (지휘 #34-4): 가장 큰 모델 DeepSeek-8B 3문제, 오프라인·순수 bool·시간 측정.

로컬 GPU는 8GB라 bf16(16.4GB)이 안 들어가므로 COMPASS_QUANT=4bit 로 측정한다(실전 워커는 bf16·96GB).
네트워크 차단 확인: HF_HUB_OFFLINE=1 · HF_DATASETS_OFFLINE=1 로 두고 실행 — 호출이 있으면 예외가 난다.

사용:  .venv/Scripts/python.exe analysis/compass_smoke.py [--model ...] [--n 3]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("COMPASS_QUANT", "4bit")
os.environ.setdefault("COMPASS_BATCH", "3")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "solutions" / "compass"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(ROOT / "experiments" / "results" / "train-main-v2.parquet")
    probs = [p.strip() for p, _ in df.groupby("problem", sort=False)][:a.n]

    import solution
    t0 = time.time()
    out = solution.are_robust(a.model, reasoning_effort="default", problems=probs)
    el = time.time() - t0

    print(f"\n=== 결과 ===")
    print(f"반환: {out}")
    print(f"타입 검사 type(x) is bool: {[type(x) is bool for x in out]}  (채점기 조건)")
    print(f"길이 일치: {len(out) == len(probs)}")
    print(f"총 시간: {el:.1f}s · 문제 {len(probs)}개 · 문제당 {el / max(1, len(probs)):.1f}s")
    print(f"\n2인자 호출(옛 경로): {solution.are_robust(a.model, probs[:1])}")
    print(f"미지 모델: {solution.are_robust('foo/bar', reasoning_effort='low', problems=probs[:2])}")
    print(f"빈 목록: {solution.are_robust(a.model, reasoning_effort='default', problems=[])}")
    assert all(type(x) is bool for x in out) and len(out) == len(probs)
    print("\n스모크 통과")


if __name__ == "__main__":
    main()
