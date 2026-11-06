"""라벨 있는 137문제마다 CoT를 생성하며 hidden state 경로를 기록하고 기하 특징을 저장한다.

trajectory 프로브 학습의 재료. 문제마다 생성 1회(max_new_tokens) — **GPU 전용**.
CPU(0.26 tok/s)에서는 1024토큰 × 137문제 ≈ 6일. 로컬은 --limit 1 --max-new-tokens 32 로
파이프라인이 도는지만 본다.

저장: results/traj_<모델>_L<층>.npz
    X      (문제수, 특징수)  float32   traj_lib.FEATURES 순서, 로그 변환 후
    y      (문제수,)         bool
    pid    (문제수,)         str
    ntok   (문제수,)         int       실제 생성 길이
    raw    dict[str, dict]            원본 특징값 (해석용)

사용:
  uv run python experiments/extract_trajectory.py --layer 20 --max-new-tokens 1024
  uv run python experiments/extract_trajectory.py --limit 1 --max-new-tokens 32     # 로컬 확인
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE.parent / "solutions" / "trajectory"))
from _paths import AGG_CSV, require  # noqa: E402
import traj_lib  # noqa: E402
from extract_hidden import DEFAULT_MODEL, alias, load_rows  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    require(AGG_CSV)

    rows = load_rows(args.limit)
    out = HERE / "results" / f"traj_{alias(args.model)}_L{args.layer}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} layer={args.layer} max_new_tokens={args.max_new_tokens} 문제 {len(rows)}개", flush=True)

    t0 = time.time()
    tok, model = traj_lib.load(args.model)
    print(f"load {time.time() - t0:.1f}s  device={model.device}", flush=True)

    X, y, pid, ntok, raw = [], [], [], [], {}
    for i, r in enumerate(rows, 1):
        t1 = time.time()
        P = traj_lib.trajectory(tok, model, r["original_problem"], args.layer, args.max_new_tokens)
        feat = traj_lib.geometry(P)
        X.append(traj_lib.to_vector(feat))
        y.append(r["model_is_robust"].strip().lower() == "true")
        pid.append(r["problem_id"])
        ntok.append(P.shape[0])
        raw[r["problem_id"]] = feat
        print(f"  [{i:3d}/{len(rows)}] {r['problem_id']}  T={P.shape[0]:5d}  {time.time() - t1:6.1f}s"
              f"  tort={feat['tortuosity']:.2f} early={feat['early_conv']:.2f} turn={feat['mean_turn']:.2f}"
              f"  robust={y[-1]}", flush=True)
        if i % 5 == 0:
            np.savez_compressed(out, X=np.stack(X), y=np.array(y), pid=np.array(pid), ntok=np.array(ntok),
                                raw=json.dumps(raw), model=args.model, layer=args.layer,
                                max_new_tokens=args.max_new_tokens)
    np.savez_compressed(out, X=np.stack(X), y=np.array(y), pid=np.array(pid), ntok=np.array(ntok),
                        raw=json.dumps(raw), model=args.model, layer=args.layer,
                        max_new_tokens=args.max_new_tokens)
    print(f"\n저장 {out}  X{np.stack(X).shape}  총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
