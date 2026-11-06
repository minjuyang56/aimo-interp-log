"""#GEN-PROBE 보충판 — 중단 지점부터 이어서 (2026-10-12).

dsrelabel_supp.py 가 off40 특징 재추출(32행)까지 끝낸 뒤 셸이 끊겨 죽었다
(로그 `dsrelabel_supp_run.log`: off80 _rep 추출 20/32 에서 중단, 평가 0건).
그 시점의 디스크 상태:
  *_gen{win}.npz        = **보충판 off40**(32행) — 이미 저장됨
  *_gen{win}_isect.npz  = 사전등록 교집합판 off40(21행) · _rep_isect = off80(18행)
  *_gen{win}_rep.npz    = 아직 교집합판(18행) — 재추출이 안 끝났다
그래서 이 스크립트는 **off40 추출을 다시 하지 않고** off80 추출부터 이어서 간다.
병합 파일(`*_greedy_merged.json`)도 이미 검증·저장돼 있어 다시 만들지 않는다.
끝에서 보충판을 `_supp` 로 떠 두고 기본 파일명을 교집합판으로 되돌리는 것은 원본과 같다.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
RES = ROOT / "experiments" / "results"
P = RES / "probe"
G = RES / "runpod" / "genprobe"
A = "DeepSeek-R1-0528-Qwen3-8B"
DS = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MERGED = G / f"{A}_greedy_merged.json"
WINS = (512, 256, 2048)
SPECIAL = re.compile(r"<[｜|][^>]{0,40}[｜|]>")


def run(cmd, why):
    print(f"\n=== {why}\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(f"!! 실패 ({why}) rc={r.returncode}")


def copy(src, dst):
    shutil.copy2(src, dst)
    print(f"복사 {src.name} → {dst.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off80", required=True)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()

    # 이어받는 전제를 계측으로 확인한다 — 틀리면 평가가 엉뚱한 묶음에서 돈다
    rows = json.loads(MERGED.read_text(encoding="utf-8"))
    mk = sum(r["greedy_text"].count("Ġ") + r["greedy_text"].count("Ċ") for r in rows)
    sp = sum(1 for r in rows if SPECIAL.search(r["greedy_text"]))
    print(f"병합 파일 {len(rows)}행 · 특수토큰 {sp} · Ġ/Ċ 마커 {mk}", flush=True)
    if mk or sp or len(rows) != 66:
        raise SystemExit("!! 병합 파일이 기대와 다르다")
    import numpy as np
    n0 = len(np.load(P / f"{A}_gen512.npz", allow_pickle=True)["y"])
    ni = len(np.load(P / f"{A}_gen512_isect.npz", allow_pickle=True)["y"])
    print(f"기본 gen512 {n0}행 · 교집합판 {ni}행", flush=True)
    if not (n0 == 32 and ni == 21):
        raise SystemExit("!! 기본 파일이 보충판 off40(32행) 이 아니다 — 수동 확인 필요")

    if not a.skip_extract:
        run([PY, "experiments/probe_gen.py", "--model", DS, "--greedy", MERGED,
             "--labels", a.off80, "--sets", "ds_off80", "--suffix", "_rep"],
            "DS 민감도 묶음 특징 재추출 (보충 포함) — 이어서")

    for w in WINS:
        o = RES / f"probe_gen_report_supp{'' if w == 512 else f'_w{w}'}.txt"
        run([PY, "analysis/probe_gen_eval.py", "--win", w, "--perms", a.perms,
             "--jobs", a.jobs, "--out", o], f"보충판 재평가 창 {w}")
    run([PY, "analysis/probe_gen_eval.py", "--win", 512, "--perms", a.perms, "--jobs", a.jobs,
         "--suffix", "_rep", "--out", RES / "probe_gen_report_supp_rep.txt"],
        "보충판 off80 민감도 (창 512)")

    for w in WINS:                       # 보충판을 접미사로 남긴다
        for extra in ("", "_rep"):
            src = P / f"{A}_gen{w}{extra}.npz"
            if src.is_file():
                copy(src, P / f"{A}_gen{w}{extra}_supp.npz")
    for w in WINS:                       # 기본 파일명은 사전등록 교집합판으로 되돌린다
        for extra in ("", "_rep"):
            src = P / f"{A}_gen{w}{extra}_isect.npz"
            if src.is_file():
                copy(src, P / f"{A}_gen{w}{extra}.npz")
    print("\n보충판 보고서: probe_gen_report_supp{,_w256,_w2048,_rep}.txt", flush=True)
    print("기본 npz 는 교집합판(사전등록 주판정)으로 되돌렸다 · 보충판은 *_supp.npz", flush=True)


if __name__ == "__main__":
    main()
