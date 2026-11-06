"""#GEN-PROBE greedy 커버리지 보충 — 병합 → 재추출 → 재평가. (지휘 승인, 2026-10-12)

## 왜 필요했나
#DS-RELABEL 로 선별이 깨끗해지자 DeepSeek 채택이 24 → 35 로 늘어 **새로 라벨된 pidx 25개**가
생겼다. 그런데 greedy 풀이 텍스트는 옛 41 pidx 분만 있어서 #GEN-PROBE 의 DS 팔이
off40 32행 → **21행(robust 20 / spurious 1)** 으로 쪼그라들었고, 소수 클래스가 1 이라
폴드가 아예 안 만들어졌다(= 판정 불가). 보충 생성($0.71, pod 8scifmo80m6rx6)으로 그 25행을
채웠다.

## 두 결과를 **둘 다** 남긴다 (지휘: 주판정은 사전등록대로)
* **교집합판** = 사전등록 주판정. greedy 가 이미 있던 행만. npz 는 `*_gen{win}_isect.npz` 로 떠 둔다.
* **보충판** = 이 스크립트가 만드는 것. `*_gen{win}_supp.npz` · 보고서 `probe_gen_report_supp*.txt`.
보충판은 **사후에 n 을 늘린 것**이므로 사전등록된 확증이 아니다. 판정 칸에 쓰지 않고 병기한다.

## 디코딩 경로가 섞이지 않았다는 확인 (이게 안 맞으면 병합 자체가 오염이다)
옛 41행은 손상본을 왕복 재디코드해 살린 것(`fix_deepseek_detok.py`, `tf.decode(ids)`),
새 25행은 생성 시점에 `decode_out`(같은 `PreTrainedTokenizerFast`, `skip_special_tokens=True`)
으로 받은 것이다. 둘 다 '같은 토큰 id 를 올바른 토크나이저로 디코드' 라서 같은 경로지만,
`skip_special_tokens` 기본값이 달라 꼬리에 특수토큰이 남을 수 있다 → **병합 전에 센다.**
실측(2026-10-12): 양쪽 모두 특수토큰 문자열 0건 · Ġ/Ċ 마커 0건.
보충 파드 로그는 `vLLM o.text 마커 174,824 · 쓰는 텍스트 마커 0` 이었다 — 가드를 안 넣었으면
보충 25행이 다시 손상돼 **같은 묶음 안에 두 디코딩이 섞였을 것**이다.

사용:
  .venv/Scripts/python.exe analysis/dsrelabel_supp.py --off40 <labels.json> --off80 <labels.json>
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
OLD = G / "oi8q4v8ia3ojq0" / "out" / f"{A}_greedy_fixed.json"
NEW = G / "8scifmo80m6rx6" / "out" / f"{A}_greedy.json"
MERGED = G / f"{A}_greedy_merged.json"
WINS = (512, 256, 2048)
SPECIAL = re.compile(r"<[｜|][^>]{0,40}[｜|]>")


def run(cmd, why):
    print(f"\n=== {why}\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(f"!! 실패 ({why}) rc={r.returncode}")


def merge():
    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))
    for tag, rows in (("옛(왕복 복원)", old), ("새(decode_out)", new)):
        sp = sum(1 for r in rows if SPECIAL.search(r["greedy_text"]))
        mk = sum(r["greedy_text"].count("Ġ") + r["greedy_text"].count("Ċ") for r in rows)
        print(f"{tag}: 행 {len(rows)} · 특수토큰 문자열 {sp} · Ġ/Ċ 마커 {mk}")
        if sp or mk:
            raise SystemExit("!! 디코딩 경로가 섞였다 — 병합하지 않는다")
    have = {r["pidx"] for r in old}
    add = [r for r in new if r["pidx"] not in have]
    rows = sorted(old + add, key=lambda r: r["pidx"])
    assert len({r["pidx"] for r in rows}) == len(rows), "pidx 중복"
    MERGED.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"병합 {len(old)} + {len(add)} = {len(rows)}행 → {MERGED.name}")
    return rows


def keep(suffix, restore=False):
    """교집합판 npz 를 접미사로 떠 두거나(restore=False) 되돌린다."""
    for w in WINS:
        for extra in ("", "_rep"):
            src = P / f"{A}_gen{w}{extra}.npz"
            dst = P / f"{A}_gen{w}{extra}{suffix}.npz"
            a, b = (dst, src) if restore else (src, dst)
            if a.is_file():
                shutil.copy2(a, b)
                print(f"{'되돌림' if restore else '보존'} {a.name} → {b.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off40", required=True)
    ap.add_argument("--off80", required=True)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()

    merge()
    keep("_isect")                      # 사전등록 교집합판을 먼저 떠 둔다
    run([PY, "experiments/probe_gen.py", "--model", DS, "--greedy", MERGED,
         "--labels", a.off40, "--sets", "ds_off40"], "DS 주묶음 특징 재추출 (보충 포함)")
    run([PY, "experiments/probe_gen.py", "--model", DS, "--greedy", MERGED,
         "--labels", a.off80, "--sets", "ds_off80", "--suffix", "_rep"],
        "DS 민감도 묶음 특징 재추출 (보충 포함)")
    for w in WINS:
        o = RES / f"probe_gen_report_supp{'' if w == 512 else f'_w{w}'}.txt"
        run([PY, "analysis/probe_gen_eval.py", "--win", w, "--perms", a.perms,
             "--jobs", a.jobs, "--out", o], f"보충판 재평가 창 {w}")
    run([PY, "analysis/probe_gen_eval.py", "--win", 512, "--perms", a.perms, "--jobs", a.jobs,
         "--suffix", "_rep", "--out", RES / "probe_gen_report_supp_rep.txt"],
        "보충판 off80 민감도 (창 512)")
    keep("_supp")                       # 보충판 npz 를 접미사로 남기고
    keep("_isect", restore=True)         # 기본 파일명은 사전등록 교집합판으로 되돌린다
    print("\n보충판 보고서: probe_gen_report_supp{,_w256,_w2048,_rep}.txt")
    print("기본 npz 는 교집합판(사전등록 주판정)으로 되돌렸다 · 보충판은 *_supp.npz 로 남는다")


if __name__ == "__main__":
    main()
