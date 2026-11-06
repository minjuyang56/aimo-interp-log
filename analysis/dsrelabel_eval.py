"""#DS-RELABEL 단계 2 — 깨끗한 DeepSeek 라벨로 **세 평가를 예전 규칙 그대로** 다시 돌린다. GPU: 로컬 4bit.

왜 드라이버로 묶나: 평가가 세 개(#77 · #DS-REP · #GEN-PROBE)이고 각각 특징 추출 → 평가 →
보고서 쓰기로 이어진다. 손으로 순서를 짜면 (a) 옛 npz 를 덮어써 비교 대상을 잃거나
(b) 접미사를 빼먹어 다른 묶음을 섞는다 — 둘 다 이 저장소에서 실제로 날 뻔했다.
그래서 **백업 → 추출 → 평가** 를 한 파일에 박고, 각 단계의 입력·출력을 찍는다.

바꾸지 않는 것(사전등록): 층 후보 · PCA 24 · C 격자 · 폴드 5 · 주라벨 label · gen_k==8 ·
null 1000 · 부호 · 창 256/512/2048 · 기준선 셋. **바뀌는 것은 라벨 파일뿐이다.**

사용:
  .venv/Scripts/python.exe analysis/dsrelabel_eval.py --stage backup
  .venv/Scripts/python.exe analysis/dsrelabel_eval.py --stage extract --off40 <labels.json> [--off80 <labels.json>]
  .venv/Scripts/python.exe analysis/dsrelabel_eval.py --stage eval [--perms 1000]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
P = ROOT / "experiments" / "results" / "probe"
RES = ROOT / "experiments" / "results"
DS = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
A = "DeepSeek-R1-0528-Qwen3-8B"
GREEDY = RES / "runpod" / "genprobe" / "oi8q4v8ia3ojq0" / "out" / f"{A}_greedy_fixed.json"
# 옛 손상 라벨로 만든 특징 — **복사해 남긴다**(옛 숫자를 다시 낼 수 있어야 비교가 성립한다)
BACKUP = ["own_k8", "own_k8rep", "gen256", "gen512", "gen2048",
          "gen256_rep", "gen512_rep", "gen2048_rep"]


def run(cmd, why):
    print(f"\n=== {why}\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(f"!! 실패 ({why}) rc={r.returncode}")


def stage_backup():
    n = 0
    for tag in BACKUP:
        src, dst = P / f"{A}_{tag}.npz", P / f"{A}_{tag}_corruptlab.npz"
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            print(f"백업 {src.name} → {dst.name}")
            n += 1
        elif dst.is_file():
            print(f"이미 백업됨 {dst.name}")
        else:
            print(f"원본 없음 {src.name}")
    print(f"백업 {n}건 — 옛(손상 라벨) 특징은 *_corruptlab.npz 로 남는다")


def stage_extract(off40, off80):
    if off40:
        run([PY, "experiments/probe_pilot.py", "--model", DS, "--labels", off40,
             "--tags", "own", "--suffix", "_k8"], "#77 DS 팔 특징 (새 off40 라벨)")
        run([PY, "experiments/probe_gen.py", "--model", DS, "--greedy", GREEDY,
             "--labels", off40, "--sets", "ds_off40"], "#GEN-PROBE DS 주묶음 특징 (새 off40 라벨)")
    if off80:
        run([PY, "experiments/probe_pilot.py", "--model", DS, "--labels", off80,
             "--tags", "own", "--suffix", "_k8rep"], "#DS-REP 학습 특징 (새 off80 라벨)")
        run([PY, "experiments/probe_gen.py", "--model", DS, "--greedy", GREEDY,
             "--labels", off80, "--sets", "ds_off80", "--suffix", "_rep"],
            "#GEN-PROBE DS 민감도 묶음 특징 (새 off80 라벨)")


def stage_eval(perms, jobs):
    # **경로는 저장소 기준 상대경로로 준다.** `probe_final_eval.py` 는 `--sets` 값을 `Path(path)`
    # 로 그대로 열기 때문에 파일명만 주면 cwd 에서 찾다 '파일 없음' 으로 조용히 건너뛴다
    # (2026-10-12 에 실제로 그렇게 비었고, 배선 점검에서도 그 칸이 빈 것을 놓쳤다).
    def pp(name):
        q = P / name
        if not q.is_file():
            raise SystemExit(f"!! 특징 파일 없음: {q}")
        return q.relative_to(ROOT).as_posix()

    sets = [f"Skywork={pp('Skywork-OR1-Math-7B_own_k8.npz')}",
            f"Olmo={pp('Olmo-3-7B-Think_own_k8.npz')}",
            f"DeepSeek={pp(A + '_own_k8.npz')}"]
    ext = [f"Skywork={pp('Skywork-OR1-Math-7B_official_v2.npz')}",
           f"Olmo={pp('Olmo-3-7B-Think_official.npz')}",
           f"DeepSeek={pp(A + '_official.npz')}"]
    # **옛 #77 보고서를 지키고 돌린다.** 평가기는 늘 probe_final_report.txt 에 쓰는데 그 파일은
    # CLAUDE.md 가 손상 라벨 시절 숫자의 출처로 가리키는 파일이다. 덮어쓰면 '나란히 비교' 의
    # 한쪽이 사라진다. 그래서 옛 것을 _corruptlab 으로 떠 두고, 새 것을 _relabel 로 받은 뒤
    # 원본 파일을 옛 내용으로 되돌린다(#DS-REP 때 썼던 그 수법).
    canon = RES / "probe_final_report.txt"
    keep = canon.read_text(encoding="utf-8") if canon.is_file() else None
    if keep is not None and not (RES / "probe_final_report_corruptlab.txt").is_file():
        (RES / "probe_final_report_corruptlab.txt").write_text(keep, encoding="utf-8")
    run([PY, "analysis/probe_final_eval.py", "--sets", *sets, "--external", *ext,
         "--perms", perms, "--jobs", jobs], "#77 재평가 (3모델, DS 팔만 교체)")
    shutil.copy2(canon, RES / "probe_final_report_relabel.txt")
    if keep is not None:
        canon.write_text(keep, encoding="utf-8")
    print(f"→ {RES / 'probe_final_report_relabel.txt'} (옛 보고서는 원위치로 복원)")

    run([PY, "analysis/probe_external_layers.py", "--sets",
         f"DS_off40={A}_own_k8.npz,{A}_official.npz",
         f"DS_off80={A}_own_k8rep.npz,{A}_official.npz",
         "--out", RES / "probe_dsrep_layers_relabel.txt"], "#DS-REP 전이 (네 층 전부, 두 묶음)")

    for w in (512, 256, 2048):
        out = RES / (f"probe_gen_report_relabel{'' if w == 512 else f'_w{w}'}.txt")
        run([PY, "analysis/probe_gen_eval.py", "--win", w, "--perms", perms,
             "--jobs", jobs, "--out", out], f"#GEN-PROBE 창 {w} 재평가")
    out = RES / "probe_gen_report_relabel_rep.txt"
    run([PY, "analysis/probe_gen_eval.py", "--win", 512, "--perms", perms, "--jobs", jobs,
         "--suffix", "_rep", "--out", out], "#GEN-PROBE off80 민감도 (창 512)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["backup", "extract", "eval"])
    ap.add_argument("--off40", default=None)
    ap.add_argument("--off80", default=None)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    if a.stage == "backup":
        stage_backup()
    elif a.stage == "extract":
        if not (a.off40 or a.off80):
            raise SystemExit("--off40 / --off80 중 하나는 줘야 한다")
        stage_extract(a.off40, a.off80)
    else:
        stage_eval(a.perms, a.jobs)


if __name__ == "__main__":
    main()
