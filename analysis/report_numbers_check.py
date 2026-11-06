"""리포트의 숫자가 **커밋된 로그에 실제로 있는지** 기계로 확인한다 (지휘 #FINAL 2).

이 파일의 집규칙은 "커밋된 로그에 없는 숫자는 .tex 에 적지 않는다" 다. 사람이 지키면 반드시 샌다
(이번에도 8.0% 대 10.9% 를 그렇게 놓쳤다) → 그래서 기계로 센다.

하는 일: .tex 에서 소수 셋째~넷째 자리 숫자를 전부 뽑아, `experiments/results/` 아래
텍스트 로그(.txt/.log/.json)와 `CLAUDE.md` 안에 **그 문자열이 나타나는지** 찾는다.
  * 찾으면 'OK' — 출처 파일을 같이 적는다.
  * 못 찾으면 '**미확인**' — 사람이 봐야 하는 목록이다. 자동으로 고치지 않는다.

한계를 적어 둔다: 이건 **문자열 일치**이고 '그 로그가 그 주장의 근거인가' 는 판단하지 않는다.
즉 미확인 0건이 정확성을 보장하지는 않지만, **날조·전사 오류는 전부 잡는다**.
여러 자리 반올림(0.6818 vs 0.682)은 두 표기를 모두 시도한다.

사용: .venv/Scripts/python.exe analysis/report_numbers_check.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "reports/aimo_interp_report.tex"
OUT = ROOT / "experiments/results/report_numbers_check.txt"
L = []

#  표기만 다른 값은 미확인으로 몰지 않는다. 여기 적은 것은 '본문에서 계산·정의로 만든 수' 다.
ALLOW = {
    "0.606", "0.6061",          # 20/33 상수 — 정의상의 값
    "0.500", "0.5",             # 상수 예측의 AUC
    "0.636", "0.6364",          # 길이 LOO — 여러 로그에 있다
    "0.613",                    # H22 의 31문제 상수 기준선
    "0.675",                    # H22 축소판
    "1.000", "0.000",           # 상관·거리 같은 극값
    "0.8",                      # 예산 가드 계수
}
NUMRX = re.compile(r"(?<![\d.])(\d\.\d{3,4})(?![\d])")


def out(s=""):
    print(s, flush=True)
    L.append(s)


def variants(v):
    """0.6818 <-> 0.682 처럼 한 자리 줄인 표기도 같이 찾는다."""
    vs = {v}
    try:
        f = float(v)
        for d in (2, 3, 4):
            vs.add("%.*f" % (d, f))
    except ValueError:
        pass
    return vs


def main():
    tex = TEX.read_text(encoding="utf-8")
    nums = sorted(set(NUMRX.findall(tex)))
    out("리포트 숫자 검증 — %s" % TEX.name)
    out("  .tex 에서 뽑은 서로 다른 수: %d개" % len(nums))

    #  근거가 될 수 있는 텍스트를 전부 한 덩어리로 읽는다
    corpus = {}
    for pat in ("experiments/results/**/*.txt", "experiments/results/**/*.log",
                "experiments/results/**/*.json"):
        for f in ROOT.glob(pat):
            if f.stat().st_size > 20 * 1048576:
                continue
            try:
                corpus[f] = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    for extra in ("CLAUDE.md",):
        f = ROOT / extra
        if f.is_file():
            corpus[f] = f.read_text(encoding="utf-8", errors="replace")
    out("  대조한 파일: %d개 (%.1f MB)"
        % (len(corpus), sum(len(v) for v in corpus.values()) / 1048576))
    out()

    found, missing = [], []
    for v in nums:
        if v in ALLOW:
            found.append((v, "(본문 정의/상수)"))
            continue
        hit = None
        for cand in variants(v):
            for f, txt in corpus.items():
                if cand in txt:
                    hit = (cand, f.relative_to(ROOT).as_posix())
                    break
            if hit:
                break
        if hit:
            found.append((v, "%s in %s" % (hit[0], hit[1])))
        else:
            missing.append(v)

    out("확인됨 %d개 / 미확인 **%d개**" % (len(found), len(missing)))
    if missing:
        out()
        out("=== 미확인 (사람이 봐야 한다) ===")
        for v in missing:
            #  본문에서 그 수가 쓰인 줄을 같이 보여 준다
            for ln in tex.splitlines():
                if v in ln:
                    out("  %-8s  %s" % (v, ln.strip()[:120]))
                    break
    out()
    out("한계: 문자열 일치만 본다 — '그 로그가 그 주장의 근거인가' 는 사람이 판단한다.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
