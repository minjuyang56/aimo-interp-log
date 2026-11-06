"""리포트의 **쪽수 계산 구역**을 재현 가능하게 센다 (지휘 #FINAL 4).

왜 스크립트로 두나: 마감 조건이 "5쪽 이내, 단 AI 사용 명시문·limitations·references·appendix 는
제외" 다. 그래서 PDF 전체 쪽수가 아니라 **Abstract + §1~§7 만** 떼어내 따로 컴파일해야 한다.
손으로 세면 고칠 때마다 틀리므로 여기서 기계로 센다.

하는 일:
  1. `reports/aimo_interp_report.tex` 에서 `\section*{Limitations}` **앞**까지만 잘라
     `reports/_counted_only.tex` 를 만든다 (그 뒤는 전부 쪽수 제외 구역이다).
  2. 둘 다 컴파일하고 TeX 로그의 `(N pages` 를 읽는다.
  3. 요약 단어수도 센다(지휘 기준 200단어 이하).

쪽수를 PDF 의 `/Type /Page` 로 세려 했으나 object stream 때문에 0 이 나왔다 → **TeX 로그**가 출처다.

사용: .venv/Scripts/python.exe analysis/report_pagecount.py [--engine <tectonic 경로>]
엔진이 없으면 .tex 생성과 단어수만 하고 쪽수는 '미측정' 으로 적는다(로컬에 LaTeX 가 없을 수 있다).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "reports/aimo_interp_report.tex"
CUT = ROOT / "reports/_counted_only.tex"
OUT = ROOT / "experiments/results/report_pagecount.txt"
BS = chr(92)
NL = chr(10)
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def make_cut(src: str) -> str:
    """쪽수에 세는 구역만 남긴다."""
    mark = BS + "section*{Limitations}"
    if mark not in src:
        raise SystemExit("Limitations 절을 못 찾았다 — 구역 경계가 바뀌었는지 확인할 것")
    i = src.index(mark)
    # 그 절 앞의 주석 구분선까지 되돌아간다 (구분선도 빼야 빈 줄이 안 남는다)
    j = src.rindex("% ====", 0, i)
    return src[:j] + NL + BS + "end{document}" + NL


def pages(engine: str, tex: Path) -> str:
    log = tex.with_suffix(".log")
    r = subprocess.run([engine, "-X", "compile", str(tex), "--keep-logs"],
                       capture_output=True, text=True, timeout=600)
    if not log.is_file():
        return "컴파일 실패 (rc=%d)" % r.returncode
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"\((\d+) pages", txt)
    errs = len(re.findall(r"(?m)^! ", txt))
    undef = len(re.findall(r"(?:Reference|Citation) .+ undefined", txt))
    return "%s쪽 · 오류 %d · 미정의 참조 %d" % (m[-1] if m else "?", errs, undef)


def abs_words(src: str) -> int:
    a = src.index(BS + "begin{abstract}") + len(BS + "begin{abstract}")
    b = src.index(BS + "end{abstract}")
    t = re.sub(BS + BS + "[a-zA-Z]+", " ", src[a:b])
    for ch in "{}$~":
        t = t.replace(ch, " ")
    return len([w for w in t.split() if any(c.isalnum() for c in w)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.environ.get("TECTONIC", "tectonic"))
    g = ap.parse_args()

    src = TEX.read_text(encoding="utf-8")
    CUT.write_text(make_cut(src), encoding="utf-8")
    out("리포트 쪽수 — %s" % TEX.name)
    out("  쪽수 계산 구역 = Abstract + 1~7절 (Limitations 앞까지) -> %s" % CUT.name)
    out("  요약 단어수 %d (기준 200 이하) -> %s"
        % (abs_words(src), "OK" if abs_words(src) <= 200 else "**초과**"))
    # 주석 줄(\TODO 를 설명하는 머리말)은 세지 않는다
    todo = sum(ln.count(BS + "TODO{") for ln in src.splitlines()
               if not ln.lstrip().startswith("%"))
    out("  남은 TODO 자리표시 %d개 (주석 줄 제외)" % todo)

    eng = shutil.which(g.engine) or (g.engine if Path(g.engine).is_file() else None)
    if not eng:
        out("  엔진 없음(%s) -> 쪽수 미측정. .tex 만 만들었다." % g.engine)
    else:
        out("  엔진 %s" % eng)
        out("  **쪽수 계산 구역: %s**" % pages(eng, CUT))
        out("    (절단판의 미정의 참조는 부록 라벨이 잘려서 생긴다 — 전체 PDF 쪽이 0 이어야 한다)")
        out("  전체 PDF:        %s" % pages(eng, TEX))
    OUT.write_text(NL.join(L) + NL, encoding="utf-8")
    print(NL + "저장 " + str(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
