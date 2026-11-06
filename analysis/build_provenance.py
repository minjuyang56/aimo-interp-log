"""공개 스냅샷에 `PROVENANCE.md` 를 만든다 — 리포트가 인용한 커밋 해시를 검증 가능하게 한다.

왜 필요한가: 공개 저장소는 **스냅샷**이라 리포트의 `[hash: file]` 표시가 그 저장소에서 해석되지
않는다(개발 히스토리 268커밋은 .npz 3.5 GB·100MB 초과 6개로 push 불가). 해시를 그냥 버리면
리포트의 "모든 숫자는 커밋된 로그에서 가져왔다" 는 주장이 공개 쪽에서 확인 불가가 된다.
→ 인용된 해시마다 **전체 해시·날짜·커밋 제목**을 표로 남긴다. 로그 파일 자체는 스냅샷 안에 있으므로
   읽는 사람은 '그 날 그 커밋이 그 숫자를 냈다' 를 실질적으로 확인할 수 있다.

사용: .venv/Scripts/python.exe analysis/build_provenance.py [스냅샷경로]
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "reports/aimo_interp_report.tex"

HEAD = """# Provenance of the commit hashes cited in the report

This public repository is a **snapshot**. The short hashes printed in
`reports/aimo_interp_report.tex` (the `[hash: file]` markers after each claim) therefore do
not resolve here. They refer to the private development history of 268 commits, which cannot
be pushed to GitHub: it tracks 3.5 GB of `.npz` feature caches, six of them above GitHub's
100 MB hard limit. Rewriting that history would change every hash anyway, so the citations
could not be preserved by any route.

What this table gives instead is enough to check each citation in substance -- the full hash,
the date, and the subject line of the commit that produced the number. The measurement logs
themselves are all present in this snapshot under `experiments/results/`, and that is where
every number in the report was read from; `analysis/report_numbers_check.py` verifies that
mechanically (122 of 122 numbers located, 0 unaccounted).

| cited | full hash | date | commit subject |
|---|---|---|---|
"""


def main():
    snap = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/side/aimo-interp-release")
    hs = sorted(set(re.findall(r"cmt\{([0-9a-f]{7,40})\}", TEX.read_text(encoding="utf-8"))))
    rows, bad = [], 0
    for h in hs:
        r = subprocess.run(["git", "log", "-1", "--format=%H%x09%ad%x09%s", "--date=short", h],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0 and r.stdout.strip():
            full, d, s = r.stdout.strip().split("\t", 2)
            rows.append("| `%s` | `%s` | %s | %s |" % (h, full, d, s.replace("|", "/")))
        else:
            bad += 1
            rows.append("| `%s` | (unresolved) | | |" % h)
    body = HEAD + "\n".join(rows) + "\n"
    (snap / "PROVENANCE.md").write_text(body, encoding="utf-8", newline="\n")
    print("인용 커밋 %d개 · 해석 실패 %d개" % (len(hs), bad))
    print("저장 %s (%d bytes)" % (snap / "PROVENANCE.md", len(body.encode())))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
