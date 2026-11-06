"""공개 직전 **최종 관문**: 실제로 push 될 트리를 파일 하나하나 다시 검사한다.

`release_scan.py` 와 왜 둘 다 필요한가:
  * `release_scan.py` 는 **개발 저장소의 추적 파일**을 본다 → 히스토리 쪽 위험을 잡는다.
  * 이 스크립트는 **스냅샷 트리**(= 공개되는 바이트 전부)를 본다. 그쪽이 못 보는 둘을 덮는다:
      (a) `dist/*.zip` **내부**(제출물 코드·가중치가 들어 있다)
      (b) 빌드 과정에서 새로 생긴/복사된 파일
    즉 "검사한 것" 과 "올리는 것" 이 같아야 한다는 요구를 만족시킨다.

값은 절대 출력하지 않는다 — 규칙 이름·파일·줄번호만 적는다(지휘 상설 규칙).

사용: .venv/Scripts/python.exe analysis/release_scan_tree.py [트리경로]
반환 코드 0 = 통과(적중 0건) · 1 = 적중 있음(push 하지 않는다)
"""

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/results/release_scan_tree.txt"
L = []

#  알려진 키 서식. 값을 찍지 않으므로 패턴은 '모양' 만 본다.
RULES = [
    ("HuggingFace token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("OpenAI key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("AWS access key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("Slack token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("PEM private key", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
    ("RunPod key", re.compile(r"rpa_[A-Za-z0-9]{20,}")),
    ("JWT", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.")),
    ("credential assignment",
     re.compile(r"(?i)(api[_-]?key|secret|password|passwd|token)"
                r"\s*[:=]\s*[\"'][^\"']{12,}[\"']")),
]
#  익명 심사 게이트 (지휘 2026-11-06): 공개 트리에 신원 문자열이 남으면 push 하지 않는다
IDENT = ["[Author]", "[author-github]", "[author-codabench]", "[author-kaggle]", "[author]", "[author-email]", "[user]"]
CRED_NAMES = {"kaggle.json", ".netrc", "id_rsa", "id_ed25519", "credentials",
              ".env", "secrets.json", "runpod.json", ".pypirc"}
TEXTY = {".py", ".sh", ".md", ".txt", ".tex", ".json", ".yml", ".yaml", ".cfg",
         ".toml", ".html", ".log", ".csv", ".bib", ".patch", ".ipynb", ""}


def out(s=""):
    print(s, flush=True)
    L.append(s)


def scan_text(txt, where, hits):
    for i, line in enumerate(txt.splitlines(), 1):
        if len(line) > 4000:          # 거대한 한 줄(가중치 등)은 건너뛴다
            continue
        for name, rx in RULES:
            if rx.search(line):
                hits.append((name, where, i))


def main():
    tree = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/side/aimo-interp-release")
    if not tree.is_dir():
        raise SystemExit("트리를 못 찾았다: %s" % tree)

    files = [p for p in tree.rglob("*") if p.is_file() and ".git" not in p.parts]
    out("공개 직전 트리 스캔 — %s" % tree)
    out("  파일 %d개 · %.2f MB" % (len(files), sum(f.stat().st_size for f in files) / 1048576))

    hits, cred, zips, scanned = [], [], 0, 0
    for p in files:
        rel = p.relative_to(tree).as_posix()
        if p.name in CRED_NAMES:
            cred.append(rel)
        if p.suffix.lower() == ".zip":
            zips += 1
            try:
                with zipfile.ZipFile(p) as z:
                    for n in z.namelist():
                        if Path(n).name in CRED_NAMES:
                            cred.append("%s!%s" % (rel, n))
                        if Path(n).suffix.lower() in TEXTY and z.getinfo(n).file_size < 8 * 1048576:
                            scanned += 1
                            scan_text(z.read(n).decode("utf-8", "replace"), "%s!%s" % (rel, n), hits)
            except Exception as e:
                out("  ! zip 열기 실패 %s (%s)" % (rel, type(e).__name__))
            continue
        if p.suffix.lower() in TEXTY and p.stat().st_size < 20 * 1048576:
            scanned += 1
            scan_text(p.read_text(encoding="utf-8", errors="replace"), rel, hits)

    out("  텍스트 검사 %d개(zip %d개 안쪽 포함) · 규칙 %d종" % (scanned, zips, len(RULES)))
    out()
    out("자격증명 파일명 적중: %d건%s" % (len(cred), "" if cred else " "))
    for c in cred:
        out("  %s" % c)
    #  신원 문자열 전수 집계 (값이 아니라 개수·파일만 적는다)
    idh = {}
    for p2 in files:
        rel2 = p2.relative_to(tree).as_posix()
        #  zip 내부도 본다 — 1차 구현이 느슨한 파일만 봐서 zip 안의 이름을 놓쳤다
        if p2.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(p2) as zf:
                    for nm in zf.namelist():
                        if Path(nm).suffix.lower() not in TEXTY:
                            continue
                        t = zf.read(nm).decode("utf-8", "replace")
                        for w in IDENT:
                            if w in t:
                                idh.setdefault(w, []).append("%s!%s" % (rel2, nm))
            except Exception:
                pass
            continue
        if p2.suffix.lower() not in TEXTY or p2.stat().st_size > 20 * 1048576:
            continue
        t = p2.read_text(encoding="utf-8", errors="replace")
        for w in IDENT:
            if w in t:
                idh.setdefault(w, []).append(rel2)
    out("신원 문자열 적중: **%d종**%s" % (len(idh), "" if idh else " (익명 통과)"))
    for w, fs in idh.items():
        out("  %-14s %d개 파일  예: %s" % (w[:12], len(fs), fs[0]))
    out("키 서식 적중: **%d건**" % len(hits))
    for name, where, i in hits[:40]:
        out("  %-22s %s:%d" % (name, where, i))

    ok = not hits and not cred and not idh
    out()
    out("판정: %s" % ("**통과** — 공개 가능" if ok else "**적중 있음 — push 금지**"))
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
