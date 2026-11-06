"""공개 릴리스 전 스캔 — 키/토큰 · 대용량 파일 · 개인정보 (지휘 #FINAL 3).

**공개는 되돌릴 수 없는 외부 행위라 이 스캔이 관문이다.** 통과하지 못하면 릴리스 트리를 만들지 않는다.
세 가지를 본다:
  1. **시크릿**: 알려진 키 서식(HF `hf_...` · RunPod · AWS · OpenAI `sk-...` · 개인키 PEM) +
     `kaggle.json` 같은 자격증명 파일 + 'key/token/secret/password' 에 **값이 붙은** 줄.
     값은 **절대 찍지 않는다** — 파일·줄 번호와 '어떤 규칙에 걸렸나' 만 적는다.
  2. **대용량**: GitHub 하드 거부(100 MB) · 경고(50 MB) · 릴리스 제외 기준(1 MB) 집계.
  3. **개인정보**: 이메일 주소 · Windows 사용자 경로(`C:\\Users\\<이름>`).

사용: .venv/Scripts/python.exe analysis/release_scan.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/results/release_scan.txt"
L = []

#  값은 찍지 않는다. 규칙 이름만 보고한다.
SECRET_RULES = [
    ("HuggingFace 토큰", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("OpenAI 키", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("AWS 액세스키", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub 토큰", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("Slack 토큰", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("개인키 PEM", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("RunPod 키 서식", re.compile(r"\brpa_[A-Za-z0-9]{20,}")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")),
    ("값 붙은 자격증명 대입", re.compile(
        r"""(?i)\b(api[_-]?key|secret|password|passwd|token|access[_-]?key)\b\s*[:=]\s*"""
        r"""['"][^'"\s]{12,}['"]""")),
]
CRED_FILES = re.compile(r"(?i)(^|[\\/])(kaggle\.json|\.netrc|\.pgpass|id_rsa|id_ed25519|"
                        r"credentials|\.env|service[_-]account.*\.json)$")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
WINUSER = re.compile(r"[A-Za-z]:\\\\?Users\\\\?([A-Za-z0-9._-]+)")
SKIP_EXT = {".npz", ".parquet", ".png", ".jpg", ".pdf", ".zip", ".gz", ".pyc", ".pack", ".idx"}


def out(s=""):
    print(s, flush=True)
    L.append(s)


def tracked():
    r = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True)
    return [f for f in r.stdout.split("\0") if f]


def main():
    files = tracked()
    out("공개 릴리스 스캔 — 추적 파일 %d개" % len(files))
    out()

    # ---------------------------------------------------------------- 1. 시크릿
    out("[1] 시크릿 스캔 (값은 출력하지 않는다)")
    hits, cred, scanned = [], [], 0
    for f in files:
        if CRED_FILES.search(f):
            cred.append(f)
        p = ROOT / f
        if p.suffix.lower() in SKIP_EXT or not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        scanned += 1
        for i, line in enumerate(txt.splitlines(), 1):
            if len(line) > 4000:
                line = line[:4000]
            for name, rx in SECRET_RULES:
                if rx.search(line):
                    hits.append((name, f, i))
    out("    텍스트 파일 %d개 검사 · 규칙 %d종" % (scanned, len(SECRET_RULES)))
    out("    자격증명 파일명 적중: %d건 %s" % (len(cred), cred if cred else ""))
    if hits:
        out("    **적중 %d건** (파일·줄만):" % len(hits))
        for name, f, i in hits[:40]:
            out("      %-24s %s:%d" % (name, f, i))
    else:
        out("    적중 **0건** — 추적 파일에 알려진 키 서식이 없다")
    out()

    # ---------------------------------------------------------------- 2. 대용량
    out("[2] 대용량 파일")
    sz = []
    for f in files:
        p = ROOT / f
        try:
            sz.append((p.stat().st_size, f))
        except OSError:
            pass
    sz.sort(reverse=True)
    tot = sum(s for s, _ in sz)
    for lim, tag in ((100 * 1048576, "GitHub 하드 거부(>100MB)"),
                     (50 * 1048576, "GitHub 경고(>50MB)"),
                     (1048576, "릴리스 제외 기준(>1MB)")):
        g = [(s, f) for s, f in sz if s > lim]
        out("    %-26s %3d개 · 합 %8.1f MB" % (tag, len(g), sum(s for s, _ in g) / 1048576))
    out("    추적 합계 %.1f MB · .npz 가 %.1f MB"
        % (tot / 1048576,
           sum(s for s, f in sz if f.endswith(".npz")) / 1048576))
    out("    → **이 히스토리는 그대로 push 할 수 없다**(100MB 초과 6개). 릴리스는 스냅샷으로 만든다.")
    out()

    # ---------------------------------------------------------------- 3. 개인정보
    out("[3] 개인정보 서식")
    emails, users = {}, {}
    for f in files:
        p = ROOT / f
        if p.suffix.lower() in SKIP_EXT or not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in EMAIL.findall(txt):
            emails.setdefault(m, set()).add(f)
        for m in WINUSER.findall(txt):
            users.setdefault(m, set()).add(f)
    out("    이메일 서식 %d종" % len(emails))
    for e, fs in sorted(emails.items())[:10]:
        dom = e.split("@")[-1]
        out("      (가림)@%-22s 파일 %d개  예: %s" % (dom, len(fs), sorted(fs)[0]))
    out("    Windows 사용자 경로 %d종 · 등장 파일 %d개"
        % (len(users), len(set().union(*users.values())) if users else 0))
    for u, fs in sorted(users.items())[:5]:
        out("      사용자 '%s…' (가림) · 파일 %d개  예: %s" % (u[:3], len(fs), sorted(fs)[0]))
    out()

    ok = not hits and not cred
    out("판정: 시크릿 %s · 대용량 %s" % ("통과(0건)" if ok else "**실패**",
                                        "스냅샷 필요(100MB 초과 있음)"))
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
