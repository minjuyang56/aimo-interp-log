"""공개 릴리스 스냅샷 빌더 (지휘 #FINAL 3) — **push 하지 않는다. 트리만 만든다.**

왜 스냅샷인가: 개발 저장소는 추적 3.58 GB 이고 그중 .npz 가 3.54 GB,
**100 MB 를 넘는 파일이 6개** 다(GitHub 하드 거부 한도). 즉 이 히스토리는 그대로 push 되지 않는다.
`git filter-repo` 는 이 머신에 없고, 히스토리를 다시 쓰면 **리포트가 인용한 커밋 해시가 바뀐다**.
→ 그래서 코드·작은 로그만 담은 **스냅샷 저장소**를 따로 만들고, 리포트에는 '개발 히스토리는
   비공개이고 공개 릴리스는 스냅샷' 이라고 적는다. 선택은 [author]가 한다.

담는 것:  .py .sh .md .txt .html .tex + 1 MB 미만 .json/.parquet + LICENSE + README
빼는 것:  .npz (전부) · 1 MB 초과 아무것 · dist/*.zip · 임시 패치 스크립트 · runpod 산출물 ·
          .venv · __pycache__ · .git

사용: .venv/Scripts/python.exe analysis/build_release.py [--dest C:/side/aimo-interp-release]
"""

import argparse
import os
import re
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/results/release_build.txt"
L = []

KEEP_EXT = {".py", ".sh", ".md", ".txt", ".html", ".tex", ".bib", ".cfg", ".toml", ".yml",
            ".yaml", ".log", ".png", ".patch", ".pdf"}   # .log = 리포트 숫자의 출처라 반드시 포함
SMALL_EXT = {".json", ".parquet", ".csv"}          # 1 MB 미만만
MAXSZ = 1048576
DROP_RX = re.compile(r"(?i)(^|/)(_patch_|_tmp_|scratch)")   # 임시 스크립트
DROP_DIR = ("experiments/results/runpod/", "experiments/results/sae/",
            "experiments/results/fbank/", "experiments/results/h10b/")


#  익명 심사용 치환 (지휘 2026-11-06). 개발 저장소 원본은 기록으로 그대로 두고,
#  **공개되는 트리에서만** 신원 문자열을 지운다. 리포트가 anonymous.4open.science 로
#  이 트리를 가리키므로 파일 안에 이름이 남으면 익명화가 깨진다.
ANON = [
    ("[Author]", "[Author]"),
    ("[author-github]", "[author-github]"),
    ("[author-codabench]", "[author-codabench]"),
    ("[author-kaggle]", "[author-kaggle]"),
    ("[Author]", "[Author]"),
    ("[author]", "[author]"),
    ("[user]", "[user]"),
]
ANON_EXT = {".py", ".sh", ".md", ".txt", ".tex", ".json", ".yml", ".yaml",
            ".cfg", ".toml", ".html", ".log", ".csv", ".bib", ".patch"}


def anon_copy(src, dst):
    """텍스트면 치환해서 쓰고, 아니면 그대로 복사한다. 치환 건수를 돌려준다."""
    if src.suffix.lower() not in ANON_EXT:
        shutil.copy2(src, dst)
        return 0
    try:
        txt = src.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        shutil.copy2(src, dst)
        return 0
    k = 0
    for a, b in ANON:
        if a in txt:
            k += txt.count(a)
            txt = txt.replace(a, b)
    dst.write_text(txt, encoding="utf-8", newline="")
    return k


def out(s=""):
    print(s, flush=True)
    L.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="C:/side/aimo-interp-release")
    a = ap.parse_args()
    dest = Path(a.dest)

    files = [f for f in subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.split("\0") if f]
    out("공개 릴리스 스냅샷 빌드 — 원본 추적 %d개" % len(files))
    out("  대상: %s" % dest)
    out()

    keep, drop = [], {}
    for f in files:
        p = ROOT / f
        if not p.is_file():
            continue
        sz = p.stat().st_size
        ext = p.suffix.lower()
        why = None
        if any(f.startswith(d) for d in DROP_DIR):
            why = "실행 산출물 디렉터리"
        elif DROP_RX.search(f):
            why = "임시 스크립트"
        elif ext == ".npz":
            why = ".npz (재생성 가능)"
        elif ext in KEEP_EXT:
            if sz > MAXSZ:
                why = "텍스트인데 1MB 초과"
        elif ext in SMALL_EXT:
            if sz > MAXSZ:
                why = "데이터 1MB 초과"
        else:
            why = "대상 확장자 아님 (%s)" % (ext or "없음")
        if why:
            drop.setdefault(why, []).append((sz, f))
        else:
            keep.append((sz, f))

    out("[제외]")
    for why in sorted(drop, key=lambda k: -sum(s for s, _ in drop[k])):
        g = drop[why]
        out("  %-26s %4d개 · %9.2f MB" % (why, len(g), sum(s for s, _ in g) / 1048576))
    out()
    out("[포함] %d개 · %.2f MB" % (len(keep), sum(s for s, _ in keep) / 1048576))
    from collections import defaultdict
    byext = defaultdict(lambda: [0, 0])
    for s, f in keep:
        e = Path(f).suffix.lower() or "(없음)"
        byext[e][0] += 1
        byext[e][1] += s
    for e, (n, s) in sorted(byext.items(), key=lambda x: -x[1][1]):
        out("    %-10s %4d개 %8.2f MB" % (e, n, s / 1048576))
    out()

    if dest.exists():
        shutil.rmtree(dest)
    nanon = 0
    for s, f in keep:
        t = dest / f
        t.parent.mkdir(parents=True, exist_ok=True)
        nanon += anon_copy(ROOT / f, t)
    out("  익명 치환 %d건 적용 (공개 트리에서만 · 개발 저장소 원본 불변)" % nanon)
    #  dist 의 제출 zip 은 작은 것만 넣는다(재현에 쓰인다)
    dz = dest / "dist"
    dz.mkdir(parents=True, exist_ok=True)
    nz = 0
    skipped = []
    for z in sorted((ROOT / "dist").glob("*.zip")):
        if z.stat().st_size > MAXSZ:
            continue
        #  zip 내부는 ANON 필터가 못 닿는다 → 신원 문자열이 있으면 공개 트리에서 뺀다.
        #  다시 압축하면 기록된 sha256 이 깨져 '제출된 그 파일' 이라는 주장이 무의미해진다.
        hit = None
        try:
            with zipfile.ZipFile(z) as zf:
                for nm in zf.namelist():
                    if not nm.endswith((".py", ".txt", ".json", ".md")):
                        continue
                    t = zf.read(nm).decode("utf-8", "replace")
                    for a, _ in ANON:
                        if a in t:
                            hit = "%s (%s)" % (nm, a[:12])
                            break
                    if hit:
                        break
        except Exception:
            hit = "zip 열기 실패"
        if hit:
            skipped.append("%s <- %s" % (z.name, hit))
            continue
        shutil.copy2(z, dz / z.name)
        nz += 1
    out("  제출 zip %d개 복사 (1MB 미만만 — sub2048 은 가중치 6.8MB 라 제외)" % nz)
    for sk in skipped:
        out("    익명화 위해 제외: %s" % sk)

    tot = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    nfile = sum(1 for p in dest.rglob("*") if p.is_file())
    out("  릴리스 트리 파일 %d개 · %.2f MB" % (nfile, tot / 1048576))
    over = [p for p in dest.rglob("*") if p.is_file() and p.stat().st_size > 50 * 1048576]
    out("  50MB 초과 파일: %d개 %s" % (len(over), over if over else "(없음 — GitHub 통과)"))
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))


if __name__ == "__main__":
    sys.exit(main())
