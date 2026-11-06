"""solutions/<이름>/ 를 제출 ZIP 두 개(Main · Small)로 싼다.

  dist/<이름>.zip          solution.py 루트, small.txt 없음
  dist/<이름>-small.zip    + 빈 small.txt

__pycache__ · .pyc · 숨김 파일은 뺀다. 싸고 나서 unzip -l 처럼 목록을 찍는다.

사용:  uv run python scripts/package.py probe [trajectory ...]
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def entries(directory: Path):
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(directory)
        if "__pycache__" in rel.parts or rel.suffix == ".pyc" or rel.name.startswith("."):
            continue
        if rel.name.casefold() == "small.txt":
            continue
        yield rel.as_posix(), path


def build(name: str, small_only: bool = False):
    """`--small-only` 는 Small ZIP 만 만든다 — 실험 제출물처럼 **한 트랙만 지명할 때** 쓴다
    (Main zip 을 같이 두면 어느 쪽을 올릴지 혼동이 생기고, 둘 다 고르면 0점이다)."""
    src = ROOT / "solutions" / name
    if not (src / "solution.py").is_file():
        raise SystemExit(f"{src} 에 solution.py 없음")
    DIST.mkdir(exist_ok=True)
    for small in ((True,) if small_only else (False, True)):
        out = DIST / f"{name}{'-small' if small else ''}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for rel, path in entries(src):
                z.write(path, rel)
            if small:
                z.writestr("small.txt", "")
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            total = sum(i.file_size for i in z.infolist())
        print(f"{out.name:28s} {total / 1e6:7.2f} MB  {len(names)} files  root={'solution.py' in names}"
              f"  small.txt={'small.txt' in names}")
        for n in names:
            print(f"    {n}")


if __name__ == "__main__":
    args = sys.argv[1:]
    small_only = "--small-only" in args
    for name in [a for a in args if not a.startswith("-")] or ["probe"]:
        build(name, small_only)
