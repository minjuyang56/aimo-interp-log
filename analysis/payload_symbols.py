"""RunPod payload 의 외부 심볼 점검 — 파드를 띄우기 **전에** 돈다. GPU 0.

왜 필요한가 (2026-10-10 에 실제로 걸렸다)
  Kaggle 커널은 여러 파일을 **한 네임스페이스로 이어 붙여** 돌린다. 그래서 `genlab_v2.py` 는
  `SOLVE_PROMPT`·`extract_answer`(solve_common.py) 와 `official_variants`(variants_official.py) 를
  **주입받는 전제**로 쓰여 있고, 파일 안에는 정의가 없다(`# noqa: F821`).
  RunPod payload 를 두 파일로만 꾸렸다가 미정의 심볼 **4개**가 잡혔다 — 안 잡았으면 파드를 띄우고
  pip 설치하고 모델까지 올린 뒤에 NameError 로 죽었을 것이다. 과금되는 자원에서는 그게 곧 돈이다.

무엇을 하나
  payload 로 묶을 파일들을 ast 로 읽어 '정의되는 이름' 집합을 만들고, 본체가 **읽기로 쓰는 이름** 중
  거기에도 builtins 에도 없는 것을 센다. 0 이어야 한다.

사용:
  .venv/Scripts/python.exe analysis/payload_symbols.py experiments/runpod/jobs/labelgen32k.json
  .venv/Scripts/python.exe analysis/payload_symbols.py            # 모든 잡 정의를 점검
"""

import ast
import builtins
import json
import keyword
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "experiments" / "runpod" / "jobs"


#   모듈 실행 시 항상 있는 던더 — 안 넣으면 `__file__` 이 미정의로 잡힌다(오탐)
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
                  "__loader__", "__builtins__", "__dict__"}


def defined_names(files):
    names = set(dir(builtins)) | MODULE_DUNDERS
    for f in files:
        p = ROOT / f
        if not p.is_file():
            print(f"  !! payload 파일이 없다: {f}")
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                names.add(n.id)
            elif isinstance(n, ast.alias):
                names.add((n.asname or n.name).split(".")[0])
            elif isinstance(n, ast.arg):
                names.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                names.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                names |= set(n.names)
    return names


def loaded_names(f):
    tree = ast.parse((ROOT / f).read_text(encoding="utf-8"))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def check(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    files = list(spec.get("files", []))
    entry = JOBS / spec["entry"]
    print(f"## {Path(spec_path).name} · entry {spec['entry']} · payload {len(files)}개")
    for f in files:
        print(f"     {f}")
    have = defined_names(files + [str(entry.relative_to(ROOT))])
    bad = 0
    for f in files + [str(entry.relative_to(ROOT))]:
        if not (ROOT / f).is_file():
            continue
        miss = sorted(u for u in loaded_names(f)
                      if u not in have and not keyword.iskeyword(u))
        if miss:
            bad += len(miss)
            print(f"  **미정의 {len(miss)}개** in {f}: {miss}")
    print("  OK — 미정의 심볼 없음" if not bad else f"  !! 미정의 심볼 총 {bad}개 — 파드를 띄우지 말 것")
    return bad


def main():
    specs = [Path(a) for a in sys.argv[1:]] or sorted(JOBS.glob("*.json"))
    total = 0
    for s in specs:
        total += check(s)
        print()
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
