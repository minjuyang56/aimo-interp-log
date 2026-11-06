"""제출 ZIP을 풀어서 are_robust를 직접 호출하고 반환 계약을 검사한다.

채점기(ingestion.py)는 예외를 조용히 삼키고 이유를 알려주지 않는다.
그러니 제출 전에 여기서 같은 조건으로 먼저 불러 본다.

검사 항목:
  - ZIP 루트에 solution.py가 있는가 (상위 폴더로 감싸면 실패)
  - 반환이 list 인가
  - 길이가 입력과 같은가
  - 모든 원소가 type(x) is bool 인가  (np.bool_ / int 는 무효)

모델 로드가 필요한 솔루션은 로컬 캐시가 없으면 실패한다. 그 경우 실패 지점을 보여준다.

사용:  python analysis/smoke_submission.py            # dist/solution-*.zip 전부
       python analysis/smoke_submission.py a.zip b.zip
"""

import importlib.util
import inspect
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

from _paths import DIST, require

PROBLEMS = [
    "What is the value of $255^2 - 245^2$?",
    "Find the number of positive divisors of 2024.",
]
MODEL_ID = "qwen3-8b:low"


def check(zip_path):
    print("=" * 66)
    print(zip_path.name)
    print("=" * 66)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        print("  파일 수 :", len(names))
        if "solution.py" not in names:
            print("  FAIL    : ZIP 루트에 solution.py 없음. 있는 것:", names[:5])
            return False
        with tempfile.TemporaryDirectory() as tmp:
            archive.extractall(tmp)
            return call(Path(tmp))


def call(root):
    sys.path.insert(0, str(root))
    module_name = "_smoke_solution"
    try:
        spec = importlib.util.spec_from_file_location(module_name, root / "solution.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        fn = module.are_robust
        sig = inspect.signature(fn)
        print("  시그니처:", sig)
        # 2026-09-22 ingestion.py: reasoning_effort 파라미터가 있으면 키워드로 3-인자 호출,
        # 없으면 옛 2-인자 호출. 채점기와 똑같이 분기한다.
        accepts_effort = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            or (p.name == "reasoning_effort" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))
            for p in sig.parameters.values())
        if accepts_effort:
            out = fn(MODEL_ID, reasoning_effort="default", problems=PROBLEMS)
            print("  호출    : 3-인자 (model_id, reasoning_effort=, problems=)")
        else:
            out = fn(MODEL_ID, PROBLEMS)
            print("  호출    : 2-인자 (legacy)")
        print("  반환값  :", out)
        assert type(out) is list, f"list 아님: {type(out)}"
        assert len(out) == len(PROBLEMS), f"길이 {len(out)} != {len(PROBLEMS)}"
        for i, x in enumerate(out):
            assert type(x) is bool, f"[{i}] type={type(x).__name__} — 순수 bool 아님"
        print("  PASS    : list / 길이 일치 / 전 원소 type(x) is bool")
        return True
    except Exception as exc:
        print(f"  FAIL    : {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        for f in traceback.extract_tb(exc.__traceback__)[-3:]:
            print(f"            {Path(f.filename).name}:{f.lineno}  {f.name}")
        return False
    finally:
        sys.path.remove(str(root))
        sys.modules.pop(module_name, None)


def main():
    global MODEL_ID
    require(DIST)
    args = list(sys.argv[1:])
    if "--model-id" in args:                       # 본선은 model_id를 그대로 넘긴다
        i = args.index("--model-id")
        MODEL_ID = args[i + 1]
        del args[i:i + 2]
    print(f"model_id = {MODEL_ID!r}")
    targets = [Path(a) for a in args] or sorted(DIST.glob("solution-*.zip"))
    results = {z.name: check(z) for z in targets}
    print()
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
