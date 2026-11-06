"""0.643이 어디서 나온 숫자인지 한 단계씩 재현한다.

run_local.py는 임시 디렉터리를 쓰고 끝나면 지워버려서 중간 산출물이 안 남는다.
여기서는 ingestion.py를 직접 돌려 predictions.jsonl을 남기고,
scoring.py와 똑같은 방식으로 점수를 다시 계산해 내역을 보여준다.

사용법 (getting-started 저장소 안에서):
    uv run python /path/to/analysis/explain_score.py solutions/uncertainty-profiling
"""

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd()
INPUT_DIR = ROOT / "data" / "val-sample" / "input"
REFERENCE = ROOT / "data" / "val-sample" / "reference" / "labels.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    submission = (ROOT / sys.argv[1]).resolve()

    with tempfile.TemporaryDirectory(prefix="explain-") as temporary:
        output = Path(temporary) / "res"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "components" / "ingestion_program" / "ingestion.py"),
                str(INPUT_DIR),
                str(output),
                str(submission),
            ],
            check=True,
        )
        predictions = {row["id"]: row for row in read_jsonl(output / "predictions.jsonl")}

    cases = {row["id"]: row for row in read_jsonl(INPUT_DIR / "cases.jsonl")}
    labels = {row["id"]: row["is_robust"] for row in read_jsonl(REFERENCE)}

    print(f"제출물: {submission.name}")
    print(f"케이스 {len(labels)}건 · 정답 True {sum(labels.values())} / False "
          f"{len(labels) - sum(labels.values())}")
    print()

    groups = defaultdict(lambda: {"n": 0, "valid": 0, "pred_true": 0, "correct": 0})
    for case_id, label in labels.items():
        model_id = cases[case_id]["model_id"]
        row = predictions[case_id]
        bucket = groups[model_id]
        bucket["n"] += 1
        if row["valid"]:
            bucket["valid"] += 1
            bucket["pred_true"] += int(row["is_robust"])
            bucket["correct"] += int(row["is_robust"] == label)

    print("{:<34} {:>4} {:>6} {:>7} {:>6}".format(
        "model_id", "건수", "유효", "True예측", "정답"))
    for model_id in sorted(groups):
        bucket = groups[model_id]
        print("{:<34} {:>4} {:>6} {:>8} {:>6}".format(
            model_id, bucket["n"], bucket["valid"], bucket["pred_true"], bucket["correct"]))
    print()

    total = len(labels)
    correct = sum(g["correct"] for g in groups.values())
    covered = sum(g["valid"] for g in groups.values())

    print("scoring.py와 같은 계산 (분모는 언제나 전체 케이스 수):")
    print(f"  accuracy           = {correct} / {total} = {correct / total:.4f}")
    print(f"  coverage           = {covered} / {total} = {covered / total:.4f}")
    print(f"  invalid_predictions= {total - covered}")


if __name__ == "__main__":
    main()
