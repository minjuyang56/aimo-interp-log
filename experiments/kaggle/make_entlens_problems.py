"""공식 33문제를 base64(JSON) 로 커널에 실을 파일(`entlens_problems.py`)을 만든다.

왜 base64 인가: 올림피아드 문제 텍스트는 LaTeX 라 백슬래시가 가득하고, 이 저장소에서 그걸
셸·정규식·heredoc 단계에서 먹혀 파일을 깨뜨린 적이 **9번** 있다. base64 는 어느 단계에서도
해석되지 않는다. 커널 안에서 길이·해시를 다시 확인한다.

사용: .venv/Scripts/python.exe experiments/kaggle/make_entlens_problems.py
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from h21_uncertainty import problems  # noqa: E402

OUT = HERE / "entlens_problems.py"


def main():
    rows = [{"pidx": int(r["pidx"]), "problem": r["problem"]} for r in problems()]
    if len(rows) != 33:
        raise SystemExit(f"문제 수가 33이 아니다: {len(rows)} — 라벨 원천을 확인하라")
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    b64 = base64.b64encode(blob.encode("utf-8")).decode("ascii")
    sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    body = (
        '"""공식 train-main-v2 문제단위 33문제 (base64 JSON). '
        'make_entlens_problems.py 가 생성한다 — 손으로 고치지 말 것."""\n\n'
        f'PROBS_SHA256 = "{sha}"\n'
        f'PROBS_N = {len(rows)}\n'
        f'PROBS_B64 = (\n'
        + "".join(f'    "{b64[i:i + 100]}"\n' for i in range(0, len(b64), 100))
        + ')\n'
    )
    OUT.write_text(body, encoding="utf-8")
    # 왕복 확인 — 쓴 파일을 다시 읽어 디코드가 원본과 같은지 센다(주장하지 않고 계측).
    ns = {}
    exec(compile(OUT.read_text(encoding="utf-8"), str(OUT), "exec"), ns)  # noqa: S102
    back = json.loads(base64.b64decode(ns["PROBS_B64"]).decode("utf-8"))
    same = back == rows
    print(f"저장 {OUT} · {len(b64)} 문자 · 문제 {len(rows)} · sha256 {sha[:16]}…")
    print(f"왕복 확인: 디코드 == 원본 → {'일치' if same else '**불일치**'}")
    words = sorted(len(r["problem"].split()) for r in rows)
    print(f"단어 수 최소 {words[0]} · 중앙 {words[len(words) // 2]} · 최대 {words[-1]}")
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
