"""H10-B 를 Kaggle 에 모델당 커널 하나로 올린다 (지휘 #19). 토큰 파일은 kaggle CLI만 읽는다; 여기선 username만 읽는다.

사용:  .venv/Scripts/python.exe experiments/kaggle/kaggle_h10b.py [--models ...] [--push-only | --status-only | --output-only]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kaggle_run import EXP, TOKEN, ensure_cli, output as _output, push, status, username, wait  # noqa: E402

MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]


SUFFIX = ""
HEADER = "h10b_kaggle.py"
BODY = "h10b_trajectory.py"
EXTRA = {}


PREFIX = "aimo-h10b-"


def slug_of(m, suffix=None):
    return PREFIX + m.split("/")[-1].lower().replace(".", "-").replace("_", "-") + (SUFFIX if suffix is None else suffix)


def stage(slug, user, model_id, batch=4, resume_from=None, out_sub="h10b"):
    st = HERE / "staging" / slug
    if st.exists():
        shutil.rmtree(st)
    st.mkdir(parents=True)
    alias = model_id.split("/")[-1].replace(".", "_")
    resume = f"\"/kaggle/input/**/{alias}*.npz\"" if resume_from else "None"
    header = (HERE / HEADER).read_text(encoding="utf-8").replace("__MODEL__", model_id).replace("__BATCH__", str(batch)).replace("__RESUME__", resume)
    for k, v in EXTRA.items():
        header = header.replace(k, v)
    for k, v in (('__LEVELS__', 'Level 3,Level 4'), ('__TAG__', ''), ('__MAXSEQS__', '48'), ('__NOFF__', '3'), ('__NUNOFF__', '2')):
        header = header.replace(k, v)          # --extra 로 안 준 자리표시자 기본값
    def _find(f):
        for base in (HERE, HERE.parent):
            if (base / f).is_file():
                return base / f
        raise SystemExit(f"본문 파일 없음: {f}")
    bodies = [(f, _find(f).read_text(encoding="utf-8")) for f in BODY.split(",")]
    parts = [header] + [f"\n\n# ===== {f} (inlined) =====\n" + b for f, b in bodies]
    (st / "kernel.py").write_text("".join(parts), encoding="utf-8")
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [f"{user}/{resume_from}"] if resume_from else [], "model_sources": []}
    (st / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return st


def output(user, slug):
    import subprocess
    dst = EXP / "results" / "h10b" / slug
    dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(dst)], capture_output=True, text=True)
    print((r.stdout or "") + (r.stderr or ""), flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--suffix", default=""); ap.add_argument("--prefix", default="aimo-h10b-"); ap.add_argument("--header", default="h10b_kaggle.py"); ap.add_argument("--body", default="h10b_trajectory.py"); ap.add_argument("--extra", default="", help="__KEY__=value,__KEY2__=value2"); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--resume-from", default=None, help="이전 커널 slug (출력 npz 재사용)")
    ap.add_argument("--push-only", action="store_true"); ap.add_argument("--status-only", action="store_true"); ap.add_argument("--output-only", action="store_true")
    a = ap.parse_args()
    if not TOKEN.is_file():
        print(f"토큰 없음: {TOKEN}"); return
    ensure_cli(); user = username(); global SUFFIX, HEADER, BODY, EXTRA; SUFFIX = a.suffix; HEADER = a.header; BODY = a.body
    # 값 안에 쉼표가 들어갈 수 있다(예: __SOURCE__=aime,math5) → **| 가 있으면 | 로만 쪼갠다**.
    sep = "|" if "|" in a.extra else ","
    EXTRA = dict(kv.split("=", 1) for kv in a.extra.split(sep) if "=" in kv)
    global PREFIX; PREFIX = a.prefix
    for m in a.models:
        slug = slug_of(m)
        if a.status_only:
            print(slug, status(user, slug)); continue
        if a.output_only:
            output(user, slug); continue
        push(stage(slug, user, m, a.batch, a.resume_from)); print(f"  → https://www.kaggle.com/code/{user}/{slug}", flush=True)
    if a.push_only or a.status_only or a.output_only:
        return
    for m in a.models:
        slug = slug_of(m); print("최종:", slug, wait(user, slug)); output(user, slug)


if __name__ == "__main__":
    main()
