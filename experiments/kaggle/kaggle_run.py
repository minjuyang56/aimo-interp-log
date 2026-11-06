"""Kaggle 자동화 — 커널 올리고, 끝날 때까지 폴링하고, 결과를 내려받는다.

전제: ~/.kaggle/kaggle.json 이 있어야 한다 ([author]가 직접 넣음). 이 스크립트는 그 파일을 읽지도 출력하지도 않는다.
      kaggle CLI가 알아서 읽는다.

순서
  1. pip install kaggle (없으면)
  2. staging/ 에 gen_labels.py + gen_labels_kaggle.py + kernel-metadata.json 구성
     (GPU on · 인터넷 on · 비공개 · script 커널)
  3. kaggle kernels push
  4. kaggle kernels status 를 60초마다 폴링 (complete / error / cancel 까지)
  5. kaggle kernels output → results/gen_labels/kaggle_<slug>/  (summary.csv · report.txt · generations.jsonl · log)

사용:  .venv/Scripts/python.exe experiments/kaggle/kaggle_run.py [--slug aimo-gen-labels-run1] [--push-only] [--status-only] [--output-only]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
TOKEN = Path(os.path.expanduser("~/.kaggle/kaggle.json"))


def sh(*args, capture=True):
    r = subprocess.run(list(args), capture_output=capture, text=True)
    if capture:
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    return r.returncode, ""


def ensure_cli():
    # venv 의 Scripts 는 PATH 에 없다 → sys.executable 옆을 먼저 보고 PATH 에 넣는다.
    bin_dir = Path(sys.executable).parent
    if (bin_dir / "kaggle.exe").is_file() or (bin_dir / "kaggle").is_file():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("kaggle") is None:
        print("kaggle CLI 설치 중…", flush=True)
        # 이 venv 는 uv 로 만들어 pip 가 없다. pip → uv pip 순으로 시도한다.
        if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"]).returncode != 0:
            subprocess.run(["uv", "pip", "install", "--python", sys.executable, "-q", "kaggle"], check=True)
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    code, out = sh("kaggle", "--version")
    print("kaggle CLI:", out.strip().splitlines()[-1] if out.strip() else code, flush=True)


def username():
    # 토큰 파일에서 username만 읽는다. key는 읽지 않는다. 출력도 username만.
    d = json.loads(TOKEN.read_text(encoding="utf-8"))
    return d["username"]


def stage(slug, user):
    st = HERE / "staging"
    if st.exists():
        shutil.rmtree(st)
    st.mkdir()
    # Kaggle script 커널은 code_file 하나만 올린다 → 헤더(설정) + gen_labels.py 소스를 한 파일로 합친다.
    header = (HERE / "gen_labels_kaggle.py").read_text(encoding="utf-8")
    body = (EXP / "gen_labels.py").read_text(encoding="utf-8")
    (st / "kernel.py").write_text(header + "\n\n# ===== gen_labels.py (inlined) =====\n" + body, encoding="utf-8")
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (st / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"staging: {st}  ({', '.join(p.name for p in st.iterdir())})", flush=True)
    return st


def push(st):
    # 올리기 전에 **조립된 커널을 파싱한다.** 헤더 자리표시자가 잘못 치환되면(따옴표 중복 등) 커널이
    # 1분 만에 SyntaxError 로 죽고 GPU 시간만 날린다 — p13 에서 실제로 한 번 날렸다.
    import ast
    src = (Path(st) / "kernel.py").read_text(encoding="utf-8")
    try:
        ast.parse(src)
    except SyntaxError as e:
        raise SystemExit(f"파서 체크 실패 {st}: 줄 {e.lineno} {e.msg}\n  {(e.text or '').strip()[:160]}")
    # **외부 심볼 점검.** 커널 본문은 다른 파일에서 inline 되는 이름을 `# noqa: F821` 로 표시해 둔다.
    # 그 이름이 (a) 어디에도 없거나 (b) main() 호출보다 뒤에 정의되면 런타임 NameError 가 난다.
    # H20-M 1차 push 가 이 경우였다: vLLM 로드까지 끝낸 뒤 edit_rename 이 없어 죽어 모델당 4분을 날렸다.
    pos_main = src.rfind("\n    main()")
    missing = []
    for ln in (l for l in src.splitlines() if "noqa: F821" in l):
        code_part = ln.split("#")[0]               # 주석(noqa 표시 자체)은 보지 않는다
        # 점 뒤의 이름(메서드 호출)은 제외 — tok.apply_chat_template 같은 건 외부 심볼이 아니다
        names = set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", code_part))
        names |= set(re.findall(r"(?<![\w.])([A-Z][A-Z_0-9]{2,})\b", code_part))
        import builtins as _b
        import keyword as _kw
        names -= set(dir(_b)) | set(_kw.kwlist)        # 내장·키워드는 외부 심볼이 아니다(sum, else …)
        for name in names:
            m = re.search(rf"^(def {name}\b|{name}\s*=)", src, re.M)
            if m is None:
                missing.append(f"{name}: 정의 없음")
            elif pos_main > 0 and m.start() > pos_main:
                missing.append(f"{name}: main() 보다 뒤에 정의 — --body 순서를 바꿀 것")
    if missing:
        raise SystemExit("외부 심볼 점검 실패 (" + str(st) + "):\n  " + "\n  ".join(sorted(set(missing))))
    print(f"파서 체크 OK ({len(src.splitlines())}줄) · 외부 심볼 점검 OK", flush=True)
    code, out = sh("kaggle", "kernels", "push", "-p", str(st))
    print(out.strip(), flush=True)
    if code != 0:
        raise SystemExit("push 실패")


def status(user, slug):
    code, out = sh("kaggle", "kernels", "status", f"{user}/{slug}")
    return out.strip()


def wait(user, slug, every=60, max_hours=12):
    t0 = time.time()
    while True:
        s = status(user, slug)
        print(f"  [{time.strftime('%H:%M')}] {s}", flush=True)
        low = s.lower()
        if "complete" in low or "error" in low or "cancel" in low:
            return s
        if time.time() - t0 > max_hours * 3600:
            return "timeout"
        time.sleep(every)


def output(user, slug):
    dst = EXP / "results" / "gen_labels" / f"kaggle_{slug}"
    dst.mkdir(parents=True, exist_ok=True)
    code, out = sh("kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(dst))
    print(out.strip(), flush=True)
    for f in ("report.txt", "summary.csv"):
        hit = list(dst.rglob(f))
        if hit:
            print(f"--- {f} ---"); print(hit[0].read_text(encoding="utf-8")[:2000])
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="aimo-gen-labels-run1")
    ap.add_argument("--push-only", action="store_true")
    ap.add_argument("--status-only", action="store_true")
    ap.add_argument("--output-only", action="store_true")
    a = ap.parse_args()
    if not TOKEN.is_file():
        print(f"토큰 없음: {TOKEN}  — [author]가 kaggle.json을 넣으면 다시 실행"); return
    ensure_cli()
    user = username()
    print(f"Kaggle 사용자: {user} · 커널: {a.slug}", flush=True)
    if a.status_only:
        print(status(user, a.slug)); return
    if a.output_only:
        output(user, a.slug); return
    st = stage(a.slug, user)
    push(st)
    if a.push_only:
        return
    final = wait(user, a.slug)
    print("최종 상태:", final)
    output(user, a.slug)


if __name__ == "__main__":
    main()
