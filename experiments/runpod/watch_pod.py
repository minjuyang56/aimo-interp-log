"""떠 있는 파드 하나에 다시 붙어 5분마다 회수하고, 작업이 끝나면 **그 파드만** 종료한다.

## 왜 필요했나 (2026-10-12, #TEMP-SENS)
`launch.py` 를 `nohup ... &` 로 띄웠는데 **로컬 폴링 프로세스가 죽었다**(파드 생성 약 1시간 뒤
로그가 멈췄고 `Get-Process python` 이 아무것도 못 찾았다). 원격 작업은 `setsid nohup` 으로
떼어 놓았으므로 **파드에서는 계속 돌지만**, 폴링이 없으면
  (a) 5분 회수가 멈추고 (b) **수명 상한 자동 종료가 안 돌아 과금이 계속된다.**
`launch.py` 에는 재연결 옵션이 없고 `--reap` 은 **떠 있는 파드를 전부 종료**하므로 돌고 있는
작업을 죽인다 → 그래서 이 스크립트를 따로 둔다. 그 모듈의 함수를 import 해서 쓰고
**종료는 작업이 끝났다고 확인한 뒤에만** 한다.

## 끝났다는 판정 (둘 다 만족)
  1. 원격에 `<entry>` 파이썬 프로세스가 없다 (`pgrep -af`)
  2. `--done-glob` 이 가리키는 산출물이 원격 out 에 있다 (기본 `*labels.json`)
그 뒤 마지막 회수를 한 번 더 하고 종료한다. 수명 상한(`--max-hours`)을 넘기면 산출물을
회수한 뒤 종료한다 — 중간 산출물은 이미 5분마다 당겨 뒀으므로 선별은 잃지 않는다.

사용:
  .venv/Scripts/python.exe experiments/runpod/watch_pod.py --pod i6hxd8z437eas4 \
      --job labelgen32k --entry labelgen32k.py --max-hours 3.6
"""

import argparse
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def load_launch():
    spec = importlib.util.spec_from_file_location("lp", ROOT / "experiments" / "runpod" / "launch.py")
    lp = importlib.util.module_from_spec(spec)
    sys.modules["lp"] = lp
    spec.loader.exec_module(lp)
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True)
    ap.add_argument("--job", required=True, help="결과를 내려받을 디렉터리 이름 (results/runpod/<job>/<pod>)")
    ap.add_argument("--entry", default="labelgen32k.py", help="원격에서 돌고 있는 파이썬 엔트리")
    ap.add_argument("--done-glob", default="*labels.json")
    ap.add_argument("--max-hours", type=float, default=3.6, help="이 시간을 넘기면 회수 후 종료")
    ap.add_argument("--grace-min", type=float, default=15.0,
                    help="기동 유예(분). 이 시간 안에는 '작업 없음' 으로 종료하지 않는다 — "
                         "원격은 pip install vllm 부터 시작해 수 분간 엔트리가 안 보인다. "
                         "2026-10-12: 유예가 없어 payload 업로드 12초 뒤 '죽었다' 로 오판해 "
                         "멀쩡한 파드를 종료했다($0.22).")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--remote-out", default="/workspace/out")
    a = ap.parse_args()

    lp = load_launch()
    key, src = lp.api_key()
    if not key:
        raise SystemExit("RUNPOD_API_KEY 를 못 찾았다")
    lp.log(f"API 키 확인: 있음 · 길이 {len(key)} · 출처 {src}")
    import runpod as rp
    rp.api_key = key

    outdir = ROOT / "experiments" / "results" / "runpod" / a.job / a.pod
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    created = None

    while True:
        pods = rp.get_pods() or []
        me = [p for p in pods if p.get("id") == a.pod]
        if not me:
            lp.log(f"파드 {a.pod} 가 없다 — 과금은 멈췄다. 감시 종료")
            return
        pod = me[0]
        if created is None:
            created = pod.get("createdAt") or pod.get("lastStatusChange")
            lp.log(f"파드 {a.pod} · {pod.get('desiredStatus')} · "
                   f"{(pod.get('machine') or {}).get('gpuDisplayName')} · ${pod.get('costPerHr')}/h · "
                   f"생성 {created}")
        rt = pod.get("runtime") or {}
        ssh = [q for q in (rt.get("ports") or []) if q.get("privatePort") == 22]
        if not ssh:
            lp.log("  SSH 포트가 없다 — 60s 뒤 다시 본다")
            time.sleep(60)
            continue
        ip, port = ssh[0]["ip"], int(ssh[0]["publicPort"])

        rc, out = lp.ssh_once(ip, port,
                              f"pgrep -af '{a.entry}' | grep -v pgrep | head -2; echo '---OUT'; "
                              f"ls -1 {a.remote_out} 2>/dev/null")
        alive = bool(rc == 0 and out.split("---OUT")[0].strip())
        files = [x.strip() for x in out.split("---OUT")[-1].splitlines() if x.strip()] if rc == 0 else []
        age_h = (time.time() - t0) / 3600
        lp.log(f"  작업 {'돌고 있다' if alive else '안 보인다'} · 원격 산출 {len(files)}개 "
               f"{files} · 감시 경과 {age_h:.2f}h")

        got = lp.fetch_out(ip, port, a.remote_out, outdir, quiet=True)
        if got:
            have = sorted(p.name for p in outdir.rglob("*") if p.is_file())
            lp.log(f"  회수 OK · 로컬 {len(have)}개")
        # **원격 로그도 매번 당긴다.** `launch.py` 는 `/workspace/job.log` 에 찍고 그 폴링이
        # 죽었으므로 로컬에는 중간부터 없다. 파드를 종료하면 그 로그는 **영구히 사라지고**,
        # 변형 단계의 tok/s·총 토큰은 거기에만 있다(라벨 JSON 에는 집계만 들어간다).
        rcl, jl = lp.ssh_once(ip, port, "cat /workspace/job.log 2>/dev/null", timeout=90)
        if rcl == 0 and jl:
            (outdir / "job.log").write_text(jl, encoding="utf-8")
            lp.log(f"  job.log {len(jl)}자 저장 · 마지막 줄: {jl.strip().splitlines()[-1][:120]!r}")

        import fnmatch
        done_file = any(fnmatch.fnmatch(f, a.done_glob) for f in files)
        if (not alive) and done_file:
            lp.log("작업 종료 + 산출물 확인 → 마지막 회수 후 파드를 종료한다")
            lp.fetch_out(ip, port, a.remote_out, outdir)
            lp.terminate_verified(rp, a.pod)
            lp.log(f"종료 완료 · 결과 {outdir}")
            return
        if not alive and not done_file:
            if age_h * 60 < a.grace_min:
                lp.log(f"  작업이 아직 안 보인다 — 기동 유예 {a.grace_min:.0f}분 안이라 기다린다 "
                       f"(경과 {age_h * 60:.1f}분 · 원격은 pip install 부터 시작한다)")
                time.sleep(a.poll)
                continue
            lp.log("!! 작업이 안 보이는데 산출물도 없다 — 죽었을 수 있다. 한 번 더 확인하고 종료한다")
            time.sleep(180)
            rc2, out2 = lp.ssh_once(ip, port, f"pgrep -af '{a.entry}' | grep -v pgrep | head -2; "
                                              f"echo '---OUT'; tail -c 1500 /workspace/*.log 2>/dev/null")
            lp.log(f"  재확인: {out2[:900]}")
            lp.fetch_out(ip, port, a.remote_out, outdir)
            lp.terminate_verified(rp, a.pod)
            return
        if age_h > a.max_hours:
            lp.log(f"!! 감시 경과 {age_h:.2f}h > 상한 {a.max_hours}h → 회수 후 종료")
            lp.fetch_out(ip, port, a.remote_out, outdir)
            lp.terminate_verified(rp, a.pod)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
