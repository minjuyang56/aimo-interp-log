"""RunPod 런처 — 파드 생성 → 코드 업로드 → 실행 → 결과 회수 → **반드시 terminate**. (지휘 #RP)

왜 이 파일이 길고 방어적인가: 과금되는 자원을 띄우는 유일한 지점이다. 실수 하나가 돈이다.
그래서 세 겹으로 막는다.
  1. **하드캡 $65.** 원장(`spend.json`)에 기록된 누적 + 이번 파드의 **최악 비용**(수명 상한 × 시간당 가격)이
     캡을 넘으면 **생성 자체를 거부**한다. 실행 중 요금이 아니라 '최악의 경우'로 막는다.
  2. **수명 상한.** 감시 스레드가 `--max-hours` 에 도달하면 작업이 끝났든 안 끝났든 terminate 한다.
     SSH 가 끊기거나 작업이 멈춰도 과금이 무한히 가지 않는다.
  3. **종료 확인.** `finally` 에서 terminate 를 부르고, `get_pod` 가 사라질/멈출 때까지 **확인**한다.
     확인에 실패하면 큰 경고와 함께 파드 id 를 남긴다(사람이 콘솔에서 지울 수 있게).

**API 키는 절대 찍지 않는다.** `RUNPOD_API_KEY` 환경변수에서만 읽고, 로그·예외 메시지에 넣지 않는다.
키가 섞일 수 있는 경로(원장·로그)는 모두 .gitignore 에 들어 있다.

전송 방식: 전용 SSH 키쌍을 `experiments/runpod/.ssh/` 에 만들고(gitignore) 공개키를 파드 env `PUBLIC_KEY` 로
넣는다(RunPod 공식 이미지가 authorized_keys 에 설치한다). 그 뒤 scp 로 payload 를 올리고 결과를 내린다.
파드는 인터넷이 되므로 **모델·문제지는 파드가 직접 HF 에서 받는다** — payload 는 우리 코드 몇 KB 뿐이다.

사용:
  .venv/Scripts/python.exe experiments/runpod/launch.py --job h22_official33 --dry-run
  .venv/Scripts/python.exe experiments/runpod/launch.py --job h22_official33 --max-hours 1.5
  .venv/Scripts/python.exe experiments/runpod/launch.py --ledger          # 누적 비용만 보기
  .venv/Scripts/python.exe experiments/runpod/launch.py --reap            # 떠 있는 파드 전부 확인·종료
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LEDGER = HERE / "spend.json"            # 옛 단일 파일 (읽기만, 호환용)
LEDGER_DIR = HERE / "spend"              # run 하나당 파일 하나 (경쟁 없음)
SSH_DIR = HERE / ".ssh"
KEY = SSH_DIR / "id_ed25519"
JOBS = HERE / "jobs"
RESULTS = ROOT / "experiments" / "results" / "runpod"

#   **캡은 추정이 아니라 API 잔액에서 계산한다** (지휘 #DS-RELABEL, 2026-10-11).
#   새 캡 = 원장 누적 + RunPod 실측 잔액 - $3 여유. 실측: 누적 $84.49 + 잔액 $52.66 - $3 = $134.15.
#   잔액은 `myself { clientBalance }` 로 읽는다 — SDK 의 GraphQL 경로로만 된다(직접 POST 는 403).
#   캡은 '총 누적 지출 상한' 이므로 실제 남은 여유는 캡 - 누적 = 잔액 - $3 = $49.66 이다.
HARD_CAP_USD = 134.15      # 옛 값 95.0 (#RP-2, $100 충전 기준)
#   후보를 순서대로 **Docker Hub 에 실제로 있는지 확인**하고 첫 번째를 쓴다.
#   태그를 틀리면 파드가 못 떠도 과금이 시작되므로, 생성 **전에** 무료로 검증한다.
#   **RunPod 공식 이미지를 먼저 쓴다.** vLLM 공식 이미지(vllm/vllm-openai)는 ENTRYPOINT 가 API 서버이고
#   RunPod 의 ssh 설치 스크립트가 없어서 `PUBLIC_KEY` 를 넣어도 sshd 가 안 뜰 수 있다 — 그러면 접속도
#   못 하면서 과금만 된다. RunPod pytorch 이미지에 `pip install vllm` 하는 쪽이 2~4분 더 걸리는 대신 안전하다.
IMAGE_CANDIDATES = [
    "runpod/pytorch:0.7.1-dev-ubuntu2204-cu1281-torch271",
    "runpod/pytorch:0.7.1-dev-ubuntu2404-cu1281-torch271",
    "runpod/pytorch:0.7.2-dev-cu1281-torch260-ubuntu2404",
    "vllm/vllm-openai:v0.11.0",
]
WANT_GPUS = [("H100", 80), ("A100", 80)]        # 우선순위: H100 80GB 계열 전부 → A100 80GB 계열 전부
POD_WAIT_S = 900                 # 파드가 RUNNING + SSH 열릴 때까지 기다리는 상한
#   **호스트의 CUDA 드라이버를 걸러야 한다 (2026-10-10 에 걸렸다).** `pip install vllm` 이 끌어오는
#   torch 는 cu130 빌드라 드라이버가 12.8 인 호스트에서 "NVIDIA driver on your system is too old
#   (found version 12080)" 로 죽는다. 같은 GPU 라도 호스트마다 드라이버가 달라 복불복이다.
#   vllm 을 옛 버전으로 내리면 Olmo3 지원이 사라질 위험이 있으므로 **호스트를 고르는 쪽**을 택했다.
CUDA_OK = [v.strip() for v in os.environ.get("RP_CUDA", "12.9,13.0").split(",") if v.strip()]


def log(msg):
    print(f"[runpod {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def api_key():
    """키를 찾아 돌려준다. **값은 절대 출력·로그에 넣지 않는다** (지휘 #RP-3).

    `setx` 로 넣은 사용자 환경변수는 **이미 떠 있는 셸에는 반영되지 않는다**. 그래서 os.environ 에
    없으면 Windows 사용자 환경변수(HKCU\\Environment)를 직접 읽는다."""
    v = os.environ.get("RUNPOD_API_KEY", "").strip()
    if v:
        return v, "프로세스 환경변수"
    try:                                          # HKCU\Environment = setx 가 쓰는 곳
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            v = str(winreg.QueryValueEx(k, "RUNPOD_API_KEY")[0]).strip()
        if v:
            return v, "HKCU\\Environment (setx)"
    except Exception:
        pass
    return "", "없음"


# ---------------------------------------------------------------- 원장 (비용 추적)

#   **원장은 run 하나당 파일 하나다 (append-only).** 전에는 단일 JSON 배열이었는데, 런처를
#   병행 실행하면 각 프로세스가 전체를 읽어 자기 항목을 붙이고 통째로 다시 써서
#   **나중에 쓴 쪽이 앞 항목을 덮었다** — 2026-10-10 에 실제로 완료 기록 2건과 진행 중 파드 2건이
#   사라졌고, 하드캡이 **과소 집계된 원장 위에서** 계산되고 있었다(돈 안전장치의 핵이 깨진 것).
#   디렉터리에 파일로 흩어 놓으면 프로세스가 서로의 기록을 건드리지 않는다.

def _run_path(rec):
    """**파일명은 `started` 만으로 정한다.** pod_id 를 섞으면 생성 전(None)과 생성 후가 서로 다른
    파일로 가서 **한 run 이 파일 둘을 만들고 옛 쪽이 미정산으로 남아 이중 계산된다**
    (2026-10-10 에 x04ghnc52b6dll 이 $13.17 두 번으로 세어졌다)."""
    key = str(rec.get("started", "unknown")).replace(":", "-")
    return LEDGER_DIR / f"{key}.json"


def ledger_write(rec):
    """이 run 의 기록만 자기 파일에 쓴다. 다른 run 을 읽지도 쓰지도 않는다."""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _run_path(rec).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    except Exception as e:
        log(f"!! 원장 기록 실패({type(e).__name__}) — 비용 추적이 어긋난다")


def ledger_runs():
    runs = []
    if LEDGER_DIR.is_dir():
        for p in sorted(LEDGER_DIR.glob("*.json")):
            try:
                runs.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    if LEDGER.is_file():                         # 옛 단일 파일이 남아 있으면 같이 센다
        try:
            runs += json.loads(LEDGER.read_text(encoding="utf-8")).get("runs", [])
        except Exception:
            pass
    # pod_id 로 중복 제거. **정산된 항목(cost_usd 있음)을 우선**하고, 둘 다 미정산이면 더 큰
    # 최악 비용을 남긴다 — 복원 항목과 런처 항목이 같은 파드에 둘 다 있을 수 있고, 그때
    # 먼저 정렬된 쪽을 쓰면 실비를 버리고 추정을 쓰게 된다.
    best = {}
    for r in runs:
        k = r.get("pod_id") or r.get("started")
        cur = best.get(k)
        if cur is None:
            best[k] = r
            continue
        new_settled = r.get("cost_usd") is not None
        cur_settled = cur.get("cost_usd") is not None
        if new_settled and not cur_settled:
            best[k] = r
        elif new_settled == cur_settled and                 float(r.get("worst_usd", 0)) > float(cur.get("worst_usd", 0)):
            best[k] = r
    return list(best.values())


def spent_usd(runs=None):
    """지금까지의 지출. **끝나지 않은 run 은 최악 비용으로 센다**(낙관하지 않는다)."""
    tot = 0.0
    for r in (runs if runs is not None else ledger_runs()):
        if r.get("cost_usd") is not None:
            tot += float(r["cost_usd"])
        else:
            tot += float(r.get("worst_usd", 0.0))
    return tot


def ledger_print():
    runs = sorted(ledger_runs(), key=lambda r: r.get("started", ""))
    s = spent_usd(runs)
    log(f"하드캡 ${HARD_CAP_USD:.2f} · 누적 ${s:.2f} · 남음 ${HARD_CAP_USD - s:.2f} · run {len(runs)}건")
    for r in runs:
        state = (f"${r['cost_usd']:.2f}" if r.get("cost_usd") is not None
                 else f"미확정(최악 ${r.get('worst_usd', 0):.2f})")
        log(f"  {r.get('started')} · {r.get('job')} · {r.get('gpu')} · "
            f"{r.get('hours', 0):.2f}h · {state} · pod {r.get('pod_id')}")


# ---------------------------------------------------------------- SSH 키

def ensure_key():
    """전용 키쌍을 만든다(없을 때만). 개인키는 .gitignore 안에 있고 절대 출력하지 않는다."""
    SSH_DIR.mkdir(parents=True, exist_ok=True)
    pub = KEY.with_suffix(".pub")
    if not pub.is_file():
        if not shutil.which("ssh-keygen"):
            raise SystemExit("ssh-keygen 이 없다 — Git for Windows 의 usr/bin 을 PATH 에 넣어야 한다")
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "aimo-runpod",
                        "-f", str(KEY)], check=True, capture_output=True)
        log(f"SSH 키쌍 생성: {KEY.relative_to(ROOT)} (개인키는 커밋되지 않는다)")
    return pub.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------- GPU 선택·가격

def image_exists(ref, timeout=25):
    """태그가 실제로 있는지 **Docker Registry v2 API** 로 확인한다. 못 확인하면 None.
    (hub.docker.com 의 웹 API 는 이 네트워크에서 403 이라 못 쓴다 — 레지스트리 쪽은 익명 토큰으로 된다.)"""
    import json as _json
    import urllib.error
    import urllib.parse
    import urllib.request
    repo, _, tag = ref.partition(":")
    tag = tag or "latest"
    if "/" not in repo:
        repo = "library/" + repo
    try:
        q = urllib.parse.urlencode({"service": "registry.docker.io",
                                    "scope": f"repository:{repo}:pull"})
        with urllib.request.urlopen(f"https://auth.docker.io/token?{q}", timeout=timeout) as r:
            token = _json.loads(r.read())["token"]
        req = urllib.request.Request(
            f"https://registry-1.docker.io/v2/{repo}/manifests/{urllib.parse.quote(tag)}",
            headers={"Authorization": f"Bearer {token}",
                     "Accept": ("application/vnd.oci.image.index.v1+json,"
                                "application/vnd.docker.distribution.manifest.list.v2+json,"
                                "application/vnd.docker.distribution.manifest.v2+json")})
        req.get_method = lambda: "HEAD"
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return False if e.code in (404, 401) else None
    except Exception:
        return None


def pick_image(preferred=None):
    """쓸 수 있는 이미지를 고른다. 확인 불가(네트워크 문제)면 첫 후보를 그대로 쓰고 경고한다."""
    for ref in ([preferred] if preferred else []) + IMAGE_CANDIDATES:
        got = image_exists(ref)
        if got is True:
            log(f"이미지 확인됨: {ref}")
            return ref
        log(f"  이미지 {ref} → {'없음' if got is False else '확인 불가'}")
    raise SystemExit("쓸 수 있는 이미지를 못 찾았다 — --image 로 직접 지정할 것")


def pick_gpu(runpod, want=None):
    """살아 있는 가격표에서 GPU 를 고른다. **가격을 상수로 박지 않는다** — 바뀌면 예산이 틀어진다.

    `get_gpus()` 는 id/이름/메모리만 주고 **가격이 없다**. 가격은 `get_gpu(id)` 에만 있어서
    후보만 추려 하나씩 조회한다. 예산 계산은 **비싼 쪽(securePrice)** 으로 한다 — 어느 클라우드에
    잡힐지 미리 알 수 없으므로 싼 쪽으로 계산하면 캡이 뚫릴 수 있다.
    반환 [(gpu_id, 표시이름, 메모리GB, 예산가, community가), …] — **싼 순**. 용량이 없으면
    호출자가 다음 후보로 넘어간다(RunPod 는 특정 GPU 가 꽉 차 있는 일이 흔하다)."""
    listing, out = runpod.get_gpus(), []
    for key, need_mem in (want or WANT_GPUS):
        cands = [g for g in listing
                 if key.lower() in f"{g.get('displayName', '')} {g.get('id', '')}".lower()
                 and int(g.get("memoryInGb") or 0) >= need_mem]
        priced = []
        for g in cands:
            try:
                d = runpod.get_gpu(g["id"]) or {}
            except Exception:
                continue
            sec, com = d.get("securePrice"), d.get("communityPrice")
            if sec or com:
                budget_price = float(sec or com)              # 보수적으로 비싼 쪽
                priced.append((g["id"], str(d.get("displayName") or g["id"]),
                               int(d.get("memoryInGb") or 0), budget_price,
                               float(com) if com else None))
        out += sorted(priced, key=lambda r: r[3])
    if not out:
        raise SystemExit(f"원하는 GPU 가 없다. 후보: "
                         f"{[(g.get('displayName'), g.get('memoryInGb')) for g in listing][:20]}")
    log("GPU 후보 (싼 순, 용량 없으면 다음으로 넘어간다):")
    for gid, name, mem, price, com in out:
        log(f"  {name} {mem}GB · 예산가 ${price:.3f}/h"
            + (f" (community ${com:.3f}/h)" if com and com < price else "") + f" · id {gid}")
    return out


# ---------------------------------------------------------------- payload

def make_payload(job):
    """작업에 필요한 파일만 tar.gz 로 싼다. 모델·데이터는 파드가 직접 받으므로 넣지 않는다."""
    spec_path = JOBS / f"{job}.json"
    if not spec_path.is_file():
        raise SystemExit(f"작업 정의가 없다: {spec_path.relative_to(ROOT)} "
                         f"(있는 것: {[p.stem for p in JOBS.glob('*.json')]})")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out = HERE / f"_payload_{job}.tar.gz"
    with tarfile.open(out, "w:gz") as t:
        for rel in spec["files"]:
            src = ROOT / rel
            if not src.exists():
                raise SystemExit(f"payload 에 넣을 파일이 없다: {rel}")
            t.add(src, arcname=Path(rel).name if spec.get("flatten") else rel)
        t.add(JOBS / spec["entry"], arcname=spec["entry"])
    log(f"payload {out.name} · {out.stat().st_size / 1024:.0f} KB · 파일 {len(spec['files']) + 1}개")
    return out, spec


# ---------------------------------------------------------------- 파드 수명

def wait_ssh(runpod, pod_id, timeout=POD_WAIT_S):
    """RUNNING + SSH 포트 매핑이 생길 때까지 기다린다. 반환 (ip, port)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = runpod.get_pod(pod_id) or {}
        rt = p.get("runtime") or {}
        for pr in (rt.get("ports") or []):
            if int(pr.get("privatePort", 0)) == 22 and pr.get("ip") and pr.get("publicPort"):
                log(f"SSH 준비 {pr['ip']}:{pr['publicPort']} ({time.time() - t0:.0f}s)")
                return pr["ip"], int(pr["publicPort"])
        log(f"  대기… 상태 {p.get('desiredStatus')} ({time.time() - t0:.0f}s)")
        time.sleep(15)
    raise TimeoutError(f"{timeout}s 안에 SSH 가 안 열렸다")


def terminate_verified(runpod, pod_id, tries=10):
    """terminate 를 부르고 **사라졌는지 확인한다**. 확인 실패는 돈이 새는 것이므로 크게 경고한다."""
    for i in range(tries):
        try:
            runpod.terminate_pod(pod_id)
        except Exception as e:
            log(f"  terminate 호출 실패({i}): {type(e).__name__}")
        time.sleep(6)
        try:
            p = runpod.get_pod(pod_id)
        except Exception:
            p = None
        if not p or str(p.get("desiredStatus", "")).upper() in ("TERMINATED", "EXITED"):
            log(f"종료 확인됨: pod {pod_id}")
            return True
    print("\n" + "!" * 70, flush=True)
    print(f"!! 종료 확인 실패 — pod {pod_id} 가 아직 살아 있을 수 있다. 콘솔에서 직접 지울 것.", flush=True)
    print("!" * 70 + "\n", flush=True)
    return False


def ssh_base(ip, port):
    return ["ssh", "-i", str(KEY), "-p", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
            "-o", "ConnectTimeout=30", f"root@{ip}"]


def ssh_once(ip, port, cmd, timeout=120):
    """짧은 ssh 한 번. (rc, 출력) 을 돌려준다. 끊겨도 예외를 던지지 않는다."""
    try:
        r = subprocess.run(ssh_base(ip, port) + [cmd], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return 255, f"[ssh 실패 {type(e).__name__}]"


def fetch_out(ip, port, remote_out, outdir, quiet=False):
    """원격 산출물 디렉터리를 내려받는다. **폴링 중에도 주기적으로** 부른다 —
    전에는 작업이 끝난 뒤에만 받아서, 감시기가 파드를 끊으면 파드에 있던 중간 산출물
    (선별 결과 `*_screen.json` 등)을 **같이 잃었다**(2026-10-10 DeepSeek off80 에서 겪었다:
    선별 3,097초가 끝나 있었는데 변형 중 수명 상한에 걸려 전부 날아갔다)."""
    outdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["scp", "-i", str(KEY), "-P", str(port), "-r",
                        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=30",
                        f"root@{ip}:{remote_out}", str(outdir)],
                       capture_output=True, text=True)
    if not quiet:
        log(f"산출물 회수 {'OK' if r.returncode == 0 else '실패(아직 없을 수 있다)'} → {outdir.name}")
    return r.returncode == 0


def run_remote(ip, port, cmd, log_path, timeout, poll=20, remote_out=None, outdir=None,
               fetch_every=15):
    """원격 작업을 **ssh 세션에서 분리해** 돌리고 로그만 주기적으로 당겨 온다.

    왜 이렇게 바꿨나 (2026-10-10 에 실제로 잃었다): 전에는 `ssh <cmd>` 한 줄의 stdout 을 그대로
    따라 읽었다. 그러면 **ssh 연결이 끊기는 순간 원격 작업도 죽는다** — 조각판 ① 이 32분 만에
    원격 종료 코드 **255**(= ssh 연결 끊김, 파이썬 오류가 아니다)로 끝나 조각 2~6 을 잃었다.
    ②③ 는 몇 시간짜리라 이 구조로는 반드시 또 잃는다.

    이제: setsid nohup 으로 띄우고 pid 를 받아 둔 뒤, 짧은 ssh 를 반복해 (a) 로그의 새 바이트만
    당기고 (b) pid 생존을 확인한다. 연결이 한 번 끊겨도 다음 폴링에서 이어진다."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 스크립트는 **base64 로 실어 보낸다.** 따옴표·줄바꿈·$ 가 섞인 명령을 ssh 인자로 넘기면
    # 중간 셸이 한 번 더 해석해 깨진다(heredoc 도 같은 이유로 못 쓴다).
    b64 = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
    rc, out = ssh_once(ip, port, "mkdir -p /workspace && echo " + b64
                       + " | base64 -d > /workspace/job.sh && chmod +x /workspace/job.sh && echo wrote")
    if "wrote" not in out:
        raise RuntimeError(f"원격 스크립트 작성 실패: {out[:200]}")
    # `sleep 2; pgrep` 를 붙였더니 ssh 채널이 안 닫혀 120초 타임아웃이 났다(작업은 이미 떠 있었다).
    # 기동 확인은 아래 폴링이 job.log 를 보고 하므로 여기서는 **떼어내고 바로 끊는다**.
    detach = ("cd /workspace && rm -f job.log job.rc && touch job.log && "
              "setsid nohup bash -lc 'bash /workspace/job.sh > /workspace/job.log 2>&1; "
              "echo $? > /workspace/job.rc' < /dev/null > /dev/null 2>&1 &")
    # **기동을 확인한다 (2026-10-18 에 30분을 잃었다).** 끝의 `&` 로 ssh 를 바로 끊으면
    # setsid 가 자리를 잡기 전에 세션이 닫히는 **경쟁 상태**가 생긴다 — 실제로 `job.log` 조차
    # 만들어지지 않았고(= 명령이 아예 안 돌았다) 폴링은 '조용한 생성 중' 과 구분하지 못해
    # 파드가 25분을 빈손으로 돌았다. 그래서 **별도의 짧은 ssh 로 기동을 확인하고, 안 떴으면
    # 한 번 더 띄운다.** (같은 ssh 호출에 `sleep; pgrep` 를 붙이면 채널이 안 닫혀 타임아웃이 난다.)
    for attempt in (1, 2):
        ssh_once(ip, port, detach, timeout=60)
        log(f"원격 작업 분리 실행 (시도 {attempt}) — 기동 확인 중")
        alive = False
        for _ in range(8):
            time.sleep(6)
            rc, o = ssh_once(ip, port,
                             "pgrep -f job.sh >/dev/null && echo alive || "
                             "(test -s /workspace/job.log && echo alive || echo dead)",
                             timeout=30)
            if "alive" in o:
                alive = True
                break
        if alive:
            log("  기동 확인됨 — ssh 가 끊겨도 계속 돈다")
            break
        log("  !! 기동 확인 실패 — 다시 띄운다" if attempt == 1 else
            "  !! 두 번 모두 기동 실패")
    else:
        raise RuntimeError("원격 작업이 기동하지 않았다 — 파드를 종료하고 다시 시도할 것")

    t0, off, idle, ticks = time.time(), 0, 0, 0
    with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
        while True:
            if time.time() - t0 > timeout:
                raise TimeoutError("원격 작업 시간 초과")
            # **오프셋은 원격 파일 크기로 센다.** 받은 문자열의 로컬 바이트 길이로 세면 줄바꿈 변환
            # 때문에 조금씩 어긋나 로그가 밀린다.
            rc0, size = ssh_once(ip, port, "stat -c %s /workspace/job.log 2>/dev/null || echo 0")
            if rc0 != 0:
                idle += 1                     # **ssh 실패만** 센다 (생성 중 출력이 조용한 건 정상이다)
            else:
                idle = 0
                have = int(size.strip() or 0)
                if have > off:
                    rc1, chunk = ssh_once(
                        ip, port, f"tail -c +{off + 1} /workspace/job.log 2>/dev/null | head -c 400000")
                    if rc1 == 0 and chunk:
                        lf.write(chunk)
                        lf.flush()
                        for line in chunk.splitlines():
                            if any(m in line for m in ("[단계]", "ERROR", "Traceback", "AssertionError",
                                                       "완료", "tok/s", "AUC")):
                                print("   | " + line.rstrip()[:160], flush=True)
                        off = min(have, off + 400000)
            ticks += 1
            if remote_out and outdir and ticks % fetch_every == 0:
                fetch_out(ip, port, remote_out, outdir, quiet=True)   # 5분마다 중간 산출물 확보
            rc2, done = ssh_once(ip, port, "cat /workspace/job.rc 2>/dev/null")
            if rc2 == 0 and done.strip().isdigit():
                rc3, rest = ssh_once(ip, port, f"tail -c +{off + 1} /workspace/job.log 2>/dev/null")
                if rc3 == 0 and rest:
                    lf.write(rest)
                    lf.flush()
                return int(done.strip())
            if idle > 30:                     # ssh 가 10분 넘게 연달아 실패하면 포기(파드는 finally 가 정리)
                raise RuntimeError("ssh 가 10분 넘게 응답하지 않는다")
            time.sleep(poll)


# ---------------------------------------------------------------- 본체

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--max-hours", type=float, default=1.5, help="파드 수명 상한 (이 시간에 강제 terminate)")
    ap.add_argument("--image", default=None, help="비우면 IMAGE_CANDIDATES 에서 존재 확인된 첫 번째를 쓴다")
    ap.add_argument("--gpu", default=None, help="예: H100 (기본은 H100 80GB → A100 80GB 순)")
    ap.add_argument("--upload", action="append", default=[], metavar="LOCAL",
                    help="payload 과 별도로 /workspace 에 올릴 파일 (여러 번 가능). "
                         "수명 상한에 걸린 실행의 *_screen.json 을 이어받을 때 쓴다.")
    ap.add_argument("--job-env", action="append", default=[], metavar="KEY=VAL",
                    help="원격 작업에 넘길 환경변수 (여러 번 쓸 수 있다). 예: --job-env H22_MODEL=Skywork/...")
    ap.add_argument("--dry-run", action="store_true", help="API 를 부르지 않고 계획·예산만 찍는다")
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--reap", action="store_true", help="떠 있는 파드를 전부 확인·종료")
    ap.add_argument("--reconcile", action="store_true",
                    help="죽은 파드의 최악 비용 예약을 실비로 정산한다. 런처 밖에서(API 로) 파드를 "
                         "끊으면 그 프로세스가 정산 기록을 못 써서 예약이 영구히 남는다 — "
                         "그러면 하드캡이 쓰지도 않은 돈을 잡고 있게 된다.")
    a = ap.parse_args()

    if a.ledger:
        ledger_print()
        return

    key, src = api_key()
    if key:                                      # **존재와 길이만** 찍는다 — 값은 절대 안 찍는다
        log(f"API 키 확인: 있음 · 길이 {len(key)} · 출처 {src}")
    if a.dry_run:
        log("DRY RUN — API 호출 없음")
    elif not key:
        raise SystemExit("RUNPOD_API_KEY 를 못 찾았다(프로세스 환경변수·HKCU\\Environment 둘 다 없음). "
                         "setx 로 넣었다면 값이 저장됐는지 확인할 것. 이 스크립트는 키를 출력하지 않는다")

    runpod = None
    if not a.dry_run:
        import runpod as _rp
        _rp.api_key = key
        runpod = _rp

    if a.reconcile:
        live = {p.get("id") for p in (runpod.get_pods() or [])}
        now = datetime.now(timezone.utc)
        n = 0
        for r in ledger_runs():
            if r.get("cost_usd") is not None or not r.get("pod_id"):
                continue
            if r["pod_id"] in live:
                log(f"  살아 있음, 건너뜀: {r['pod_id']}")
                continue
            try:
                st = datetime.fromisoformat(str(r["started"]).replace("Z", "+00:00"))
                if st.tzinfo is None:
                    st = st.replace(tzinfo=timezone.utc)
                hours = min(float(r.get("max_hours", 99)), max(0.0, (now - st).total_seconds() / 3600))
            except Exception:
                hours = float(r.get("max_hours", 0))
            r["hours"] = round(hours, 3)
            r["cost_usd"] = round(hours * float(r.get("price_per_h", 0)), 2)
            r["_note"] = (r.get("_note", "") + " · --reconcile 로 정산(런처 밖 종료)").strip(" ·")
            ledger_write(r)
            log(f"  정산 {r['pod_id']} · {hours:.2f}h · ${r['cost_usd']:.2f} "
                f"(예약 ${r.get('worst_usd', 0):.2f} 해제)")
            n += 1
        log(f"정산 {n}건 · 누적 ${spent_usd():.2f} / ${HARD_CAP_USD:.2f}")
        return

    if a.reap:
        pods = runpod.get_pods() or []
        if not pods:
            log("떠 있는 파드 없음")
        for p in pods:
            log(f"파드 {p.get('id')} · {p.get('name')} · 상태 {p.get('desiredStatus')} → 종료 시도")
            terminate_verified(runpod, p["id"])
        return

    if not a.job:
        raise SystemExit("--job 을 줘야 한다")

    # --- 예산 검사 (생성 전에)
    want = [(a.gpu.upper(), 80)] if a.gpu else None
    if a.dry_run:
        cands = [("DRY", (a.gpu or "H100") + " 80GB", 80, 2.99, None)]
        log(f"GPU(가정) {cands[0][1]} · ${cands[0][3]:.3f}/h — 실제 가격은 실행 시 API 에서 읽는다")
    else:
        cands = pick_gpu(runpod, want)
    gid, gname, _mem, price, _com = cands[0]     # 예산은 **가장 비싼 후보**로 잡아야 안전하다
    price = max(c[3] for c in cands)
    worst = price * a.max_hours
    have = spent_usd()
    log(f"예산: 누적 ${have:.2f} + 이번 최악 ${worst:.2f} = ${have + worst:.2f} / 캡 ${HARD_CAP_USD:.2f}")
    if have + worst > HARD_CAP_USD:
        raise SystemExit(f"거부: 하드캡 ${HARD_CAP_USD:.2f} 초과 (최악 기준). "
                         f"--max-hours 를 {max(0.1, (HARD_CAP_USD - have) / price):.2f} 이하로 줄일 것")

    image = pick_image(a.image)          # 생성 전에 태그 존재를 확인한다(무료)
    payload, spec = make_payload(a.job)
    pub = ensure_key()
    # 작업 환경변수는 원격 명령 앞에 export 로 붙인다 (API 키는 절대 넘기지 않는다)
    bad = [kv for kv in a.job_env if "=" not in kv or kv.split("=")[0].strip().upper().startswith("RUNPOD")]
    if bad:
        raise SystemExit(f"--job-env 형식 오류이거나 넘길 수 없는 키: {bad}")
    exports = "".join(f"export {kv.split('=', 1)[0].strip()}='{kv.split('=', 1)[1]}'; " for kv in a.job_env)
    remote_cmd = "cd /workspace && tar xzf payload.tar.gz && " + exports + spec["cmd"]
    log(f"작업 {a.job} · entry {spec['entry']} · 수명 상한 {a.max_hours}h")

    if a.dry_run:
        log("DRY RUN 끝 — 실제 생성·업로드·실행·종료는 하지 않았다")
        log(f"  이미지 {image}")
        log(f"  원격 명령: {remote_cmd}")
        return

    run_rec = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "job": a.job,
               "gpu": gname, "price_per_h": price, "worst_usd": round(worst, 2),
               "max_hours": a.max_hours, "pod_id": None, "hours": 0.0, "cost_usd": None}
    ledger_write(run_rec)                        # **생성 전에** 최악 비용으로 적어 둔다

    pod_id, t_start, stop = None, time.time(), threading.Event()
    try:
        pod, last = None, None
        for cg, cname, _m, cprice, _c in cands:
            try:
                pod = runpod.create_pod(
                    name=f"aimo-{a.job}", image_name=image, gpu_type_id=cg,
                    cloud_type="ALL", support_public_ip=True, start_ssh=True,
                    gpu_count=1, volume_in_gb=0,
                    container_disk_in_gb=int(spec.get("disk_gb", 120)),
                    ports="22/tcp", env={"PUBLIC_KEY": pub},
                    allowed_cuda_versions=CUDA_OK)
            except Exception as e:
                last = f"{cname}: {str(e)[:100]}"
                log(f"  {cname} 용량 없음 → 다음 후보")
                continue
            gname, price = cname, cprice
            run_rec["gpu"], run_rec["price_per_h"] = cname, cprice
            break
        if pod is None:
            raise SystemExit(f"모든 GPU 후보에 용량이 없다. 마지막 오류: {last}")
        pod_id = pod["id"]
        run_rec["pod_id"] = pod_id
        ledger_write(run_rec)
        log(f"파드 생성 {pod_id} · {gname} · ${price:.3f}/h")

        # **감시기보다 먼저 정의한다** — 감시기가 먼저 울리면 rundir 가 없어 NameError 가 된다.
        rundir = RESULTS / a.job / pod_id
        conn = {}                                # 감시기가 ip/port 를 보고 산출물을 챙길 수 있게

        def watchdog():                          # 수명 상한 — 작업이 멈춰도 과금을 끊는다
            if stop.wait(a.max_hours * 3600):
                return
            log(f"수명 상한 {a.max_hours}h 도달 → **끊기 전에 산출물 먼저 챙긴다**")
            if conn.get("ip"):                   # 안 챙기면 파드에 있던 중간 결과를 같이 잃는다
                try:
                    fetch_out(conn["ip"], conn["port"], spec.get("out", "/workspace/out"), rundir)
                except Exception as e:
                    log(f"  마지막 회수 실패: {type(e).__name__}")
            log("강제 종료")
            terminate_verified(runpod, pod_id)

        threading.Thread(target=watchdog, daemon=True).start()

        ip, port = wait_ssh(runpod, pod_id)
        conn["ip"], conn["port"] = ip, port
        # **파드별로 분리한다.** 같은 작업을 병행 실행하면 한 디렉터리를 두 파드가 써서
        # 나중 scp 가 앞 결과를 덮고 remote.log 가 뒤섞인다(실제로 겪었다).
        logf = rundir / "remote.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        # 이미지에 /workspace 가 없을 수 있다(runpod/pytorch 는 없었다) → 먼저 만든다.
        subprocess.run(ssh_base(ip, port) + ["mkdir -p /workspace/out && echo workspace-ok"],
                       check=True, capture_output=True, text=True, timeout=120)
        subprocess.run(["scp", "-i", str(KEY), "-P", str(port), "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                        str(payload), f"root@{ip}:/workspace/payload.tar.gz"], check=True)
        for up in a.upload:                  # run 마다 다른 파일(이전 선별 결과 등)을 따로 올린다
            src = Path(up)
            if not src.is_file():
                raise SystemExit(f"--upload 파일이 없다: {up}")
            subprocess.run(["scp", "-i", str(KEY), "-P", str(port), "-o", "StrictHostKeyChecking=no",
                            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                            str(src), f"root@{ip}:/workspace/{src.name}"], check=True)
            log(f"추가 업로드 {src.name} → /workspace/{src.name}")
        log("payload 업로드 완료 → 원격 실행 시작")
        rc = run_remote(ip, port, remote_cmd, logf, timeout=a.max_hours * 3600,
                        remote_out=spec.get('out', '/workspace/out'), outdir=rundir)
        log(f"원격 종료 코드 {rc}")

        fetch_out(ip, port, spec.get("out", "/workspace/out"), rundir)
        log(f"결과 회수 → {rundir.relative_to(ROOT)}")
    finally:
        stop.set()
        hours = (time.time() - t_start) / 3600.0
        run_rec["hours"] = round(hours, 3)
        run_rec["cost_usd"] = round(hours * price, 2)
        ledger_write(run_rec)
        if pod_id:
            terminate_verified(runpod, pod_id)
        log(f"이번 실행 {hours:.2f}h · ${run_rec['cost_usd']:.2f} · 누적 ${spent_usd():.2f} / ${HARD_CAP_USD:.2f}")


if __name__ == "__main__":
    sys.exit(main())
