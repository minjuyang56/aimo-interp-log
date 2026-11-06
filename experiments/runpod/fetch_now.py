"""돌고 있는 파드의 산출물을 **지금 당겨온다**. (지휘 #LG-RP 안전장치)

왜 필요한가: 주기 회수 패치 전에 띄운 런처는 작업이 끝날 때만 결과를 받는다. 그 파드가 수명
상한에 걸리면 파드에 있던 중간 산출물(선별 `*_screen.json` 등)까지 **같이 날아간다** —
2026-10-10 DeepSeek off80 에서 선별 3,097초를 그렇게 잃었다($6.04 전액 손실).
그래서 '지금 당겨오기' 를 손으로 부를 수 있게 따로 뺐다. 파드를 건드리지 않고 읽기만 한다.

사용:  .venv/Scripts/python.exe experiments/runpod/fetch_now.py            # 떠 있는 파드 전부
       .venv/Scripts/python.exe experiments/runpod/fetch_now.py <pod_id>  # 하나만
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launch import RESULTS, api_key, fetch_out, log  # noqa: E402


def main():
    key, _ = api_key()
    if not key:
        raise SystemExit("RUNPOD_API_KEY 를 못 찾았다")
    import runpod
    runpod.api_key = key
    want = sys.argv[1] if len(sys.argv) > 1 else None

    pods = runpod.get_pods() or []
    if not pods:
        log("떠 있는 파드 없음")
        return
    for p in pods:
        pid = p.get("id")
        if want and pid != want:
            continue
        rt = p.get("runtime") or {}
        conn = None
        for pr in (rt.get("ports") or []):
            if int(pr.get("privatePort", 0)) == 22 and pr.get("ip") and pr.get("publicPort"):
                conn = (pr["ip"], int(pr["publicPort"]))
        if not conn:
            log(f"{pid} · {p.get('name')} — SSH 포트가 아직 없다, 건너뜀")
            continue
        job = str(p.get("name", "")).replace("aimo-", "") or "unknown"
        outdir = RESULTS / job / pid
        log(f"{pid} · {p.get('name')} · {conn[0]}:{conn[1]} → {outdir.relative_to(RESULTS.parent.parent)}")
        fetch_out(conn[0], conn[1], "/workspace/out", outdir)


if __name__ == "__main__":
    main()
