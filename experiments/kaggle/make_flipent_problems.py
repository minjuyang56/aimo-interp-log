"""#FLIP-ENT 시험 문제를 **모델별로** base64(JSON) 로 커널에 실을 파일(`flipent_problems.py`)을 만든다.

P2 와 다른 점: 팔이 모델마다 다른 문제집합이므로 리스트가 아니라 **{모델 id: [문제…]} dict** 다.
`entlens.load_problems` 가 dict 면 `ENTLENS_MODEL` 키를 고른다(리스트면 기존 동작 그대로).

행 규칙(사전등록 #FLIP-ENT 와 같다):
  * 라벨이 있는 자체 라벨 행만 — 탈락 문제에는 robust/spurious 라벨이 없다
  * Ġ 손상 pod 4개와 T=1.0 pod(`i6hxd8z437eas4`) 제외 — 이미 npz 가 깨끗한 묶음만 담고 있다
  * 공식 33 과 텍스트가 겹치면 배제(설계상 0 이고 측정으로도 0)
  * 같은 문제는 **처음 것만** (DeepSeek 은 off40 우선)

base64 인 이유: LaTeX 백슬래시를 셸·정규식·heredoc 이 먹어 파일을 깨뜨린 적이 이 저장소에서
**9번** 있다. base64 는 어느 단계에서도 해석되지 않는다.

사용: .venv/Scripts/python.exe experiments/kaggle/make_flipent_problems.py
"""

import base64
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from h21_uncertainty import problems  # noqa: E402

NPZ = ROOT / "experiments/results/probe"
OUT = HERE / "flipent_problems.py"
ARMS_OUT = ROOT / "experiments/results/flipent_arms.json"
WS = re.compile(r"\s+")


def norm(t):
    return WS.sub(" ", str(t)).strip()


def from_npz(name, tag):
    d = np.load(NPZ / name, allow_pickle=True)
    return [dict(arm=tag, pidx=int(d["pidx"][i]), problem=str(d["problems"][i]),
                 y=int(d["y"][i]))
            for i in range(len(d["y"])) if int(d["y"][i]) in (0, 1)]


def main():
    off33 = {norm(r["problem"]) for r in problems()}

    def dedupe(rows):
        seen, out = set(), []
        for r in rows:
            t = norm(r["problem"])
            if t in off33 or t in seen:
                continue
            seen.add(t)
            out.append(r)
        return out

    arms = {
        "Skywork_k8": dedupe(from_npz("Skywork-OR1-Math-7B_own_k8.npz", "Skywork_k8")),
        "Olmo_k8": dedupe(from_npz("Olmo-3-7B-Think_own_k8.npz", "Olmo_k8")),
        "DeepSeek_k8": dedupe(from_npz("DeepSeek-R1-0528-Qwen3-8B_own_k8.npz", "DeepSeek_k8")
                              + from_npz("DeepSeek-R1-0528-Qwen3-8B_own_k8rep.npz", "DeepSeek_k8")),
    }
    m = json.loads((ROOT / "experiments/results/genlab/skywork_labels_merged.json")
                   .read_text(encoding="utf-8"))
    rows = m if isinstance(m, list) else m.get("rows", [])
    arms["Skywork_k4"] = dedupe([dict(arm="Skywork_k4", pidx=int(r.get("pidx", -1)),
                                      problem=r["problem"], y=int(bool(r["label"])))
                                 for r in rows if isinstance(r.get("label"), bool)])

    #  모델별 생성 목록 = 그 모델 팔들의 문제 합집합(텍스트 유일). Skywork 는 k4 가 k8 을 포함한다.
    plan = {"Skywork/Skywork-OR1-Math-7B": ["Skywork_k4", "Skywork_k8"],
            "allenai/Olmo-3-7B-Think": ["Olmo_k8"],
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": ["DeepSeek_k8"]}
    gen = {}
    for mid, keys in plan.items():
        seen, lst = set(), []
        for k in keys:
            for r in arms[k]:
                t = norm(r["problem"])
                if t not in seen:
                    seen.add(t)
                    lst.append({"pidx": len(lst), "problem": r["problem"]})
        gen[mid] = lst

    blob = json.dumps(gen, ensure_ascii=False, sort_keys=True)
    b64 = base64.b64encode(blob.encode("utf-8")).decode("ascii")
    sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    body = ('"""#FLIP-ENT 모델별 시험 문제 (base64 JSON dict). '
            'make_flipent_problems.py 가 생성한다 — 손으로 고치지 말 것."""\n\n'
            f'PROBS_SHA256 = "{sha}"\n'
            f'PROBS_N = {json.dumps({k: len(v) for k, v in gen.items()})}\n'
            f'PROBS_B64 = (\n'
            + "".join(f'    "{b64[i:i + 100]}"\n' for i in range(0, len(b64), 100))
            + ')\n')
    OUT.write_text(body, encoding="utf-8")

    #  왕복 확인 — 쓴 파일을 다시 읽어 디코드가 원본과 같은지 **센다**(주장하지 않고 계측)
    ns = {}
    exec(compile(OUT.read_text(encoding="utf-8"), str(OUT), "exec"), ns)  # noqa: S102
    back = json.loads(base64.b64decode(ns["PROBS_B64"]).decode("utf-8"))
    same = back == gen

    ARMS_OUT.write_text(json.dumps(arms, ensure_ascii=False), encoding="utf-8")
    print(f"저장 {OUT} · {len(b64)} 문자 · sha256 {sha[:16]}…")
    print(f"왕복 확인: 디코드 == 원본 → {'일치' if same else '**불일치**'}")
    for mid, lst in gen.items():
        w = sorted(len(p["problem"].split()) for p in lst)
        print(f"  {mid:40} 문제 {len(lst):>3} · 단어 최소 {w[0]} 중앙 {w[len(w) // 2]} 최대 {w[-1]}")
    for k, rs in arms.items():
        y = np.array([r["y"] for r in rs])
        print(f"  팔 {k:12} n={len(y):>3} robust={int(y.sum()):>3} spurious={len(y) - int(y.sum()):>3} "
              f"상수={y.mean():.3f}")
    #  팔의 문제가 전부 생성 목록에 있는지 — 없으면 평가에서 조용히 행이 사라진다
    for k, rs in arms.items():
        mid = [m2 for m2, ks in plan.items() if k in ks][0]
        have = {norm(p["problem"]) for p in gen[mid]}
        miss = [r for r in rs if norm(r["problem"]) not in have]
        if miss:
            raise SystemExit(f"**{k} 의 {len(miss)}행이 생성 목록에 없다**")
    print(f"커버리지 확인: 네 팔의 모든 행이 생성 목록에 있다 · 팔 저장 {ARMS_OUT}")
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
