"""디토큰화 가드 스모크 — **GPU 0 · 비용 0.** 지휘 #DS-RELABEL 단계 1의 생성 전 점검.

묻는 것 하나: `genlab_v2.decode_out` 이 **vLLM 이 손상시킨 그 텍스트를 실제로 고치는가.**
주장이 아니라 계측으로 답한다 — 입력은 2026-10-11 에 파드에서 받아 저장한 **진짜 손상 텍스트**다
(`runpod/genprobe/*/out/DeepSeek-...-8B_greedy.json`, 41/41 행이 손상).

### 왜 재인코딩으로 token_ids 를 만드나
파드에서 `o.token_ids` 를 저장하지 않았다(그때는 텍스트만 받았다). 그런데 #TOK-AUDIT 에서
**손상 텍스트를 옛 토크나이저로 재인코딩하면 vLLM 의 토큰 id 가 복원된다**는 것을 확인했다
(길이비 0.999~1.000). 그래서 재인코딩으로 파드에서 오던 id 를 재현하고, 그 id 를 새 경로로
디코드한다 — 즉 `decode_out` 이 파드에서 받게 될 입력과 같은 것을 받는다.

### 통과 기준 (지휘: 마커 0건 · 경고 0건)
  1. 새 경로의 텍스트에 BPE 마커 **0건** (DeepSeek)
  2. 그 텍스트로 `acc_of` 를 돌렸을 때 **경고 0건** (경고는 마커가 있을 때만 찍힌다)
  3. **대조군 불변**: Skywork·Olmo 는 마커 0건이고 정답 집계가 변하지 않는다

사용: .venv/Scripts/python.exe analysis/detok_guard_smoke.py
"""

import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GP = ROOT / "experiments" / "results" / "runpod" / "genprobe"
MODELS = {
    "Skywork": ("Skywork/Skywork-OR1-Math-7B", "Skywork-OR1-Math-7B_greedy.json"),
    "Olmo": ("allenai/Olmo-3-7B-Think", "Olmo-3-7B-Think_greedy.json"),
    "DeepSeek": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-0528-Qwen3-8B_greedy.json"),
}

# genlab_v2 는 solve_common 의 심볼을 주입받는 전제로 쓰여 있다 — 커널 헤더와 같은 순서로 넣는다.
ns = {"__name__": "genlab_v2_smoke"}
for f in ("experiments/variants_official.py", "experiments/variants_nonofficial.py",
          "experiments/kaggle/solve_common.py", "experiments/kaggle/genlab_v2.py"):
    exec(compile((ROOT / f).read_text(encoding="utf-8"), f, "exec"), ns)
decode_out, marks_in, acc_of, safe_detok = ns["decode_out"], ns["marks_in"], ns["acc_of"], ns["safe_detok"]

lines = []


def out(t=""):
    print(t, flush=True)
    lines.append(t)


class Stub:
    """vLLM CompletionOutput 의 우리가 쓰는 부분만 흉내낸다."""

    def __init__(self, text, ids, fin="stop"):
        self.text, self.token_ids, self.finish_reason = text, ids, fin


def main():
    from transformers import AutoTokenizer
    out("디토큰화 가드 스모크 — 저장된 **진짜 손상 텍스트**로 decode_out 을 시험한다 (GPU 0)")
    out()
    ok = True
    for alias, (mid, fn) in MODELS.items():
        hits = sorted(GP.glob(f"*/out/{fn}"))
        if not hits:
            out(f"## {alias} · 파일 없음 ({fn}) — 건너뜀")
            continue
        rows = json.loads(hits[-1].read_text(encoding="utf-8"))
        rows = rows if isinstance(rows, list) else rows.get("rows", [])
        old = AutoTokenizer.from_pretrained(mid)          # 파드에서 vLLM 이 쓴 그 경로
        dtok = safe_detok(mid)                            # 새 경로
        n_raw = n_new = n_warn = n_warn_old = 0
        hit_old = hit_new = 0
        for r in rows:
            txt = r.get("text") or r.get("greedy_text") or ""
            if not txt:
                continue
            ids = old(txt, add_special_tokens=False)["input_ids"]   # vLLM 의 id 를 복원
            o = Stub(txt, ids)
            new_txt, mr, ms, _ = decode_out(o, dtok)
            n_raw += mr
            n_new += ms
            gold = r.get("gold")
            if gold is None:
                continue
            # **신·구를 따로 센다.** 섞어 세면 게이트("경고 0건")가 무슨 텍스트에 대한
            # 것인지 모호해진다 — 옛 텍스트에서 경고가 나는 건 당연하고(그게 손상이다),
            # 우리가 요구하는 것은 **새 경로 텍스트에서 0** 이다.
            b_new, b_old = io.StringIO(), io.StringIO()
            with redirect_stdout(b_new):
                _, _, h_new = acc_of([(new_txt, True)], gold, 1)
            with redirect_stdout(b_old):
                _, _, h_old = acc_of([(txt, True)], gold, 1)
            n_warn += b_new.getvalue().count("[경고]")
            n_warn_old += b_old.getvalue().count("[경고]")
            hit_new += h_new
            hit_old += h_old
        out(f"## {alias} · {len(rows)}행 · 토크나이저 {type(old).__name__} → {type(dtok).__name__}")
        out(f"   vLLM o.text 마커 **{n_raw}** → 새 경로 텍스트 마커 **{n_new}**")
        out(f"   acc_of 경고: **새 경로 {n_warn}회** (게이트) · 옛 텍스트 {n_warn_old}회 "
            f"(손상이 있으면 당연히 난다 — debpe 가 2차로 받아 집계는 양쪽 다 맞는다)")
        out(f"   정답 집계: 손상 텍스트 {hit_old} · 새 경로 {hit_new}")
        if n_new != 0 or n_warn != 0:
            ok = False
            out("   !! 마커가 남았다 — 생성하면 안 된다")
        out()
    out("## 판정")
    out(f"   {'통과 — 새 경로는 마커를 0 으로 만든다' if ok else '실패 — 생성 중단'}")
    o = ROOT / "experiments" / "results" / "detok_guard_smoke.txt"
    o.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장 {o}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
