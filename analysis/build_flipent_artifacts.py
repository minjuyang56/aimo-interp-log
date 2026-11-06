"""#FLIPENT-ZIP 가중치 — 모델별 `ent_front` 임계를 공식 33(P2) 산출물에서 굽는다. 비용 $0.

임계의 정의는 **#FLIP-ENT 사전등록 그대로**다: 공식 `train-main-v2` 33문제를 같은 설정
(HF · fp16 · greedy · cap 1024)으로 생성한 P2 산출물에서 그 모델의 `ent_front` **중앙값**.
**공식 라벨을 보지 않는다** — 점수 분포만 쓴다(그래서 AUC 최적화가 아니다).

`ent_front` 정의는 `analysis/entlens_eval.ent_feats` 를 **import 해서** 쓴다(베껴 적지 않는다 —
이 저장소에서 공식을 베껴 적다 어긋난 적이 여러 번 있다).

**재현 게이트**: CLAUDE.md #FLIP-ENT 결과 절에 적힌 세 임계(DeepSeek 0.1906 · Olmo 0.2414 ·
Skywork 0.1364)를 소수 넷째 자리까지 재현하지 않으면 **가중치를 쓰지 않고 멈춘다**.
Qwen3.5 는 그 표에 없다(그 팔은 자체 라벨 0개로 판정 불가였다) — 그래서 **이번에 처음 계산**되고
게이트가 없다. 그 사실을 보고서에 적는다.

사용:  .venv/Scripts/python.exe analysis/build_flipent_artifacts.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))

from entlens_eval import ent_feats  # noqa: E402  — 같은 정의를 import
from flipent_eval import load_p2  # noqa: E402

RES = ROOT / "experiments/results"
ORG = RES / "entlens"                     # P2 공식 33 — 임계의 유일한 원천
ART = ROOT / "solutions/flipent/artifacts/flipent.json"
OUT = RES / "flipent_artifacts.txt"
WS = re.compile(r"\s+")
LINES = []

#  별칭(커널 산출물 파일명) -> 채점기가 넘기는 `model_id`
ALIAS_MODEL = {
    "Qwen3_5-4B": "Qwen/Qwen3.5-4B",
    "Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
    "Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think",
    "DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
#  CLAUDE.md #FLIP-ENT 결과 절의 기록값 (재현 게이트)
RECORDED = {"DeepSeek-R1-0528-Qwen3-8B": 0.1906, "Olmo-3-7B-Think": 0.2414,
            "Skywork-OR1-Math-7B": 0.1364}
#  h15C 출하 임계(폴백)는 **그 제출물의 가중치 파일에서 직접 읽는다** — 베껴 적으면 어긋난다.
H15C_THR = float(json.loads((ROOT / "solutions/h15C/artifacts/h15.json")
                            .read_text(encoding="utf-8"))["thr"]["C"])


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def main():
    out("#FLIPENT-ZIP 가중치 — ent_front 모델별 임계 = 공식 33(P2) 중앙값 · 라벨 미사용 · $0")
    out()
    per, gate_fail = {}, []
    for alias in sorted(ALIAS_MODEL):
        rows = load_p2(ORG, alias)
        if not rows:
            out("   **%s: 공식 33 P2 산출물이 없다 → 임계 없음(그 모델은 폴백으로 간다)**" % alias)
            continue
        fr = [ent_feats(r["ent"]) for r in rows]
        v = np.array(sorted(f[0] for f in fr if f is not None))
        bad = sum(1 for f in fr if f is None)
        thr = float(np.median(v))
        #  고정 임계가 그 학습셋에서 찍는 True 비율 — 중앙값이므로 설계상 0.5 근처여야 한다
        tru = float((v >= thr).mean())
        per[ALIAS_MODEL[alias]] = dict(thr=thr, n=int(len(v)), train_true_rate=tru,
                                       front_min=float(v.min()), front_max=float(v.max()),
                                       alias=alias)
        rec = RECORDED.get(alias)
        tag = ""
        if rec is not None:
            ok = abs(round(thr, 4) - rec) < 5e-5
            tag = "  기록값 %.4f %s" % (rec, "일치" if ok else "**불일치**")
            if not ok:
                gate_fail.append((alias, thr, rec))
        else:
            tag = "  (기록 없음 — 이번에 처음 계산)"
        out("   %-30s n=%3d · 중앙값 **%.6f** · 범위 [%.4f, %.4f] · train True 비율 %.3f · T<8 %d%s"
            % (alias, len(v), thr, v.min(), v.max(), tru, bad, tag))
    out()
    if gate_fail:
        out("## 재현 게이트 실패 — 가중치를 쓰지 않는다")
        for alias, got, rec in gate_fail:
            out("   %s: 계산 %.6f vs 기록 %.4f" % (alias, got, rec))
        OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        raise SystemExit("재현 게이트 실패 — CLAUDE.md 기록과 임계가 어긋난다")
    out("## 재현 게이트 통과 — 기록된 세 임계를 소수 넷째 자리까지 재현했다 (Qwen3.5 는 신규)")
    out()

    art = dict(
        rule=("score = ent_front = mean token entropy (full-vocab, natural log) over the first "
              "ceil(T/4) generated tokens of a greedy continuation capped at 1024 tokens. "
              "HIGHER entropy => robust (True). Per-model threshold = median ent_front on the "
              "organisers' official train-main-v2 33 problems, measured with the same generation "
              "setting (P2 artifacts); the official labels are NOT used."),
        sign="ent_front >= thr  ->  True (robust)",
        cap=1024, front_frac=0.25, dtype_measured="fp16",
        prereg="CLAUDE.md ## #FLIP-ENT (pre-registered 2026-10-20, commit 4fcd838)",
        experimental=True,
        note=("EXPERIMENTAL submission (directive #FLIPENT-ZIP): the standing build gate requires "
              ">0.645 on the official 33, and this score was never evaluated there (the #FLIP-ENT "
              "pre-registration forbade re-judging on data already seen). On its own four self-label "
              "arms all four fell short of their shuffle nulls (AUC 0.721/0.562/0.337/0.526)."),
        per_model=per, h15c_thr=H15C_THR,
    )
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out("## 저장 %s (%d bytes · 모델 %d종)" % (ART.relative_to(ROOT), ART.stat().st_size, len(per)))
    out("   h15C 폴백 임계 %.6f (−log 단어수 · h15C 출하값과 같다)" % H15C_THR)
    out()
    out("## 읽는 법 — 이 제출물의 근거와 한계")
    out("   * 네 팔 전부 미달이었다(#FLIP-ENT): Skywork k8 0.721 [null 0.740] · Olmo 0.562 [0.686] ·")
    out("     DeepSeek 0.337 [0.756] · Skywork k4 0.526 [0.618]. 즉 **신호 미입증 상태의 실험 제출**이다.")
    out("   * 그 팔들의 고정임계 정확도는 0.406~0.723 이고 각 팔의 상수 기준선(0.68~0.92) **아래**였다.")
    out("     리더보드에서 상수를 넘을 근거는 측정에 없다 — [author]가 실험용으로 지명한 것이다.")
    out("   * **정밀도 위험**: 임계는 Kaggle **fp16** 측정에서 나왔다. P2 규칙③ 감사에서 같은 greedy")
    out("     설정인데 4bit↔fp16 으로 Qwen3.5 의 ent_front 가 0.034 움직였다(문제별 ρ_s 0.449).")
    out("     임계가 0.136~0.241 이므로 그 크기는 12~25% 다 → 제출물 기본 정밀도를 **fp16 으로 맞췄다**")
    out("     (`FLIPENT_DTYPE=bf16` 로 바꿀 수 있다). 이게 sub2048 의 교정 불량과 같은 계열의 위험이다.")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))


if __name__ == "__main__":
    main()
