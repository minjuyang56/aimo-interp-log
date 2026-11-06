"""rename 섭동의 기호 충돌 — 자체 라벨에 섞인 '정답이 안 보존된 변형'을 센다. GPU 0. (2026-10-10, H22 작업 중 발견)

무엇이 문제인가
  `variants_official.edit_rename` 은 수식 구간 안에서 **홀로 선 한 글자**만 치환한다(대문자 15칸 시프트,
  소문자 짝 교환). 그런데 `\\frac{kc+l}{mc+n}` 에서 `mc` 의 `m` 은 두 글자로 붙어 있어 '홀로 선 글자' 가
  아니라 **안 바뀌고**, 홀로 선 `n` 은 짝 규칙에 따라 `m` 으로 바뀐다 → `mc+m`.
  서로 다른 두 기호가 한 기호로 **뭉친다**. 이건 정답 보존 섭동이 아니라 **깨진 문제**다.

왜 중요한가
  우리 자체 라벨은 "변형에서 max_drop ≥ 0.25 면 spurious" 다. 깨진 변형에서 모델이 틀리는 것은
  spuriousness 가 아니라 **문제가 말이 안 되는 것**이다. 그런 변형이 라벨을 결정했다면 그 라벨은 못 쓴다.
  #77 의 프로브가 이 라벨로 학습·평가되므로 그 몫을 알아야 한다.

세는 방법 (치환이 **구간마다 따로** 돌기 때문에 구간 단위로 본다 — 이걸 뭉쳐서 보면 0 건으로 오판한다)
  구간 안에서 홀로 선 글자 L 을 M 으로 바꿀 때, M 이 그 구간에 존재하면서 M 자신은 홀로 서지 않으면
  (= 같이 바뀌어 나가지 않으면) **충돌**이다.

사용:  .venv/Scripts/python.exe analysis/rename_collision.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
from variants_official import LATEX_CMD, LOWER_PAIRS, MATH_SPAN, edit_rename  # noqa: E402

MERGED = ROOT / "experiments" / "results" / "genlab"
OUT = ROOT / "experiments" / "results" / "rename_collision.txt"
ALONE = re.compile(r"(?<![A-Za-z])([a-z])(?![A-Za-z])")


def span_bodies(text):
    """수식 구간마다 LaTeX 명령어를 걷어낸 본문. 치환이 실제로 보는 문자열이다."""
    for m in MATH_SPAN.finditer(text):
        s, pos, segs = m.group(0), 0, []
        for cmd in LATEX_CMD.finditer(s):
            segs.append(s[pos:cmd.start()])
            pos = cmd.end()
        segs.append(s[pos:])
        yield "".join(segs)


def collisions(text):
    """충돌 목록 [(구간본문, 'n->m'), …]."""
    bad = []
    for body in span_bodies(text):
        alone = set(ALONE.findall(body))
        any_l = set(re.findall(r"[a-z]", body))
        for L in sorted(alone):
            M = LOWER_PAIRS.get(L)
            if M and M in any_l and M not in alone:
                bad.append((body, f"{L}->{M}"))
    return bad


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("rename 기호 충돌 — 자체 라벨에 섞인 '정답 미보존' 변형")
    out()
    files = sorted(MERGED.glob("*_labels_merged.json"))
    if not files:
        out("병합 라벨 파일이 없다.")
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        lab = [r for r in d if isinstance(r.get("label"), bool)]
        sp = [r for r in lab if not r["label"]]
        hit = [r for r in lab if collisions(r["problem"])]
        hit_sp = [r for r in hit if not r["label"]]
        # rename 이 **라벨을 결정한**(var_acc 최저) 충돌 문제 = 라벨 자체가 못 믿을 것
        dec = []
        for r in hit_sp:
            va = r.get("var_acc") or {}
            if va and min(va, key=lambda k: va[k]) == "rename":
                dec.append(r)
        out(f"## {f.stem} — 라벨 {len(lab)}건 (spurious {len(sp)})")
        out()
        out(f"  충돌이 있는 문제 **{len(hit)}건** ({len(hit) / max(1, len(lab)):.0%})")
        out(f"    그중 spurious 로 라벨된 것 {len(hit_sp)}건 / 전체 spurious {len(sp)}건 "
            f"= {len(hit_sp) / max(1, len(sp)):.0%}")
        out(f"    그중 **rename 이 라벨을 결정한** 것 **{len(dec)}건** → 이 라벨은 신뢰할 수 없다 "
            f"(전체 라벨의 {len(dec) / max(1, len(lab)):.0%})")
        out()
        if dec:
            out("  라벨이 깨진 변형에 끌려간 문제:")
            for r in dec:
                cl = collisions(r["problem"])
                out(f"    pidx {r['pidx']:3d} · max_drop {r['max_drop']} · var_acc {r['var_acc']} "
                    f"· 충돌 {[c for _, c in cl]}")
                out(f"      원본 {r['problem'][:120]}")
                out(f"      변형 {str(edit_rename(r['problem']))[:120]}")
        out()
        out("  (충돌이 있어도 rename 이 라벨을 결정하지 않았다면 라벨은 다른 변형이 정했으므로 살아 있다.)")
        out()
    out("## 조치")
    out()
    out("  * 제출물 `solutions/h22/h22_lib.py` 의 rename 은 **구간별 충돌 가드**를 넣어 고쳤다")
    out("    (`_safe_pairs`: M 이 구간에 있으면서 홀로 서지 않으면 그 치환을 포기한다).")
    out("  * `experiments/variants_official.py` 는 **고치지 않았다** — 이미 생성한 라벨이 그 규칙으로 만들어졌고,")
    out("    고치면 Skywork 88 과 앞으로 나올 DeepSeek·Olmo 라벨의 비교 가능성이 깨진다.")
    out("    대신 위 '라벨을 결정한' 문제들을 **사후 제외 후보**로 기록해 둔다(#77 민감도 분석에 추가).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
