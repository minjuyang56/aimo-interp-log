"""라벨 파일의 **파생 필드를 소급 계산**한다 — `label_fin` 과 `worst_kind`/`worst_cause`. GPU 0.

## 왜 필요한가 (2026-10-10 발견)
`genlab_v2.py` 에는 변형·라벨을 마무리하는 경로가 **둘** 있고 둘이 **다른 필드를 쓴다**:
  * 전체 실행 경로(`run`)           → `base_acc_fin` · `max_drop_fin` · **`label_fin`** 를 쓴다.
                                     `worst_kind`/`worst_cause` 는 **안 쓴다**.
  * 선별 재사용 경로(`_variants_and_labels`, `--screen`) → `worst_kind` · **`worst_cause`** 와
                                     `gen_cap`/`gen_k` 를 쓴다. **`label_fin` 은 안 쓴다**.
RunPod 에서 선별을 재사용한 실행(Olmo off40 · DeepSeek off40)은 뒤쪽 경로를 탔으므로
**`label_fin` 이 아예 없다**. Skywork off40 은 앞쪽 경로라 `worst_cause` 가 없다.

이게 #77 을 직접 깨뜨린다:
  * 통과 기준 3 = "대조 라벨 `label_fin` 의 AUC 가 0.5 미만으로 뒤집히지 않는다" → Olmo·DeepSeek 에서 판정 불가.
  * 사전지정 민감도 부분집합 = "spurious 원인이 **오답 우세**(n_wrong ≥ n_unfin)인 행" → Skywork 에서 판정 불가.

## 왜 GPU 가 필요 없나
두 필드 모두 **이미 저장된 수**에서 나온다. `var_detail[유형] = {n_fin, n_hit, n_unfin, n_wrong}` 이고
원본은 행의 `n_fin`·`n_hit` 다. 생성을 다시 할 이유가 없다 — 공식만 같게 적용하면 된다.

## 공식이 같다는 것을 **계측으로** 보인다
`--verify` 는 그 필드를 **이미 가진** 파일에서 값을 다시 계산해 저장값과 비교한다. 하나라도 어긋나면
멈춘다. 이 검사를 통과하지 않으면 소급 계산을 신뢰할 수 없다(코드를 베껴 적다 미묘하게 어긋나는 게
이 저장소에서 여러 번 났다 — 층 인덱스 off-by-one, heredoc 백슬래시).

사용:
  python analysis/backfill_derived_labels.py            # 검증 + 소급 계산(쓰기)
  python analysis/backfill_derived_labels.py --dry-run  # 쓰지 않고 무엇이 바뀌는지만
"""

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#  genlab_v2.py 와 **같은 임계값**이어야 한다. 바꾸면 라벨 정의가 바뀐다.
DROP_ROBUST, DROP_SPUR = 0.10, 0.25

PATTERNS = [
    "experiments/results/genlab/*/genlab/*_labels.json",
    "experiments/results/genlab/*_labels_merged.json",
    "experiments/results/runpod/labelgen32k/*/out/*_labels.json",
]


def derive_fin(r):
    """완주 샘플만 분모로 쓴 대조 라벨. genlab_v2.py 전체경로 262–269 줄과 같은 식."""
    #  옛 파일에는 `n_hit` 이 없다(그때는 base_acc×n_fin 으로 되돌려 썼다 — genlab_v2 204줄 주석).
    #  되돌림은 base_acc 가 이미 미완주를 오답으로 세고 있어 틀린 값을 만들므로, **그 행은 포기한다**
    #  (label_fin = None). 없는 수를 추정해 채우면 대조 라벨이 대조가 아니게 된다.
    bf = (r["n_hit"] / r["n_fin"]) if (r.get("n_fin") and r.get("n_hit") is not None) else None
    vf = [(d["n_hit"] / d["n_fin"]) for d in (r.get("var_detail") or {}).values() if d.get("n_fin")]
    if bf is None or not vf:
        return {"base_acc_fin": bf, "max_drop_fin": None, "label_fin": None}
    df = bf - min(vf)
    return {"base_acc_fin": bf, "max_drop_fin": df,
            "label_fin": True if df <= DROP_ROBUST else (False if df >= DROP_SPUR else None)}


def derive_worst(r):
    """가장 많이 떨어뜨린 변형 유형과 그 원인 분해. 선별재사용 경로 338–340 줄과 같은 식."""
    va = {k: v for k, v in (r.get("var_acc") or {}).items() if v is not None}
    if not va:
        return {}
    worst = min(va, key=lambda k: va[k])
    return {"worst_kind": worst, "worst_cause": (r.get("var_detail") or {}).get(worst)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted({f for p in PATTERNS for f in glob.glob(str(ROOT / p))})
    print(f"라벨 파일 {len(files)}개\n")

    # ---- 1) 검증: 필드를 이미 가진 파일에서 공식을 재현하는가
    checked = {"label_fin": 0, "worst_kind": 0}
    bad = []
    for f in files:
        rows = json.loads(Path(f).read_text(encoding="utf-8"))
        for r in rows:
            if "label_fin" in r:
                d = derive_fin(r)
                checked["label_fin"] += 1
                for k in ("base_acc_fin", "max_drop_fin", "label_fin"):
                    x, y = r.get(k), d[k]
                    same = (x is None and y is None) or (
                        x is not None and y is not None
                        and (abs(x - y) < 1e-9 if isinstance(y, float) else x == y))
                    if not same:
                        bad.append((Path(f).name, r["pidx"], k, x, y))
            if "worst_kind" in r:
                d = derive_worst(r)
                checked["worst_kind"] += 1
                if d and (r.get("worst_kind") != d["worst_kind"]
                          or r.get("worst_cause") != d["worst_cause"]):
                    bad.append((Path(f).name, r["pidx"], "worst_kind",
                                r.get("worst_kind"), d["worst_kind"]))
    print(f"[검증] 기존 값 재현: label_fin 계열 {checked['label_fin']}행 · "
          f"worst 계열 {checked['worst_kind']}행 · **불일치 {len(bad)}건**")
    for b in bad[:10]:
        print("   ", b)
    if bad:
        raise SystemExit("공식이 저장값을 재현하지 못한다 — 소급 계산을 중단한다.")

    # ---- 2) 소급 계산
    print()
    for f in files:
        p = Path(f)
        rows = json.loads(p.read_text(encoding="utf-8"))
        n_fin_add = n_worst_add = 0
        for r in rows:
            if "label_fin" not in r:
                r.update(derive_fin(r))
                r["label_fin_backfilled"] = True      # 출처를 남긴다
                n_fin_add += 1
            if "worst_kind" not in r:
                d = derive_worst(r)
                if d:
                    r.update(d)
                    r["worst_backfilled"] = True
                    n_worst_add += 1
        if n_fin_add or n_worst_add:
            lab = [r for r in rows if r.get("label_fin") is not None]
            print(f"{p.name}: label_fin +{n_fin_add} · worst +{n_worst_add} → "
                  f"label_fin 유효 {len(lab)} "
                  f"(robust {sum(1 for r in lab if r['label_fin'])} / "
                  f"spurious {sum(1 for r in lab if r['label_fin'] is False)})")
            if not a.dry_run:
                p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        else:
            print(f"{p.name}: 이미 둘 다 있음 — 변경 없음")
    print("\n(--dry-run: 쓰지 않았다)" if a.dry_run else "\n저장 완료")


if __name__ == "__main__":
    main()
