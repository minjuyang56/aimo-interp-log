"""#FLIP-ENT 평가기 — `ent_front` 부호를 뒤집어(큼 = robust) 자체 라벨 네 팔에서 1회 시험. 비용 $0.

사전등록은 CLAUDE.md `## #FLIP-ENT` 절이다. **이 파일은 커널 결과를 보기 전에 썼다**(커밋 순서로 확인된다).

고정된 것
  특징   `ent_front` **하나** — 생성 앞 25%(`t < ceil(T/4)`) 의 전어휘 자연로그 엔트로피 평균.
         정의는 `analysis/entlens_eval.ent_feats` 를 **import 해서** 쓴다(베껴 적지 않는다).
  부호   **큼 = robust** (P2 의 '큼 = spurious' 를 뒤집은 것). 결과 보고 뒤집지 않는다.
  임계   **공식 33(P2 산출물)의 모델별 `ent_front` 중앙값** — 공식 라벨을 보지 않는다.
  판정   팔마다 `AUC > 0.5` **그리고** `AUC > 그 팔의 셔플 null 95% 상단` 둘 다.

`ent_slope`·`ent_back` 은 **계산하지도 보고하지도 않는다**(지휘가 '셋 중 미리 하나' 로 지정).
원시 `ent` 는 전부 저장돼 있으므로 새 사전등록이 있으면 나중에 잴 수 있다.

사용:
  .venv/Scripts/python.exe analysis/flipent_eval.py --scan
  .venv/Scripts/python.exe analysis/flipent_eval.py --perms 1000
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))

from entlens_eval import ent_feats  # noqa: E402  — 같은 정의를 import 한다
from probe_final_eval import auc_of, resid, _perm_list  # noqa: E402

RES = ROOT / "experiments/results"
FLIP = RES / "flipent"          # #FLIP-ENT 커널 산출물 (공식 33 은 RES/entlens 에 따로 있다)
ORG = RES / "entlens"           # P2 공식 33 — 임계의 원천
ARMS = RES / "flipent_arms.json"
OUT = RES / "flipent_report.txt"
WS = re.compile(r"\s+")
LINES = []

#  팔 -> 그 팔을 재는 모델(커널 별칭). Skywork 두 팔은 **같은 생성**을 공유한다(k4 포함 k8).
ARM_MODEL = {"Skywork_k8": "Skywork-OR1-Math-7B", "Skywork_k4": "Skywork-OR1-Math-7B",
             "Olmo_k8": "Olmo-3-7B-Think", "DeepSeek_k8": "DeepSeek-R1-0528-Qwen3-8B"}
MAIN_ARMS = ["Skywork_k8", "Olmo_k8", "DeepSeek_k8"]      # 주팔 = k=8 (Amendment 2)
AUX_ARMS = ["Skywork_k4"]


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def norm(t):
    return WS.sub(" ", str(t)).strip()


def load_p2(root, alias):
    """`<root>/**/<alias>_p2.json` 하나를 찾아 읽는다. 둘 이상이면 멈춘다(섞이면 안 된다)."""
    hits = sorted(root.rglob(alias + "_p2.json")) if root.is_dir() else []
    if not hits:
        return None
    if len(hits) > 1:
        raise SystemExit("같은 별칭 p2 가 여러 개다 — 섞이면 안 된다: " + str(hits))
    return json.loads(hits[0].read_text(encoding="utf-8"))


def front_of(rows):
    """텍스트(정규화) -> ent_front. T<8 인 행은 넣지 않는다(판정 불가)."""
    d, bad = {}, []
    for r in rows:
        f = ent_feats(r["ent"])
        if f is None:
            bad.append((r.get("pidx"), len(r["ent"])))
            continue
        d[norm(r["problem"])] = dict(front=f[0], n_tok=r["n_tok"], fin=r["finished"],
                                     marks=r["n_marks"], text=r["text"], pidx=r.get("pidx"))
    return d, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()

    arms = json.loads(ARMS.read_text(encoding="utf-8"))
    aliases = sorted(set(ARM_MODEL.values()))
    got = {al: load_p2(FLIP, al) for al in aliases}
    org = {al: load_p2(ORG, al) for al in aliases}

    if a.scan:
        for al in aliases:
            n = len(got[al]) if got[al] else 0
            o = len(org[al]) if org[al] else 0
            print("  %-30s flipent %3d행 · 공식33(P2) %3d행" % (al, n, o))
        raise SystemExit(0)

    out("#FLIP-ENT — ent_front 부호 반전(큼 = robust) · 자체 라벨 네 팔 · 사전등록 1회 · 비용 $0")
    out("   임계 = 공식 33(P2) 의 모델별 ent_front 중앙값(라벨 미사용) · 셔플 null %d회" % a.perms)
    out("   부호 고정: ent_front 큼 = robust (뒤집지 않는다) · ent_slope·ent_back 은 보고하지 않는다")
    out()

    #  -- 임계: 공식 33 의 모델별 중앙값 --
    thr = {}
    for al in aliases:
        if not org[al]:
            out("   **%s: 공식 33 P2 산출물이 없다 → 임계를 못 만든다**" % al)
            continue
        fo, _ = front_of(org[al])
        v = np.array(sorted(fo[k]["front"] for k in fo))
        thr[al] = float(np.median(v))
        out("   임계 %-30s n=%d · 중앙값 **%.4f** · 범위 [%.4f, %.4f]"
            % (al, len(v), thr[al], v.min(), v.max()))
    out()

    #  -- 규칙 2: 받은 원문 점검 --
    out("## 상설 규칙 ② — 받은 원문 점검 (이 줄을 읽은 뒤에만 판정으로 넘어간다)")
    for al in aliases:
        if not got[al]:
            out("   %-30s **아직 안 받음**" % al)
            continue
        fr, bad = front_of(got[al])
        marks = sum(v["marks"] for v in fr.values())
        empty = sum(1 for v in fr.values() if not v["text"].strip())
        fin = sum(1 for v in fr.values() if v["fin"])
        toks = sorted(v["n_tok"] for v in fr.values())
        out("   %-30s 행 %d · 마커 %d · 빈텍스트 %d · 완주 %d · 토큰 중앙 %d · T<8 버린 행 %d"
            % (al, len(fr), marks, empty, fin, toks[len(toks) // 2], len(bad)))
    out()

    #  -- 팔별 판정 --
    res, order = {}, []
    for arm in MAIN_ARMS + AUX_ARMS:
        al = ARM_MODEL[arm]
        lab = arms[arm]
        if not got[al] or al not in thr:
            out("## %s (%s) — 산출물/임계 없음 → **판정 불가**" % (arm, al))
            out()
            continue
        fr, _ = front_of(got[al])
        keep = [r for r in lab if norm(r["problem"]) in fr]
        miss = len(lab) - len(keep)
        y = np.array([r["y"] for r in keep], int)
        s = np.array([fr[norm(r["problem"])]["front"] for r in keep], float)
        w = np.array([len(r["problem"].split()) for r in keep], float)
        if len(y) < 10 or len(set(y.tolist())) < 2:
            out("## %s — 행 %d · 클래스 %s → **판정 불가**" % (arm, len(y), sorted(set(y.tolist()))))
            out()
            continue
        A = auc_of(y, s)                                  # 점수 큼 = robust(y=1)
        Ar = auc_of(y, resid(s, np.log(w)))               # 길이 잔차 (게이트 아님)
        pred = s >= thr[al]
        acc = float((pred == (y == 1)).mean())
        tru = float(pred.mean())
        const = float((y == 1).mean())
        perms = _perm_list(y, a.perms)
        nul = np.array([auc_of(yp, s) for yp in perms])
        up = float(np.quantile(nul, 0.95))
        nacc = np.array([float((pred == (yp == 1)).mean()) for yp in perms])
        uacc = float(np.quantile(nacc, 0.95))
        pas = bool(A > 0.5 and A > up)
        res[arm] = dict(n=len(y), spur=int((y == 0).sum()), auc=A, up=up, resid=Ar, acc=acc,
                        tru=tru, const=const, uacc=uacc, pas=pas, miss=miss,
                        mr=float(s[y == 1].mean()), ms=float(s[y == 0].mean()))
        order.append(arm)
        out("## %s (%s) · 행 %d (robust %d / spurious %d)%s"
            % (arm, al, len(y), int(y.sum()), int((y == 0).sum()),
               (" · 산출물에 없는 라벨행 %d" % miss) if miss else ""))
        out("   **AUC %.3f** · 셔플 null 95%% 상단 **%.3f** → %s  (AUC>0.5 %s · AUC>null %s)"
            % (A, up, "**통과**" if pas else "미달", "O" if A > 0.5 else "X",
               "O" if A > up else "X"))
        out("   고정임계 정확도 %.3f (null상단 %.3f) · True 비율 %.3f · 그 팔의 상수 기준선 %.3f"
            % (acc, uacc, tru, const))
        out("   길이잔차 AUC %.3f (게이트 아님) · ent_front robust평균 %.4f vs spurious평균 %.4f"
            % (Ar, s[y == 1].mean(), s[y == 0].mean()))
        out()

    #  -- 같은 부호 수 (지휘 지정) --
    out("## 같은 부호 수 (지휘 지정) — 사전등록 방향(큼 = robust)과 같은 팔이 몇 개인가")
    main_ok = [k for k in MAIN_ARMS if k in res]
    same = [k for k in main_ok if res[k]["auc"] > 0.5]
    detail = " · ".join("%s %.3f" % (k, res[k]["auc"]) for k in main_ok) or "없음"
    out("   주팔(모델별 k=8) %d개 중 AUC > 0.5 = **%d개**   [%s]" % (len(main_ok), len(same), detail))
    for k in AUX_ARMS:
        if k in res:
            out("   보조 %s AUC %.3f (k=4 라벨 · Skywork_k8 과 문제가 겹친다 → 같은 부호 수에 "
                "**중복 계산하지 않는다**)" % (k, res[k]["auc"]))
    out("   Qwen3.5-4B: 자체 라벨 **0개** → 판정 불가 (사전등록에 적은 그대로)")
    out()
    out("## 요약 표")
    out("   %-14s%4s%5s%8s%10s%8s%8s%8s%7s" % ("팔", "n", "spur", "AUC", "null상단", "잔차",
                                               "정확도", "상수", "판정"))
    for k in order:
        r = res[k]
        out("   %-14s%4d%5d%8.3f%10.3f%8.3f%8.3f%8.3f%7s"
            % (k, r["n"], r["spur"], r["auc"], r["up"], r["resid"], r["acc"], r["const"],
               "통과" if r["pas"] else "미달"))
    out()
    out("## 읽는 법")
    out("   * spurious 가 4건인 두 팔(Skywork_k8 · DeepSeek_k8)은 **생성 전에** null 상단이 0.76 으로")
    out("     계산됐다 — 이 프로젝트가 기록한 분해능 밖이라 부호만 읽고 유의성을 주장하지 않는다.")
    out("   * 실제로 판정되는 팔은 **Olmo_k8(>0.686) · Skywork_k4(>0.623)** 둘이다(사전등록 표).")
    out("   * zip 은 만들지 않는다 — 상설 규칙이 '공식 33 에서 0.645 초과' 를 요구하고 이 라운드는")
    out("     공식 33 에서 재지 않는다. 이 팔들의 상수 기준선이 0.68~0.92 라 '정확도 >0.65' 도 못 쓴다.")
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))


if __name__ == "__main__":
    main()
