"""Codabench 제출 ID 수집 — OpenReview 양식에 적을 submission_id 를 찾는다 (지휘 #FINAL 1).

**인증 없이 받을 수 있는 것과 받을 수 없는 것을 가른다.**
  * 받을 수 있다: `GET /api/phases/{phase}/get_leaderboard/?page_size=500&page=N`
    - 인증 불필요. 응답은 {"title","id","count","submissions":[...]} 이고 **키가 `submissions`** 다
      (`results` 가 아니다 — 10/28 에 내가 이걸 잘못 읽어 0행으로 봤다).
    - 각 행에 `id`(= submission_id) · `owner` · `created_when` · `scores`(task_id x column_key).
    - **리더보드에 올린 제출만** 보인다. Codabench 는 '실행' 과 '리더보드 제출' 이 별개라서
      실행만 한 제출은 여기 안 나온다 → sub2048 처럼 안 보이는 것이 있다.
  * 받을 수 없다: `GET /api/submissions/` 는 **인증된 사용자로 필터**되므로 비인증이면 `count: 0`.
    `/api/my_submissions/` · `/api/phases/{id}/get_submissions/` 는 404, `/api/phases/{id}/` 는 403.
    → **우리 계정의 전체 제출 이력은 토큰/세션 없이 API 로 못 받는다.** 비밀번호는 쓰지 않는다.

규칙 8 준수: 다른 참가자 행은 **집계 점수 분포만** 세고 `detailed_results` 등 내부는 열지 않는다.

사용: .venv/Scripts/python.exe analysis/codabench_ids.py
"""

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "experiments/results/codabench_board_raw.json"
OUT = ROOT / "experiments/results/codabench_ids.txt"
PHASE = 30038
TASK = {36237: "Main(N=21)", 36241: "Small(N=14)"}
OWNER = "[author-codabench]"
L = []


def out(s=""):
    print(s, flush=True)
    L.append(s)


def pull():
    tot = {}
    for page in range(1, 9):
        u = ("https://www.codabench.org/api/phases/%d/get_leaderboard/?page_size=500&page=%d"
             % (PHASE, page))
        got = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(u, timeout=90) as r:
                    got = json.loads(r.read())
                break
            except Exception:
                time.sleep(2)
        if got is None:
            break
        rows = got.get("submissions", [])
        for x in rows:
            tot[x["id"]] = x
        if len(rows) < 500:
            break
    return tot


def main():
    tot = pull()
    out("Codabench phase %d — 리더보드에서 받은 제출 %d건" % (PHASE, len(tot)))
    if tot:
        #  **빈 응답으로 저장본을 덮지 않는다** — 이 엔드포인트는 간헐적으로 0건을 준다
        #  (한 번 덮어서 1,726건 저장본을 날렸고, 그래서 이 가드를 넣었다).
        RAW.write_text(json.dumps({"count": len(tot), "submissions": list(tot.values())},
                                  ensure_ascii=False), encoding="utf-8")
    else:
        out("  **응답 0건 — 저장본을 덮지 않고 기존 파일로 계속한다**")
        if RAW.is_file():
            old = json.loads(RAW.read_text(encoding="utf-8"))
            tot = {s["id"]: s for s in old.get("submissions", [])}
            out("  저장본에서 %d건 복원" % len(tot))

    ours = sorted((s for s in tot.values() if str(s.get("owner")) == OWNER),
                  key=lambda x: str(x.get("created_when")))
    out()
    out("=== 우리 계정 '%s' — 리더보드에 보이는 전부 (%d건) ===" % (OWNER, len(ours)))
    for s in ours:
        cells = {}
        for sc in s.get("scores", []):
            cells[(TASK.get(sc.get("task_id"), sc.get("task_id")), sc.get("column_key"))] = sc.get("score")
        tr = [t for t in TASK.values() if cells.get((t, "accuracy")) is not None]
        out("  submission_id %-8s  %s UTC  트랙 %s" % (s["id"], str(s.get("created_when"))[:19], tr))
        for t in TASK.values():
            a = cells.get((t, "accuracy"))
            if a is not None:
                out("      %-11s accuracy %-6s coverage %-5s invalid %s"
                    % (t, a, cells.get((t, "coverage")), cells.get((t, "invalid_predictions"))))

    out()
    out("=== 리더보드 집계 분포 (accuracy 칸만 · 규칙 8) ===")
    for tid, nm in TASK.items():
        v = []
        for s in tot.values():
            for sc in s.get("scores", []):
                if (sc.get("task_id") == tid and sc.get("column_key") == "accuracy"
                        and sc.get("score") is not None):
                    try:
                        v.append(float(sc["score"]))
                    except Exception:
                        pass
        v.sort(reverse=True)
        out("  %-11s 유효 점수 %4d · 최고 %.2f · 1.00 %3d건 · >=0.86 %3d건 · 중앙 %.2f"
            % (nm, len(v), v[0] if v else -1, sum(1 for x in v if x >= 0.999),
               sum(1 for x in v if x >= 0.857), v[len(v) // 2] if v else -1))

    out()
    out("=== [author]가 UI 에서 읽어야 하는 것 (API 로는 불가) ===")
    out("  경로: codabench.org 로그인 → 상단 메뉴 'My Submissions'")
    out("        (또는 대회 페이지 → My Submissions 탭). 'Submission ID' 열을 본다.")
    out("  읽을 행 — 아래 넷은 리더보드에 없어서 ID 를 모른다:")
    out("    1) sub2048.zip        Main   0.62 (13/21)   ← 관측된 우리 최고")
    out("    2) sub2048-small.zip  Small  0.57 (8/14)")
    out("    3) h15C-small.zip     Small  0.57            ← lencal 의 대조군")
    out("    4) compass-small.zip  Small  0.57            ← 화이트박스 계열")
    out("  이미 아는 ID 는 다시 찾을 필요 없다:")
    for sid, what in ((1000005, "lencal.zip  Main 0.48"),
                      (1000047, "lencal-small.zip  Small 0.50"),
                      (1005084, "10/20 Small 0.50 — 파일 미확인(flipent-small 로 추정)")):
        out("    %-8s %s" % (sid, what))
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n저장 " + str(OUT))


if __name__ == "__main__":
    main()
