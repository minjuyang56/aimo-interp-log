"""Codabench 리더보드를 받아 트랙(task)별로 갈라서 읽는다.

핵심: 리더보드는 한 칼럼에 여러 task의 점수를 섞어 표시한다. task가 다르면
채점 대상 데이터가 다르므로 점수를 나란히 비교하면 안 된다.

한 제출에는 점수가 세 개 붙는다 (index 1·2·3):
    accuracy · coverage · invalid_predictions
coverage가 0인 제출은 케이스 전량이 무효 처리된 것이고, 그때
invalid_predictions 값이 곧 그 task의 케이스 수 N이다.

N을 알면 모든 점수를 k/N으로 읽을 수 있다. 합이 1.0인 두 점수가 다수를
차지하면 그게 상수 예측(전부 True / 전부 False)이다.

사용:  python analysis/leaderboard.py            # 본선 phase
       python analysis/leaderboard.py 26647      # 워밍업 phase
"""

import json
import sys
import urllib.request
from collections import Counter, defaultdict

PHASE_ID = 30038  # Competition phase (2026-09-01 ~ 2026-11-01). 워밍업은 26647
TRACK = {35766: "Main", 35767: "Small"}
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(phase_id):
    url = f"https://www.codabench.org/api/phases/{phase_id}/get_leaderboard/?page_size=500"
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def infer_n(rows):
    """전량 무효 제출의 invalid_predictions가 N이다. 없으면 k/N 산수로 후보를 낸다."""
    zero = {int(r["inv"]) for r in rows if r["cov"] == 0 and r["inv"] is not None}
    if zero:
        return sorted(zero)
    scores = sorted({round(r["acc"], 2) for r in rows if r["cov"]})
    return [n for n in range(2, 300)
            if all(abs(s * n - round(s * n)) < 0.006 * n for s in scores)][:5]


def main():
    phase_id = int(sys.argv[1]) if len(sys.argv) > 1 else PHASE_ID
    board = fetch(phase_id)
    names = {t["id"]: t["name"] for t in board.get("tasks", [])}
    print(f"phase: {board['title']} (id {board['id']})   제출 {board.get('count', len(board['submissions']))}건")
    for tid, name in names.items():
        print(f"  task {tid}  {name}")
    print()

    by_task = defaultdict(list)
    for entry in board["submissions"]:
        cols = {s["index"]: (float(s["score"]), s.get("task_id")) for s in entry["scores"]}
        if 1 not in cols:
            continue
        acc, tid = cols[1]
        by_task[tid].append({
            "acc": acc,
            "cov": cols.get(2, (None,))[0],
            "inv": cols.get(3, (None,))[0],
            "owner": entry["owner"],
            "date": entry["created_when"][:10],
        })

    for tid in sorted(by_task):
        rows = by_task[tid]
        scored = [r for r in rows if r["cov"]]
        dead = [r for r in rows if not r["cov"]]
        tag = TRACK.get(tid, "")
        dates = sorted(r["date"] for r in rows)
        print("=" * 64)
        print(f"task {tid} {tag}   {dates[0]} ~ {dates[-1]}")
        print(f"  제출 {len(rows)}건 = 채점 {len(scored)} + 전량무효 {len(dead)}"
              f"   참가자 {len({r['owner'] for r in rows})}명")

        n_candidates = infer_n(rows)
        n = n_candidates[0] if n_candidates else None
        print(f"  N = {n_candidates}  {'(전량무효 제출의 invalid 값)' if dead else '(k/N 산수 후보)'}")

        dist = Counter(round(r["acc"], 2) for r in scored)
        print("  분포:")
        for score, count in sorted(dist.items(), reverse=True):
            k = f"{round(score * n):2d}/{n}" if n else ""
            const = ""
            partner = round(1 - score, 2)
            if partner in dist and dist[score] + dist[partner] >= 0.4 * len(scored):
                const = "  ← 상수 예측"
            print(f"    {score:.2f} {k:>7}  ×{count:<3d}{const}")

        best = {}
        for r in sorted(scored, key=lambda r: (-r["acc"], r["date"])):
            best.setdefault(r["owner"], r)
        print("  참가자별 최고:")
        for r in list(best.values())[:10]:
            print(f"    {r['acc']:.2f}  {r['owner']:<18} {r['date']}")
        print()


if __name__ == "__main__":
    main()
