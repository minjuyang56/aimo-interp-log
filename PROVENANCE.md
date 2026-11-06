# Provenance of the commit hashes cited in the report

This public repository is a **snapshot**. The short hashes printed in
`reports/aimo_interp_report.tex` (the `[hash: file]` markers after each claim) therefore do
not resolve here. They refer to the private development history of 268 commits, which cannot
be pushed to GitHub: it tracks 3.5 GB of `.npz` feature caches, six of them above GitHub's
100 MB hard limit. Rewriting that history would change every hash anyway, so the citations
could not be preserved by any route.

What this table gives instead is enough to check each citation in substance -- the full hash,
the date, and the subject line of the commit that produced the number. The measurement logs
themselves are all present in this snapshot under `experiments/results/`, and that is where
every number in the report was read from; `analysis/report_numbers_check.py` verifies that
mechanically (122 of 122 numbers located, 0 unaccounted).

| cited | full hash | date | commit subject |
|---|---|---|---|
| `0b72579` | `0b7257973d3e8c2aeff72facfef4c95caef0b24a` | 2026-10-12 | #DS-RELABEL: 깨끗한 DeepSeek 라벨로 세 평가 재실행 — 판정은 하나도 바뀌지 않았다 |
| `0fe94a2` | `0fe94a249fe45b059c30d419c1b40253dc7fc186` | 2026-10-18 | #LOOP R1 결과: #TT-PERTURB 기각 — 메커니즘은 작동했고 신호가 없다 (0.530, $2.55) |
| `261131c` | `261131caf7c8941525e0dbad656d7bafaa1cf7ec` | 2026-10-12 | #TEMP-SENS 결과: 기각 — 디코딩으로 설명되지 않는다. 그러나 라벨 1/3이 뒤집혔다 |
| `4fcd838` | `4fcd838b485a8fd4bd946eaec88dc25627949b65` | 2026-10-20 | #FLIP-ENT 사전등록 + 커널·평가기 (결과 보기 전 · Kaggle push 직후) |
| `58b2e7d` | `58b2e7dfc306e5af4163ea4cc2a6887b8ff9ec9c` | 2026-10-12 | #SAMP-FEAT 결과: 개선 미입증 — 샘플 평균이 greedy 보다 전부 낮다 |
| `612707b` | `612707bada3fcea33395d69ac202ab1b578de8c6` | 2026-10-15 | #ORG-LABEL 시험2 결과: 전이 미입증 + sub2048 고정 임계가 공식셋에서 무너진다 |
| `64c6ca3` | `64c6ca302f46d547da6d9ce9a7140274b25328ad` | 2026-10-10 | H23 결과: 신호 없음 — 그리고 부호 반전으로 '암기 신호 = 길이 대리' 를 입증 (지휘 #H23) |
| `6e76312` | `6e7631261e65af5c8150538f2ecc9f60b0271015` | 2026-10-10 | Domain v2 45% = 기준 미달 → #STRONG 본실행 중단, '정의 작동 불가'로 정리 (지휘 #DOMAIN-FIX) |
| `73ca8d1` | `73ca8d1d6c0040421b14fbe0a87994d10fc25a48` | 2026-10-18 | #LOOP R5 결과: 기각 — 방향이 33행 밖에서도 안 잡힌다 ($0) |
| `78709ee` | `78709eea9ab513011372e0fbfe70890187350660` | 2026-10-18 | #LOOP R4 기각 → 루프 종료 + 최종 지명 후보 정리 |
| `8adae75` | `8adae75e39c1a77bfb4e014523b7d3dc075ac0fe` | 2026-10-15 | #ORG-LABEL 시험1: 규칙은 82/82 맞지만 in-sample · 진짜 발견은 주최측 해상도 0.01 |
| `98d20b7` | `98d20b7cb706b23260d1670770ba10e580de6266` | 2026-11-02 | #FINAL 1: H-FIN 하이브리드 게이트 통과(0.6818 > null 0.6367) · hfin-small.zip · 제출 ID 3건 확정 |
| `9c7a10e` | `9c7a10e38bc20377f98f4afe0651dc3211ce3e16` | 2026-10-14 | #CMP4 결과: 채택 없음 — 분류기는 병목이 아니었다 (A>B>C>D 단조 하락) |
| `a1e7498` | `a1e749894087f9d24ca47c3bc328fcfb705c9a46` | 2026-10-15 | #ORG-TRAIN 결과: 미채택 — 라벨을 주최측 것으로 바꿔도 결론이 같다 |
| `abf4043` | `abf40434d233056411701eeddb7f63561ab0fac8` | 2026-10-11 | #TOK-AUDIT: vLLM DeepSeek 디토큰화 손상이 라벨 집계까지 번졌다 — 복원 불가 |
| `cb9c291` | `cb9c29156f8fe8b156dcbbf3683ffa4aa253d98a` | 2026-10-15 | #LEN-CAL 제출물 빌드 — 임계가 사전등록 목표에 정확히 맞았다 (업로드 없음) |
| `e0feda3` | `e0feda3a6d1536d20fa8204e7954981d9b10a6d9` | 2026-10-18 | #LABEL-AUDIT 5/5 통과 — 버그 아님. 재계산 안 한다. R3 반대방향 가설 사전등록 |
| `f730878` | `f73087850ce8132b1e091fc2b41371c40887bf5d` | 2026-10-11 | #PROBE-AUDIT (3)+(b) · 외부 테스트 · 분해능 측정 — 전부 탐색, #77 판정 불변 (지휘 #PROBE-AUDIT·#VAR-CHK) |
| `f79c33f` | `f79c33fecb39e01f33e4d169280ffe790e96289c` | 2026-10-18 | #LOOP R3 결과: 부호 없음 — 그리고 세 라운드가 왜 다 실패했는지 숫자로 나왔다 ($0.46) |
| `ff9cb8c` | `ff9cb8cc69680854998d5328f18042cd2867bd67` | 2026-10-12 | 규칙 ② 가 두 번째 결함을 잡았다: extract_answer 가 \[ \boxed{} \] 를 못 뽑는다 |
