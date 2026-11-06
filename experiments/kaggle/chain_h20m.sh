#!/usr/bin/env bash
# H20-M (지휘 #63): 세 커널이 끝나면 내려받아 사전등록 분석을 돌린다.
# 사용: bash experiments/kaggle/chain_h20m.sh   (로그 experiments/results/h20m_chain.log)
cd "$(dirname "$0")/../.." || exit 1
export PATH="$PWD/.venv/Scripts:$PATH"
LOG=experiments/results/h20m_chain.log
PY=.venv/Scripts/python.exe
USER=[author-kaggle]
SLUGS=(aimo-h20m-skywork-or1-math-7b aimo-h20m-deepseek-r1-0528-qwen3-8b aimo-h20m-olmo-3-7b-think)

say() { echo "  [$(date +%H:%M)] $*" >> "$LOG"; }
say "=== 대기 시작 (3커널) ==="
while :; do
  left=0
  for s in "${SLUGS[@]}"; do
    st=$(kaggle kernels status "$USER/$s" 2>/dev/null)
    case "$st" in *RUNNING*|*QUEUED*) left=$((left+1));; esac
  done
  [ "$left" -eq 0 ] && break
  say "실행 중 $left 개 → 대기"
  sleep 300
done
for s in "${SLUGS[@]}"; do
  st=$(kaggle kernels status "$USER/$s" 2>/dev/null)
  say "$s → $st"
  kaggle kernels output "$USER/$s" -p "experiments/results/h20m/$s" >> "$LOG" 2>&1
done
say "내려받기 완료 → 분석"
$PY analysis/h20m_eval.py --perms 1000 >> "$LOG" 2>&1
say "=== 분석 완료 (experiments/results/h20m_report.txt) ==="
