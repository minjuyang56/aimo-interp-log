# 4개 커널이 끝나면 내려받고 h10b_eval + h10b_ext 자동 실행 (지휘 #23). 10분마다 상태 확인.
export PATH="$PWD/.venv/Scripts:$PATH"
declare -A SLUG=( [Qwen/Qwen3.5-4B]=aimo-h10b-qwen3-5-4b-r2 [Skywork/Skywork-OR1-Math-7B]=aimo-h10b-skywork-or1-math-7b [allenai/Olmo-3-7B-Think]=aimo-h10b-olmo-3-7b-think [deepseek-ai/DeepSeek-R1-0528-Qwen3-8B]=aimo-h10b-deepseek-r1-0528-qwen3-8b )
declare -A DONE=()
while true; do
  for m in "${!SLUG[@]}"; do
    s=${SLUG[$m]}; [ -n "${DONE[$s]}" ] && continue
    st=$(kaggle kernels status "[author-kaggle]/$s" 2>&1 | grep -oE "COMPLETE|ERROR|RUNNING|CANCEL|404")
    echo "[$(date +%H:%M)] $s $st"
    if [ "$st" = "COMPLETE" ]; then
      ( kaggle kernels output "[author-kaggle]/$s" -p "experiments/results/h10b/$s" 2>&1 | grep -E "downloaded|rror" ) && DONE[$s]=1
    elif [ "$st" = "ERROR" ] || [ "$st" = "CANCEL" ]; then
      echo "  !! $s 실패 — 수동 확인 필요"; DONE[$s]=fail
    fi
  done
  n=0; for s in "${SLUG[@]}"; do [ "${DONE[$s]}" = "1" ] && n=$((n+1)); done
  nf=0; for s in "${SLUG[@]}"; do [ "${DONE[$s]}" = "fail" ] && nf=$((nf+1)); done
  if [ $((n+nf)) -ge 4 ]; then break; fi
  sleep 600
done
echo "=== 내려받음 $n/4, 실패 $nf → 평가 $(date +%H:%M) ==="
.venv/Scripts/python.exe experiments/h10b_eval.py 2>&1 | grep -vE "Warning|out\(f"
echo "=== 확장 (A)(B) ==="
.venv/Scripts/python.exe experiments/h10b_ext.py 2>&1 | grep -vE "Warning|out\(f"
echo "=== 끝 $(date +%H:%M) ==="
