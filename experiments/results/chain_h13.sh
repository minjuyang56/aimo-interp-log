until [ -f experiments/results/h12_report_v2.txt ]; do sleep 30; done
echo "H13 시작 $(date +%H:%M)"
for m in "Skywork/Skywork-OR1-Math-7B" "allenai/Olmo-3-7B-Think" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" "Qwen/Qwen3.5-4B"; do
  echo "=== $m ==="; .venv/Scripts/python.exe experiments/h13_subspace.py --extract --model "$m" 2>&1 | grep -vE "Loading weights|rope_parameters|triton|fast path|flash-linear|causal-conv1d|^\s*$" | tail -4
done
echo "=== evaluate ==="; .venv/Scripts/python.exe experiments/h13_subspace.py 2>&1 | grep -vE "^\s*$"
