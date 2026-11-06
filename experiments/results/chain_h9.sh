until [ -f experiments/results/h13_report.txt ]; do sleep 30; done
echo "H9 시작 $(date +%H:%M)"
for m in "Skywork/Skywork-OR1-Math-7B" "allenai/Olmo-3-7B-Think" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" "Qwen/Qwen3.5-4B"; do
  echo "=== $m ==="; .venv/Scripts/python.exe experiments/h9_essence.py --extract --model "$m" --set v2 2>&1 | grep -vE "Loading weights|rope_parameters|triton|fast path|flash-linear|causal-conv1d|^\s*$" | tail -4
done
echo "=== evaluate H9 ==="; .venv/Scripts/python.exe experiments/h9_essence.py --set v2 2>&1 | grep -vE "^\s*$"
echo "=== H10 compass with H9 weights ==="; .venv/Scripts/python.exe experiments/h10_compass.py --src h9 --set v2 2>&1 | grep -vE "^\s*$"
