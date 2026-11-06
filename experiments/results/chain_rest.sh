# 현재 H12(DeepSeek·Qwen) 끝날 때까지 대기 → 죽었던 Skywork·Olmo 재실행 → H12 평가 → H13 4모델 → H9 4모델 + H9가중 나침반
while pgrep -f "h12_patch.py --extract" > /dev/null 2>&1; do sleep 60; done
echo "H12 재실행(Skywork·Olmo) $(date +%H:%M)"
for m in "Skywork/Skywork-OR1-Math-7B" "allenai/Olmo-3-7B-Think"; do
  echo "=== $m ==="; .venv/Scripts/python.exe experiments/h12_patch.py --extract --model "$m" --set v2 2>&1 | grep -vE "Loading weights|rope_parameters|triton|fast path|flash-linear|causal-conv1d|^\s*$" | tail -3
done
echo "=== H12 evaluate ==="; .venv/Scripts/python.exe experiments/h12_patch.py --set v2 2>&1 | grep -vE "^\s*$"
echo "=== H13 시작 $(date +%H:%M) ==="
for m in "Skywork/Skywork-OR1-Math-7B" "allenai/Olmo-3-7B-Think" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" "Qwen/Qwen3.5-4B"; do
  echo "=== $m ==="; .venv/Scripts/python.exe experiments/h13_subspace.py --extract --model "$m" 2>&1 | grep -vE "Loading weights|rope_parameters|triton|fast path|flash-linear|causal-conv1d|^\s*$" | tail -3
done
echo "=== H13 evaluate ==="; .venv/Scripts/python.exe experiments/h13_subspace.py 2>&1 | grep -vE "^\s*$"
echo "=== H9 시작 $(date +%H:%M) ==="
for m in "Skywork/Skywork-OR1-Math-7B" "allenai/Olmo-3-7B-Think" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" "Qwen/Qwen3.5-4B"; do
  echo "=== $m ==="; .venv/Scripts/python.exe experiments/h9_essence.py --extract --model "$m" --set v2 2>&1 | grep -vE "Loading weights|rope_parameters|triton|fast path|flash-linear|causal-conv1d|^\s*$" | tail -3
done
echo "=== H9 evaluate ==="; .venv/Scripts/python.exe experiments/h9_essence.py --set v2 2>&1 | grep -vE "^\s*$"
echo "=== H10 나침반 (H9 가중치) ==="; .venv/Scripts/python.exe experiments/h10_compass.py --src h9 --set v2 2>&1 | grep -vE "^\s*$"
echo "=== 끝 $(date +%H:%M) ==="
