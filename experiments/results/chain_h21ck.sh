#!/usr/bin/env bash
# H21cK (지휘 #69): 4모델 순차 측정 → K 곡선·부트스트랩 분석. GPU 하나.
cd /c/side/aimo-interp-log || exit 1
PY=.venv/Scripts/python.exe
L=experiments/results/h21ck_run.log
for m in Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B; do
  echo "[$(date +%H:%M)] h21cK $m" >> "$L"
  $PY experiments/h21ck.py --model "$m" >> "$L" 2>&1 || echo "실패 $m" >> "$L"
done
echo "[$(date +%H:%M)] 측정 끝 · 분석" >> "$L"
$PY analysis/h21ck_eval.py --perms 1000 --boot 1000 >> "$L" 2>&1
echo "[$(date +%H:%M)] === H21cK 완료 ===" >> "$L"
