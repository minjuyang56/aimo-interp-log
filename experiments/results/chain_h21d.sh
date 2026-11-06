#!/usr/bin/env bash
# H21d·H21p (지휘 #67): 4모델 sens 단계 → H21a~d + H21p 한 표로 분석.
cd /c/side/aimo-interp-log || exit 1
PY=.venv/Scripts/python.exe
L=experiments/results/h21_run.log
for m in Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B; do
  echo "[$(date +%H:%M)] sens $m" >> "$L"
  $PY experiments/h21_uncertainty.py --stage sens --model "$m" >> "$L" 2>&1 || echo "sens 실패 $m" >> "$L"
done
echo "[$(date +%H:%M)] sens 끝 · 전체 분석" >> "$L"
$PY analysis/h21_eval.py --perms 1000 >> "$L" 2>&1
$PY analysis/h21_combo.py --perms 1000 >> "$L" 2>&1
echo "[$(date +%H:%M)] === H21d 완료 ===" >> "$L"
