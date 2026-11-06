#!/usr/bin/env bash
# H21 (지휘 #65): conf(싼 것) 4모델 → gen(1024토큰) 4모델. GPU 하나라 **순차**.
# 배치는 모델별 KV 로 정한다(토큰당 Qwen 128KB · Skywork 56 · DeepSeek 144 · Olmo 512 → Olmo 만 작게).
cd /c/side/aimo-interp-log || exit 1
PY=.venv/Scripts/python.exe
L=experiments/results/h21_run.log
say() { echo "[$(date +%H:%M)] $*" >> "$L"; }

say "=== conf 단계 시작 (즉답 확신 + 상수변형 하락) ==="
for m in Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B; do
  say "conf $m"
  $PY experiments/h21_uncertainty.py --stage conf --model "$m" >> "$L" 2>&1 || say "conf 실패 $m"
done
say "=== conf 끝 · 분석 ==="
$PY analysis/h21_eval.py --stage conf >> "$L" 2>&1 || say "conf 분석 실패(스크립트 없음?)"

say "=== gen 단계 시작 (1024토큰) ==="
for mb in "Qwen/Qwen3.5-4B 8" "Skywork/Skywork-OR1-Math-7B 8" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B 6" "allenai/Olmo-3-7B-Think 3"; do
  set -- $mb
  say "gen $1 (배치 $2)"
  $PY experiments/h21_uncertainty.py --stage gen --model "$1" --batch "$2" >> "$L" 2>&1 || say "gen 실패 $1"
done
say "=== gen 끝 · 분석 ==="
$PY analysis/h21_eval.py >> "$L" 2>&1 || say "분석 실패"
say "=== H21 완료 ==="
