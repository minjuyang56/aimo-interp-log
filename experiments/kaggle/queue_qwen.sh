# Qwen r2(19번부터 재개)만 슬롯 비면 push. Olmo·DeepSeek은 이미 실행 중.
export PATH="$PWD/.venv/Scripts:$PATH"
until .venv/Scripts/python.exe experiments/kaggle/kaggle_h10b.py --push-only --models Qwen/Qwen3.5-4B --suffix=-r2 --resume-from aimo-h10b-qwen3-5-4b 2>&1 | grep -v -i key | tee -a experiments/results/h10b_queue.log | grep -q "successfully pushed"; do
  echo "  [$(date +%H:%M)] 슬롯 대기 (Qwen r2)" >> experiments/results/h10b_queue.log; sleep 300
done
echo "Qwen r2 push 완료 $(date +%H:%M)" >> experiments/results/h10b_queue.log
