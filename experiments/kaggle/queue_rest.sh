# 남은 커널을 GPU 슬롯(최대 2)이 비면 올린다. 토큰은 kaggle CLI만 읽는다.
export PATH="$PWD/.venv/Scripts:$PATH"
try_push() { .venv/Scripts/python.exe experiments/kaggle/kaggle_h10b.py --push-only "$@" 2>&1 | grep -v -i key | tee -a experiments/results/h10b_queue.log | grep -q "successfully pushed"; }
until try_push --models Qwen/Qwen3.5-4B --suffix=-r2 --resume-from aimo-h10b-qwen3-5-4b; do echo "  [$(date +%H:%M)] 슬롯 대기 (Qwen r2)" >> experiments/results/h10b_queue.log; sleep 300; done
for m in "allenai/Olmo-3-7B-Think" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"; do
  until try_push --models "$m"; do echo "  [$(date +%H:%M)] 슬롯 대기 ($m)" >> experiments/results/h10b_queue.log; sleep 300; done
done
echo "전부 push 완료 $(date +%H:%M)" >> experiments/results/h10b_queue.log
