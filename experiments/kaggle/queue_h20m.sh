#!/usr/bin/env bash
# H20-M (지휘 #63) 순차 push. GPU 동시 세션 2개 제한 + 2026-10-04 Kaggle 이미지 변경(torchaudio CUDA 불일치) 대응.
# 절차: 한 모델을 올리고 **18분 뒤 상태 확인** 후 다음을 올린다. 8분은 짧았다 — vLLM 로드 중이라
# RUNNING 으로 보였고, 실제 실패(NameError)는 6~10분 뒤에 났다.
# 사용: bash experiments/kaggle/queue_h20m.sh   (로그 experiments/results/h20m_queue.log)
cd "$(dirname "$0")/../.." || exit 1
LOG=experiments/results/h20m_queue.log
PY=.venv/Scripts/python.exe
# body 순서 주의: genlab_h14.py 가 마지막이어야 한다(그 끝에서 main() 이 바로 실행된다).
COMMON=(--prefix aimo-h20m- --header genlab_h14_kaggle.py
        --body variants_official.py,solve_common.py,genlab_h14.py
        --extra "__K__=1,__MAXNEW__=12288,__MAXSEQS__=8,__KINDS__=orig")
ALL=("Skywork/Skywork-OR1-Math-7B" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" "allenai/Olmo-3-7B-Think")

say() { echo "  [$(date +%H:%M)] $*" >> "$LOG"; }
st()  { $PY experiments/kaggle/kaggle_h10b.py --prefix aimo-h20m- --status-only --models "$1" 2>/dev/null | tail -1; }
try_push() {                            # 성공 시 0
  $PY experiments/kaggle/kaggle_h10b.py --push-only "${COMMON[@]}" --models "$1" 2>&1 \
    | grep -v -i key | tee -a "$LOG" | grep -q "successfully pushed"
}

say "=== 순차 push 시작 ==="
for m in "${ALL[@]}"; do
  until try_push "$m"; do say "슬롯 대기 ($m)"; sleep 240; done
  say "push 성공 ($m) → 18분 뒤 상태 확인(vLLM 로드 뒤 생성까지 들어갔는지)"
  sleep 1080
  s=$(st "$m"); say "상태: $s"
  case "$s" in
    *ERROR*) say "!! $m 이 8분 안에 ERROR — 환경 문제 가능. 다음 모델 push 중단"; exit 1;;
  esac
done
say "=== 세 모델 push 완료 ==="
