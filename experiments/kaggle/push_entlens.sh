#!/usr/bin/env bash
# #LIT-PLAN P2·P3 커널을 **한 번만** 시도하고 끝낸다(루프 없음·멱등 — push_once.sh 와 같은 양식).
# 이미 RUNNING/QUEUED/COMPLETE 인 커널은 건너뛰고 빈 슬롯만큼만 올린다(동시 GPU 세션 2개).
# 깨어날 때마다 한 번 부르면 남은 모델이 채워진다.
cd /c/side/aimo-interp-log || exit 1
export PATH="$PWD/.venv/Scripts:$PATH"
PY=.venv/Scripts/python.exe
L=experiments/results/entlens_push.log
BODY=entlens_problems.py,entlens.py
say() { echo "[$(date +%H:%M)] $*" | tee -a "$L"; }
slug_of() { echo "aimo-entlens-$(echo "$1" | sed 's|.*/||; s|\.|-|g; s|_|-|g' | tr 'A-Z' 'a-z')"; }

# Small 트랙 4모델. **Olmo 만 배치를 쪼갠다** — 토큰당 KV 512 KB(MHA)라 33시퀀스 × 1024토큰이
# 17 GB 로 T4x2 의 가중치 제외 여유에 닿는다. 나머지 셋은 한 배치(0 = 전부).
MODELS="Skywork/Skywork-OR1-Math-7B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B Qwen/Qwen3.5-4B allenai/Olmo-3-7B-Think"
pushed=0
for m in $MODELS; do
  s=$(slug_of "$m")
  case "$m" in *Olmo*) B=11;; *) B=0;; esac
  cur=$(kaggle kernels status "[author-kaggle]/$s" 2>/dev/null)
  case "$cur" in
    *RUNNING*|*QUEUED*) say "진행 중 $s"; continue;;
    *COMPLETE*)         say "완료 $s"; continue;;
  esac
  if $PY experiments/kaggle/kaggle_h10b.py --push-only --prefix "aimo-entlens-" \
       --header entlens_kaggle.py --body "$BODY" --models "$m" --batch "$B" \
       --extra "__MAXNEW__=1024|__NPROB__=0" \
       2>&1 | grep -v -i key | tee -a "$L" | grep -q "successfully pushed"; then
    say "push 성공 $s (batch $B)"; pushed=$((pushed+1))
  else
    say "push 불가(슬롯 꽉 찼거나 오류) $s → 다음 기회에"
    exit 0
  fi
done
say "이번 호출 push $pushed 건"
