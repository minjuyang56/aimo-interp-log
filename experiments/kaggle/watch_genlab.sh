#!/usr/bin/env bash
# 자체 라벨 생성 감시자 (지휘 #31) — 세션이 멈춰도 혼자 돈다. 파일 기반 상태.
#   1. 4모델 커널 상태를 5분마다 확인
#   2. COMPLETE 이면 내려받고(한 번만) 사전등록 분석(h10c_eval.py) 실행 → results/genlab_watch.log
#   3. 아직 안 올린 모델은 GPU 슬롯이 비면 push (동시 2개 제한이라 실패하면 다음 회차에 재시도)
#   4. Skywork 공식 라벨 spurious 비율 < 15% 면 level 5 보충 커널(태그 -l5, 50문제)을 4모델 전부에 push
# 상태 파일: results/genlab_state/<slug>.{pushed,downloaded,analyzed}
set -u
cd "$(dirname "$0")/../.." || exit 1
export PATH="$PWD/.venv/Scripts:$PATH"
LOG=experiments/results/genlab_watch.log
ST=experiments/results/genlab_state
mkdir -p "$ST" experiments/results/genlab
PY=.venv/Scripts/python.exe
RUNNER=experiments/kaggle/kaggle_h10b.py
BODY=variants_official.py,variants_nonofficial.py,genlab_vllm.py
MODELS=("Skywork/Skywork-OR1-Math-7B" "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" "allenai/Olmo-3-7B-Think" "Qwen/Qwen3.5-4B")
say() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }

slug_of() { echo "aimo-genlab-$(echo "$1" | sed 's#.*/##' | tr 'A-Z' 'a-z' | tr '._' '--')$2"; }

push_run() {   # $1 모델 · $2 접미 · $3 N · $4 levels · $5 tag
  $PY "$RUNNER" --push-only --models "$1" --prefix aimo-genlab- --suffix="$2" \
      --header genlab_vllm_kaggle.py --body "$BODY" \
      --extra "__N__=$3,__MAXNEW__=6144,__MAXSEQS__=48,__NOFF__=3,__NUNOFF__=2,__LEVELS__=$4,__TAG__=$5" 2>&1 | grep -v -i key | grep -E "pushed|error"
}

# 이미 올린 두 커널은 pushed 로 표시
touch "$ST/$(slug_of "${MODELS[0]}" -run).pushed" "$ST/$(slug_of "${MODELS[1]}" -run).pushed"

for round in $(seq 1 200); do
  for m in "${MODELS[@]}"; do
    for suf in -run -l5; do
      s=$(slug_of "$m" "$suf")
      # l5 보충은 트리거 파일이 있어야 시작
      if [ "$suf" = "-l5" ] && [ ! -f "$ST/l5.trigger" ]; then continue; fi
      if [ ! -f "$ST/$s.pushed" ]; then
        if [ "$suf" = "-l5" ]; then out=$(push_run "$m" "$suf" 50 "Level 5" "-l5"); else out=$(push_run "$m" "$suf" 150 "Level 3|Level 4" ""); fi
        if echo "$out" | grep -q "successfully pushed"; then touch "$ST/$s.pushed"; say "push 성공 $s"; else say "push 대기 $s ($(echo "$out" | head -1))"; fi
        continue
      fi
      st=$(kaggle kernels status "[author-kaggle]/$s" 2>&1 | grep -oE "COMPLETE|ERROR|RUNNING|QUEUED|CANCEL")
      if [ "$st" = "COMPLETE" ] && [ ! -f "$ST/$s.downloaded" ]; then
        say "완료 $s → 내려받기"
        kaggle kernels output "[author-kaggle]/$s" -p "experiments/results/genlab/$s" > /dev/null 2>&1
        # 커널 출력의 npz 를 분석이 찾는 위치로 모으기
        find "experiments/results/genlab/$s" -name "*.npz" -exec cp {} experiments/results/genlab/ \; 2>/dev/null
        touch "$ST/$s.downloaded"
        say "분석 시작 $s"
        $PY experiments/h10c_eval.py --perms 1000 >> "$LOG" 2>&1
        touch "$ST/$s.analyzed"
        # Skywork 본실행이면 spurious 비율로 level 5 보충 결정
        if [ "$s" = "$(slug_of "${MODELS[0]}" -run)" ] && [ ! -f "$ST/l5.trigger" ] && [ ! -f "$ST/l5.decided" ]; then
          rate=$($PY - <<'EOF'
import glob, numpy as np
r = ""
for f in glob.glob("experiments/results/genlab/*.npz"):
    if "Skywork" not in f: continue
    st = list(np.load(f, allow_pickle=True)["store"])
    lab = [s for s in st if s.get("label_official") is not None]
    if lab:
        r = f"{sum(s['label_official'] is False for s in lab) / len(lab):.4f} {len(lab)}"
print(r)
EOF
)
          touch "$ST/l5.decided"
          say "Skywork 공식 라벨 spurious 비율 = $rate (비율 라벨수)"
          frac=$(echo "$rate" | awk '{print $1}')
          if [ -n "$frac" ] && awk -v f="$frac" 'BEGIN{exit !(f < 0.15)}'; then
            touch "$ST/l5.trigger"; say "15% 미만 → level 5 보충 50문제를 4모델에 push 예약"
          else
            say "15% 이상 → level 5 보충 불필요"
          fi
        fi
      elif [ "$st" = "ERROR" ] || [ "$st" = "CANCEL" ]; then
        if [ ! -f "$ST/$s.failed" ]; then touch "$ST/$s.failed"; say "실패 $s — 수동 확인 필요"; fi
      fi
    done
  done
  done_n=$(ls "$ST" 2>/dev/null | grep -c ".analyzed" || true)
  want=4; [ -f "$ST/l5.trigger" ] && want=8
  if [ "$done_n" -ge "$want" ]; then say "전부 완료 ($done_n/$want)"; break; fi
  sleep 300
done
say "감시자 종료"
