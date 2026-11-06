#!/usr/bin/env bash
# 본실행 (지휘 #73→#3 재개): cap 16384 · M=100/모델 · 남은 2모델(deepseek·olmo) · 슬라이스 40문제(offset 0/40/80)
# skywork(run0/run50)는 COMPLETE라 제외. 9h 세션 초과로 출력 잃는 걸 막으려 50→40 축소. 동시 2슬롯, idempotent skip.
# 커널 6개(2모델×offset3)를 슬롯 비는 대로 순차 push. push 전 파서·외부 심볼 점검은 push() 안에서 돈다.
cd /c/side/aimo-interp-log || exit 1
export PATH="$PWD/.venv/Scripts:$PATH"
PY=.venv/Scripts/python.exe
L=experiments/results/labelgen_run.log
say() { echo "[$(date +%H:%M)] $*" >> "$L"; }
BODY=variants_official.py,variants_nonofficial.py,solve_common.py,genlab_v2.py

say "=== 본실행 재개 push 시작 (cap 16384 · 슬라이스 40 · offset 0/40/80 · 남은 2모델) ==="
slug_of() { echo "aimo-lg-run$2-$(echo "$1" | sed 's|.*/||; s|\.|-|g; s|_|-|g' | tr 'A-Z' 'a-z')"; }
for m in deepseek-ai/DeepSeek-R1-0528-Qwen3-8B allenai/Olmo-3-7B-Think; do
  for off in 0 40 80; do
    # **이미 올라간 커널은 건너뛴다.** 큐가 2시간 제한에 죽어도 다시 띄우면 이어지게(idempotent).
    cur=$(kaggle kernels status "[author-kaggle]/$(slug_of "$m" "$off")" 2>/dev/null)
    case "$cur" in
      *RUNNING*|*COMPLETE*|*QUEUED*) say "건너뜀 ($m off=$off → $cur)"; continue;;
    esac
    pfx="aimo-lg-run$off-"
    until $PY experiments/kaggle/kaggle_h10b.py --push-only --prefix "$pfx" --header genlab_v2_kaggle.py \
          --body "$BODY" --models "$m" \
          --extra "__N__=40|__K__=4|__MAXNEW__=16384|__SOURCE__=aime,math5|__MAXSEQS__=24|__SCREEN__=None|__CAP__=0|__OFFSET__=$off" \
          2>&1 | grep -v -i key | tee -a "$L" | grep -q "successfully pushed"; do
      say "슬롯 대기 ($m off=$off)"; sleep 300
    done
    say "push 성공 ($m off=$off) → 25분 뒤 상태 확인"
    sleep 1500
    slug="$pfx$(echo "$m" | sed 's|.*/||; s|\.|-|g; s|_|-|g' | tr 'A-Z' 'a-z')"
    st=$(kaggle kernels status "[author-kaggle]/$slug" 2>/dev/null)
    say "상태: $slug → $st"
    case "$st" in *ERROR*) say "!! ERROR → push 중단"; exit 3;; esac
  done
done
say "=== 본실행 6커널 push 완료 ==="
