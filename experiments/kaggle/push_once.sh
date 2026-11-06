#!/usr/bin/env bash
# 본실행 커널을 **한 번만** 시도하고 끝낸다(루프 없음·멱등). 지휘 #73.
# 왜 루프를 안 쓰나: 백그라운드 도구가 2시간에 죽고, 분리 프로세스는 추적이 불안정했다.
# 커널이 8.4시간씩 돌아 push 가 필요한 순간은 드물므로, 깨어날 때마다 이걸 한 번 부르면 충분하다.
cd /c/side/aimo-interp-log || exit 1
export PATH="$PWD/.venv/Scripts:$PATH"
PY=.venv/Scripts/python.exe
L=experiments/results/labelgen_run.log
BODY=variants_official.py,variants_nonofficial.py,solve_common.py,genlab_v2.py
say() { echo "[$(date +%H:%M)] $*" | tee -a "$L"; }
slug_of() { echo "aimo-lg-run$2-$(echo "$1" | sed 's|.*/||; s|\.|-|g; s|_|-|g' | tr 'A-Z' 'a-z')"; }

# **중복 push 방지 (지휘 #75):** AIMO-QUEUE 창에서 queue_labelgen.sh 가 돌고 있으면 그 쪽이 push 를 담당한다.
# 둘이 같이 올리면 실행 중 커널이 새 버전으로 재시작돼 8시간이 날아간다.
if ps -ef 2>/dev/null | grep -q "[q]ueue_labelgen.sh"; then
  say "queue_labelgen.sh 가 실행 중 → push 는 그 쪽이 담당한다. 이 스크립트는 종료."
  exit 0
fi

# **설정은 queue_labelgen.sh(커밋 64a77aa)와 한 글자까지 같아야 한다 (지휘 #78 에서 발견한 충돌).**
# 전에는 여기가 N=50·offset 0/50 이고 큐는 N=40·offset 0/40/80 이었다. 슬러그가 offset 으로 만들어지므로
# 둘 다 `aimo-lg-run0-<모델>` 을 만들고, **내용이 다른 같은 이름**이라 늦게 올린 쪽이 돌던 커널을 재시작시킨다.
# 9h 세션 상한 때문에 [author]가 50→40 으로 줄였으므로 그 값을 단일 출처로 삼는다. Skywork 는 두 슬라이스 COMPLETE.
MODELS="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B allenai/Olmo-3-7B-Think"
OFFSETS="0 40 80"
NSLICE=40
pushed=0
for m in $MODELS; do
  for off in $OFFSETS; do
    s=$(slug_of "$m" "$off")
    cur=$(kaggle kernels status "[author-kaggle]/$s" 2>/dev/null)
    case "$cur" in
      *RUNNING*|*QUEUED*) say "진행 중 $s"; continue;;
      *COMPLETE*)         say "완료 $s"; continue;;
    esac
    if $PY experiments/kaggle/kaggle_h10b.py --push-only --prefix "aimo-lg-run$off-" --header genlab_v2_kaggle.py \
         --body "$BODY" --models "$m" \
         --extra "__N__=$NSLICE|__K__=4|__MAXNEW__=16384|__SOURCE__=aime,math5|__MAXSEQS__=24|__SCREEN__=None|__CAP__=0|__OFFSET__=$off" \
         2>&1 | grep -v -i key | tee -a "$L" | grep -q "successfully pushed"; then
      say "push 성공 $s"; pushed=$((pushed+1))
    else
      say "push 불가(슬롯 꽉 찼거나 오류) $s → 다음 기회에"
      exit 0            # 슬롯이 없으면 더 시도해도 같다
    fi
  done
done
say "이번 호출 push $pushed 건"
