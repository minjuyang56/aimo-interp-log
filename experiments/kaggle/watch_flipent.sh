#!/usr/bin/env bash
# #FLIP-ENT 감시기: 60초마다 세 커널 상태를 보고 (a) 빈 슬롯이 생기면 push_flipent.sh 를 한 번
# 부르고 (b) 셋 다 COMPLETE/ERROR 면 산출물을 내려받고 끝낸다.
# 동시 GPU 세션이 2개라 Olmo 는 처음에 못 올라간다 — 그걸 사람이 지키고 있을 필요가 없게 한다.
cd /c/side/aimo-interp-log || exit 1
export PATH="$PWD/.venv/Scripts:$PATH"
PY=.venv/Scripts/python.exe
L=experiments/results/flipent_watch.log
S="aimo-flipent-skywork-or1-math-7b aimo-flipent-deepseek-r1-0528-qwen3-8b aimo-flipent-olmo-3-7b-think"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

for i in $(seq 1 180); do
  done_n=0 run_n=0 miss=0
  for s in $S; do
    st=$(kaggle kernels status "[author-kaggle]/$s" 2>&1)
    case "$st" in
      *COMPLETE*) done_n=$((done_n+1));;
      *ERROR*|*CANCEL*) done_n=$((done_n+1)); say "**$s 이상 종료**: $(echo "$st" | tail -1)";;
      *RUNNING*|*QUEUED*) run_n=$((run_n+1));;
      *) miss=$((miss+1));;
    esac
  done
  say "완료 $done_n · 진행 $run_n · 미생성 $miss"
  if [ "$miss" -gt 0 ] && [ "$run_n" -lt 2 ]; then
    say "빈 슬롯 → push 시도"
    bash experiments/kaggle/push_flipent.sh >>"$L" 2>&1
  fi
  if [ "$done_n" -eq 3 ]; then
    #  `kaggle_h10b.py --output-only` 은 받는 폴더가 results/h10b 로 **하드코딩**돼 있다.
    #  #FLIP-ENT 산출물은 results/flipent 로 따로 받아야 한다(공식 33 P2 와 별칭이 같아서
    #  한 폴더에 섞이면 평가기가 조용히 덮어쓴다) → CLI 를 직접 부른다.
    say "셋 다 종료 → 산출물 회수 (results/flipent)"
    for s in $S; do
      d="experiments/results/flipent/$s"
      mkdir -p "$d"
      kaggle kernels output "[author-kaggle]/$s" -p "$d" >>"$L" 2>&1 && say "회수 $s"
    done
    say "회수 끝 · p2 파일 $(find experiments/results/flipent -name '*_p2.json' | wc -l)개"
    exit 0
  fi
  sleep 60
done
say "180회(3시간) 넘김 — 감시기 종료"
