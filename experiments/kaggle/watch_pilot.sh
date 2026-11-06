#!/usr/bin/env bash
# 파일럿 감시 → 통과하면 **본실행을 모델별로 순차 push** (지휘 #71-2). 매 push 전 파서·심볼 점검은 push() 안에 있다.
# 통과 기준: 라벨 중 spurious 비율 >= 15% (v1 본실행은 2% 로 실패했다).
# 로컬에서 선별을 돌리는 Qwen3.5 는 본실행 push 대상에서 빼고, 나머지 3모델만 올린다.
# 사용: bash experiments/kaggle/watch_pilot.sh   (로그 experiments/results/labelgen_watch.log)
cd /c/side/aimo-interp-log || exit 1
export PATH="$PWD/.venv/Scripts:$PATH"
PY=.venv/Scripts/python.exe
L=experiments/results/labelgen_watch.log
USER=[author-kaggle]
# **슬러그를 인자로 받는다.** 1차 때는 파일럿1 슬러그가 하드코딩돼 있었고, 그 평가가 rc=3 로
# 끝나 감시기가 이미 종료한 상태였다 → 파일럿2 에는 반응할 수 없었다(지휘 #72).
PILOT=${1:-aimo-lg-pilot2-skywork-or1-math-7b}
GLOB=${2:-"$PWD/experiments/results/genlab/${PILOT%%-skywork*}-*/**/*_v2_*.json"}
say() { echo "[$(date +%H:%M)] $*" >> "$L"; }

say "=== 파일럿 감시 시작 ($PILOT) ==="
while :; do
  st=$(kaggle kernels status "$USER/$PILOT" 2>/dev/null)
  case "$st" in
    *COMPLETE*) say "파일럿 완료"; break;;
    *ERROR*|*CANCEL*) say "파일럿 실패: $st → 본실행 중단"; exit 1;;
  esac
  say "진행 중 → 대기"
  sleep 300
done

kaggle kernels output "$USER/$PILOT" -p "experiments/results/genlab/$PILOT" >> "$L" 2>&1
$PY analysis/labelgen_pilot_eval.py --glob "$GLOB" >> "$L" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  say "파일럿 기준 미달(rc=$rc) → 본실행 보류. 난이도·변형 강도 조정 후 재파일럿 필요"
  exit 2
fi
say "파일럿 통과 → 본실행 순차 push (Qwen3.5 는 로컬 선별이라 제외)"

COMMON=(--prefix aimo-lg-run- --header genlab_v2_kaggle.py
        --body variants_official.py,variants_nonofficial.py,solve_common.py,genlab_v2.py
        --extra "__N__=200|__K__=4|__MAXNEW__=6144|__SOURCE__=aime,math5|__MAXSEQS__=48|__SCREEN__=None|__CAP__=0")
for m in Skywork/Skywork-OR1-Math-7B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B allenai/Olmo-3-7B-Think; do
  until $PY experiments/kaggle/kaggle_h10b.py --push-only "${COMMON[@]}" --models "$m" 2>&1 \
        | grep -v -i key | tee -a "$L" | grep -q "successfully pushed"; do
    say "슬롯 대기 ($m)"; sleep 300
  done
  say "push 성공 ($m) → 20분 뒤 상태 확인"
  sleep 1200
  s2=$(kaggle kernels status "$USER/aimo-lg-run-$(echo "$m" | sed 's|.*/||; s|\.|-|g; s|_|-|g' | tr 'A-Z' 'a-z')" 2>/dev/null)
  say "상태: $s2"
  case "$s2" in *ERROR*) say "!! $m 20분 내 ERROR → 이후 push 중단"; exit 3;; esac
done
say "=== 본실행 3모델 push 완료 ==="
