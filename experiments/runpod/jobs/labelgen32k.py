"""라벨 생성 cap 32768 — RunPod H100. (지휘 #LG-RP ②③)

왜 RunPod 로 옮겼나 (`analysis/labelgen_cap32k.py` 의 계산)
  Kaggle T4×2 에서는 cap 32768 이 **Olmo 에게 불가능하다** — 시퀀스 하나의 KV 가 17.0 GB 인데
  T4×2 의 KV 풀이 12.2 GB 다. vLLM 은 '전체 문맥 하나가 캐시에 안 들어간다' 며 시작조차 못 한다.
  DeepSeek 도 9시간 세션에 8문제뿐이라 주간 30h 를 다 써도 24문제다(목표 120).
  H100 80 GB 는 KV 풀이 약 55 GB 라 Olmo 도 들어간다(17.0 GB/시퀀스).

**커널과 같은 코드를 쓴다.** `experiments/kaggle/genlab_v2.py` 를 그대로 payload 에 넣고,
Kaggle 커널 헤더가 하던 것처럼 `variants_official.py` 를 **같은 네임스페이스에 먼저 넣는다**
(genlab_v2 는 `official_variants` 를 주입받는 전제로 쓰여 있다 — `# noqa: F821`).
여기서 로직을 다시 쓰면 Kaggle 라벨과 RunPod 라벨이 다른 코드로 만들어져 섞을 수 없게 된다.

문제 범위는 **Kaggle 큐와 겹치지 않게** 바깥에서 `--pool-offset` 으로 준다.
라벨마다 `gen_cap` 이 기록되므로(지휘 #LG-32k-3) 나중에 16384/32768 라벨을 가릴 수 있다.

사용(파드 안):
  python3 labelgen32k.py --model allenai/Olmo-3-7B-Think --n 40 --offset 80 --max-new 32768
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/workspace/out")


def log(m):
    print(f"[단계] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40, help="이 잡이 처리할 문제 수")
    ap.add_argument("--offset", type=int, default=0, help="풀 앞쪽 건너뛸 수 (Kaggle 큐와 겹치지 않게)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=32768)
    ap.add_argument("--source", default="aime,math5")
    ap.add_argument("--max-seqs", type=int, default=32)
    ap.add_argument("--strong", default=None,
                    help="센 섭동 파일(strong_variants.json). 주면 기존 3유형에 **덧붙여** 6유형이 된다. "
                         "genlab_v2 는 고치지 않고 official_variants 를 감싸서 주입한다 — "
                         "Kaggle 라벨과 같은 코드여야 하기 때문이다.")
    ap.add_argument("--smoke-tokens", type=int, default=0,
                    help="0 보다 크면 **본 생성 전에** 그 토큰 수로 1문제 짧은 생성을 돌려 "
                         "디토큰화 마커·경고가 0 인지 확인하고, 아니면 본 생성을 하지 않고 죽는다 "
                         "(지휘 #DS-RELABEL). 엔진을 한 번 더 올리는 대가($0.1~0.2)로 "
                         "수 시간·$20 짜리 실행이 손상 위에서 도는 것을 막는다.")
    ap.add_argument("--screen", default=None,
                    help="이전 실행의 *_screen.json 글롭. 주면 선별을 건너뛰고 **변형만** 돈다 "
                         "— 수명 상한에 걸린 실행을 싸게 이어받는 길이다(선별이 전체의 ~37%).")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # **vLLM 의 torch.compile 경로를 끈다 (2026-10-10 Olmo 에서 실패).**
    # 증상: AssertionError: Source mismatch for collections.abc (torch._dynamo/package.py add_code).
    # vllm 0.31.0 + torch 2.13 + Olmo3 조합에서 엔진 초기화가 죽는다. Skywork(Qwen2)는 같은 이미지에서
    # 통과했으므로 아키텍처 특정 문제다. enforce_eager 면 컴파일을 아예 안 해서 피할 수 있다 —
    # 디코딩이 조금 느려지는 대가로 돌아가게 만드는 교환이다.
    # genlab_v2.py 는 **고치지 않는다**(Kaggle 라벨과 같은 코드여야 한다) → 여기서 LLM 을 감싼다.
    # ---- 디코딩 민감도 (지휘 #TEMP-SENS, 2026-10-12) ----
    # **치환은 여기서 하지 않는다.** 처음엔 `vllm.SamplingParams` 를 감싸는 함수로 바꿨는데
    # vllm 0.31.0 이 **lazy import 중에 그 이름을 타입으로 평가**한다:
    #   offline_utils.py: `_P = TypeVar("_P", bound=SamplingParams | PoolingParams | None)`
    #   → TypeError: unsupported operand type(s) for |: 'function' and 'StructMeta'
    # 클래스 자리에 함수를 끼우면 `|`(타입 유니온)가 깨진다. 1분 만에 죽고 $0.28 을 썼다.
    # → 오버라이드를 **`genlab_v2.Gen.samples` 안의 SamplingParams 생성 지점**으로 옮겼다
    #   (`TS_TEMP`/`TS_TOPK` 환경변수. 없으면 기존 동작과 글자 그대로 같다).
    if os.environ.get("TS_TEMP"):
        log(f"[#TEMP-SENS] temperature {os.environ['TS_TEMP']} · "
            f"top_k {os.environ.get('TS_TOPK', '40')} — 치환은 genlab_v2.Gen.samples 안에서 한다")

    if os.environ.get("LG_EAGER", "1") == "1":
        import vllm
        _LLM = vllm.LLM

        def _patched(*args, **kw):
            kw.setdefault("enforce_eager", True)
            log("vLLM enforce_eager=True 로 생성 (컴파일 경로 우회)")
            return _LLM(*args, **kw)

        vllm.LLM = _patched

    # 커널 헤더와 같은 네임스페이스 주입
    ns = {"__name__": "genlab_v2_runpod"}
    # Kaggle 커널의 BODY 와 **같은 네 파일·같은 순서**다
    # (genlab_v2 는 solve_common 의 SOLVE_PROMPT·extract_answer 를 주입받는 전제로 쓰여 있다 —
    #  두 파일만 넣었다가 미정의 심볼 4개가 잡혔다. 심볼 점검을 꼭 돌려야 하는 이유).
    for f in ("variants_official.py", "variants_nonofficial.py", "solve_common.py", "genlab_v2.py"):
        src = (HERE / f) if (HERE / f).is_file() else Path(f)
        log(f"로드 {src.name}")
        exec(compile(src.read_text(encoding="utf-8"), str(src), "exec"), ns)
    run = ns["run"]

    # ---- 문제 고정 + BAND 해제 (지휘 #TEMP-SENS) ----
    # **왜 `--n 40 --offset 40` 으로 불러야 하는가**: `load_pool(n)` 은 permutation 의 앞 n 개를
    # 뽑아 정렬한 뒤 pidx 0..n-1 을 붙인다 → **n 이 달라지면 같은 pidx 가 다른 문제를 가리킨다.**
    # 원래 Olmo off40 은 `--n 40 --offset 40`(= load_pool(80) 의 pidx 40~79)로 만들어졌으므로
    # 그 호출을 그대로 유지하고, 반환 리스트에서 **선정되지 않은 뒤쪽만** 걷어낸다
    # (앞 40개는 그대로 둬야 run() 의 `[pool_offset:]` 슬라이스가 안 밀린다).
    # 로컬 선행 검증: load_pool(80) 의 pidx 가 Olmo 라벨 31문제와 **문제 텍스트 31/31 일치**.
    if os.environ.get("TS_PIDX"):
        _keep = set(int(x) for x in os.environ["TS_PIDX"].replace(" ", "").split(",") if x)
        _lp = ns["load_pool"]

        def _sel_pool(n_problems, source="aime", seed=0):
            rows = _lp(n_problems, source, seed)
            if n_problems < a.n + a.offset:
                # 스모크 게이트가 `load_pool(offset+1)` 로 1문제만 가져오는 경로. 거기서 선정
                # 필터를 걸면 tail 이 0개가 되어 죽는다. 스모크는 디토큰화 마커만 보는 것이므로
                # 어느 문제든 무관하다 → 통과시킨다. 본 생성은 n+offset 으로 부르니 영향 없다.
                return rows
            head = [r for r in rows if r["pidx"] < a.offset]
            tail = [r for r in rows if r["pidx"] >= a.offset and r["pidx"] in _keep]
            if len(tail) != len(_keep) or len(head) != a.offset:
                raise SystemExit(f"!! 문제 고정 실패: head {len(head)}(기대 {a.offset}) · "
                                 f"tail {len(tail)}(기대 {len(_keep)}) — 풀이 달라졌다. 돌리지 않는다")
            log(f"[#TEMP-SENS] 문제 고정 {len(tail)}개: {[r['pidx'] for r in tail]}")
            return head + tail

        ns["load_pool"] = _sel_pool
        # **BAND 해제.** 쌍별 라벨 전환표를 만들려면 선정 20문제 **전부** 변형을 돌아야 한다.
        # T=1.0 에서 base_acc 가 0.5 아래로 떨어지는 것이 바로 측정하려는 효과인데, 그 문제를
        # 채택 대역에서 떨어뜨리면 그 효과가 비교에서 사라진다(= 생존 편향).
        # 대가: base_acc 가 낮은 문제의 max_drop 은 '원본부터 못 푼다' 와 섞인다 →
        # 분석에서 (ㄱ) 20문제 전체 · (ㄴ) 기존 BAND 통과분만 **둘 다** 보고한다.
        ns["BAND"] = (0.0, 1.0)
        log("[#TEMP-SENS] BAND 해제 (0.0, 1.0) — 선정 문제 전부 변형을 돈다")


    # ---- 센 섭동 주입 (지휘 #STRONG)
    if a.strong:
        import glob as _g2
        sp = a.strong if _g2.glob(a.strong) else Path(a.strong).name
        if not _g2.glob(sp):
            raise SystemExit(f"센 섭동 파일을 못 찾았다: {a.strong}")
        STRONG = json.loads(Path(_g2.glob(sp)[0]).read_text(encoding="utf-8"))
        _orig_ov = ns["official_variants"]

        def _with_strong(text, seed=0, kinds=None):
            """genlab_v2 는 `official_variants(problem, seed=pidx)` 로 부른다 → seed 가 pidx 다."""
            got = list(_orig_ov(text, seed=seed, kinds=kinds))
            for x in STRONG.get(str(seed), []):
                if x.get("text") and x["text"] != text:
                    got.append((x["kind"], x["text"]))
            return got

        ns["official_variants"] = _with_strong
        n = sum(len(v) for v in STRONG.values())
        log(f"센 섭동 주입: 문제 {len(STRONG)}개 · 변형 {n}건 → 유형 합계 3+N")

    if a.smoke_tokens > 0:
        # **생성 전 스모크 (지휘 #DS-RELABEL).** 묻는 것 하나: 이 파드·이 vLLM 버전에서
        # 우리가 쓰는 텍스트에 BPE 마커가 0 인가. 2026-10-11 에 DeepSeek 라벨 전부가
        # 손상 텍스트 위에서 집계됐고 **텍스트를 저장하지 않아 복원이 불가능**했다.
        # 그래서 비싼 단계에 들어가기 전에 싼 생성으로 확인하고, 아니면 **멈춘다**.
        Gen, chat, load_pool = ns["Gen"], ns["chat"], ns["load_pool"]
        marks_in, acc_of = ns["marks_in"], ns["acc_of"]
        probs = load_pool(a.offset + 1, a.source)[a.offset:]
        g = Gen(a.model, a.smoke_tokens, 4)
        grp, _ = g.samples([chat(g.tok, probs[0]["problem"])], 2, 0.6)
        n_mark = sum(marks_in(t) for t, _ in grp[0])
        import io as _io
        from contextlib import redirect_stdout as _rs
        buf = _io.StringIO()
        with _rs(buf):
            acc_of(grp[0], probs[0]["gold"], 2)
        n_warn = buf.getvalue().count("[경고]")
        log(f"스모크: 샘플 2개 × {a.smoke_tokens}토큰 · 쓰는 텍스트 마커 **{n_mark}** · "
            f"acc_of 경고 **{n_warn}** · vLLM o.text 마커 누적 {g.m_raw} · "
            f"두 텍스트가 다른 샘플 {g.n_diff}/{g.n_out}")
        log(f"스모크 샘플 앞 160자: {grp[0][0][0][:160]!r}")
        g.close()
        if n_mark or n_warn:
            raise SystemExit(f"!! 스모크 실패 (마커 {n_mark} · 경고 {n_warn}) — 본 생성을 하지 않는다")
        log("스모크 통과 — 본 생성으로 간다")

    log(f"모델 {a.model} · 문제 {a.n} (offset {a.offset}) · k={a.k} · cap {a.max_new} · source {a.source}")
    t0 = time.time()
    # do_traj=False: 궤적 npz 는 프로브에 안 쓰고(로컬 forward 로 뽑는다) OOM 을 유발했다
    screen = a.screen
    if screen:
        # **경로 변환 방어 (2026-10-10 에 당했다).** Git Bash 는 `/workspace/x.json` 같은 Unix 모양
        # 인자를 네이티브 exe 에 넘길 때 `C:/.../Git/workspace/x.json` 으로 바꿔 버린다(MSYS path
        # conversion). 그러면 glob 이 빗나가 **선별을 처음부터 다시 돌아** 절약이 날아간다.
        # 그래서 준 경로가 안 맞으면 **파일명만 떼어 작업 디렉터리에서 찾는다**.
        import glob as _g
        if not _g.glob(screen):
            alt = Path(screen).name
            if _g.glob(alt):
                log(f'선별 경로가 안 맞아 파일명으로 대체: {screen} -> {alt}')
                screen = alt
            else:
                log(f'!! 선별 파일을 못 찾았다: {screen} (파일명 {alt} 도 없음) → 새로 선별하게 된다')
        log(f'선별 재사용: {screen}')
    run(a.model, str(OUT), a.n, a.k, a.max_new, a.source, a.max_seqs,
        do_traj=False, screen_glob=screen, cap=0, pool_offset=a.offset)
    el = time.time() - t0
    log(f"완료 {el:.0f}s · 문제당 {el / max(1, a.n):.0f}s")
    files = sorted(p.name for p in OUT.glob("*"))
    log(f"산출 파일 {files}")


if __name__ == "__main__":
    main()
