"""민감도 프로파일 — 문제의 요소를 하나씩 바꿔 내부 표현이 얼마나 움직이는지 잰다.

가설: 잘 읽은 모델은 답에 영향 있는 요소(숫자)엔 민감하고, 없는 요소(이름·오타·군더더기)엔
둔감하다. 그 차이가 robust 라벨과 상관이 있는가.

편집은 전부 규칙 기반 — 모델 없이 만든다. 그래서 런타임에도 쓸 수 있다.
    number   숫자 하나를 n+1로                    ← 답에 영향 있음 (관련)
    typos    수식 밖 단어 3개에 인접 글자 교환       ← 없음 (무관)
    swap     이야기 명사·이름을 사전으로 교체        ← 없음 (무관, 조잡한 domain 변형)
    filler   앞에 무해한 문장 한 줄                  ← 없음 (무관)

측정: 원본과 변형본을 각각 forward → 모든 층의 마지막 토큰 벡터 → 층별 코사인 거리.
저장: results/sensitivity_<모델>.npz   D (문제, 편집, 층) · y · pid
끝나면 층별 AUC 표를 찍는다. 문제마다 저장하므로 죽어도 --resume 으로 이어간다.

사용:  uv run python experiments/sensitivity.py [--limit N] [--resume]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis"))
sys.path.insert(0, str(HERE))
from _paths import BASELINES, require  # noqa: E402
from extract_hidden import DEFAULT_MODEL, alias, load_rows, load_tokenizer  # noqa: E402

SOLVE_PROMPT = (BASELINES / "prompts" / "solve.txt")
EDITS = ["number", "typos", "swap", "filler"]
RELEVANT = {"number"}

SWAP = {
    "Alice": "Carol", "Bob": "Dave", "John": "Peter", "Mary": "Susan", "Tom": "Jake", "Sam": "Leo",
    "baker": "farmer", "bakers": "farmers", "cookies": "apples", "cookie": "apple", "cake": "pie",
    "students": "workers", "student": "worker", "teacher": "manager", "class": "team", "school": "office",
    "cars": "trains", "car": "train", "bus": "boat", "train": "truck",
    "dollars": "euros", "dollar": "euro", "coins": "tokens", "coin": "token",
    "airline": "shipping", "companies": "firms", "company": "firm", "flights": "voyages", "flight": "voyage",
    "island": "harbor", "days": "weeks", "day": "week", "garden": "field", "park": "yard", "river": "canal",
    "marbles": "beads", "balls": "cubes", "ball": "cube", "boxes": "bags", "box": "bag",
    "apples": "pears", "oranges": "plums", "candies": "stickers", "books": "pens", "book": "pen",
    "people": "guests", "friends": "cousins", "children": "players", "politicians": "lawyers", "lawyers": "doctors",
}
MATH_SPAN = re.compile(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
NUMBER = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def edit_number(text, rng):
    """수식 안팎 가리지 않고 숫자 하나를 n+1로. 숫자가 없으면 원문 그대로(변화 0)."""
    nums = [m for m in NUMBER.finditer(text) if int(m.group(1)) >= 2]
    if not nums:
        return text
    m = nums[rng.integers(len(nums))]
    return text[:m.start(1)] + str(int(m.group(1)) + 1) + text[m.end(1):]


def edit_typos(text, rng, k=3):
    spans = [(m.start(), m.end()) for m in MATH_SPAN.finditer(text)]
    def in_math(i): return any(a <= i < b for a, b in spans)
    words = [m for m in re.finditer(r"[A-Za-z]{5,}", text) if not in_math(m.start())]
    if not words:
        return text
    picks = rng.choice(len(words), size=min(k, len(words)), replace=False)
    out = list(text)
    for i in sorted(picks, reverse=True):
        m = words[i]; j = m.start() + 1 + int(rng.integers(m.end() - m.start() - 3))
        out[j], out[j + 1] = out[j + 1], out[j]
    return "".join(out)


def edit_swap(text, rng):
    def rep(m):
        w = m.group(0)
        return SWAP.get(w, SWAP.get(w.lower(), w).capitalize() if w[0].isupper() and w.lower() in SWAP else w)
    return re.sub(r"\b[A-Za-z]+\b", rep, text)


def edit_filler(text, rng):
    return "Here is a problem from a practice set that I would like you to work through.\n\n" + text


EDIT_FN = {"number": edit_number, "typos": edit_typos, "swap": edit_swap, "filler": edit_filler}


def hidden_all_layers(tok, model, prompt, problem):
    import torch
    text = tok.apply_chat_template([{"role": "user", "content": f"{prompt}\n\n{problem}"}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    return np.stack([h[0, -1].float().cpu().numpy() for h in hs])      # (L, D)


def cosine_dist(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return 1.0 - (A * B).sum(1)                                          # (L,)


def report(D, y, pids, edits):
    from sklearn.metrics import roc_auc_score
    L = D.shape[2]
    y = y.astype(int)
    rel = [i for i, e in enumerate(edits) if e in RELEVANT]
    irr = [i for i, e in enumerate(edits) if e not in RELEVANT]
    print(f"\n문제 {len(y)} · robust {y.sum()} · 층 {L}")
    print("편집별 평균 표현변화 (전 층 평균):")
    for i, e in enumerate(edits):
        print(f"  {e:7s} {D[:, i, :].mean():.4f}   robust {D[y == 1, i, :].mean():.4f}  spurious {D[y == 0, i, :].mean():.4f}")
    print(f"\n{'층':>3}  {'AUC rel':>8} {'AUC irr':>8} {'AUC rel-irr':>12}")
    best = (0.5, None)   # 0.5 = 무정보에서 시작. 0에서 시작하면 |0-0.5|가 항상 이겨서 갱신이 안 된다
    for l in range(L):
        s_rel = D[:, rel, l].mean(1); s_irr = D[:, irr, l].mean(1)
        a1 = roc_auc_score(y, s_rel); a2 = roc_auc_score(y, s_irr); a3 = roc_auc_score(y, s_rel - s_irr)
        flag = " *" if abs(a3 - 0.5) > 0.1 else ""
        print(f"{l:3d}  {a1:8.3f} {a2:8.3f} {a3:12.3f}{flag}")
        if abs(a3 - 0.5) > abs(best[0] - 0.5): best = (a3, l)
    print(f"\nrel−irr 최고: 층 {best[1]}  AUC {best[0]:.3f}   (0.5 = 무정보, 0.6/0.4 넘으면 신호)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    require(SOLVE_PROMPT)
    out = HERE / "results" / f"sensitivity_{alias(args.model)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.limit)
    done = {}
    if (args.resume or args.report_only) and out.is_file():
        d = np.load(out, allow_pickle=True)
        done = {p: (dd, yy) for p, dd, yy in zip(d["pid"], d["D"], d["y"])}
        print(f"이미 {len(done)}문제 완료", flush=True)
    if args.report_only:
        report(np.stack([v[0] for v in done.values()]), np.array([v[1] for v in done.values()]),
               list(done), EDITS)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompt = SOLVE_PROMPT.read_text(encoding="utf-8").strip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True,
                                                 low_cpu_mem_usage=True).to(device).eval()
    print(f"model={args.model} device={device} load {time.time() - t0:.0f}s 문제 {len(rows)}개 편집 {EDITS}", flush=True)

    rng = np.random.default_rng(0)
    for i, r in enumerate(rows, 1):
        if r["problem_id"] in done:
            continue
        t1 = time.time()
        base = hidden_all_layers(tok, model, prompt, r["original_problem"])
        dists = []
        for e in EDITS:
            v = EDIT_FN[e](r["original_problem"], rng)
            dists.append(np.zeros(base.shape[0], np.float32) if v == r["original_problem"]
                         else cosine_dist(base, hidden_all_layers(tok, model, prompt, v)).astype(np.float32))
        D_i = np.stack(dists)                                          # (편집, 층)
        done[r["problem_id"]] = (D_i, r["model_is_robust"].strip().lower() == "true")
        np.savez_compressed(out, D=np.stack([v[0] for v in done.values()]),
                            y=np.array([v[1] for v in done.values()]), pid=np.array(list(done)),
                            edits=np.array(EDITS), model=args.model)
        mid = base.shape[0] // 2
        print(f"  [{i:3d}/{len(rows)}] {r['problem_id']}  {time.time() - t1:5.0f}s  "
              + "  ".join(f"{e}={D_i[j, mid]:.3f}" for j, e in enumerate(EDITS))
              + f"  robust={done[r['problem_id']][1]}", flush=True)

    report(np.stack([v[0] for v in done.values()]), np.array([v[1] for v in done.values()]), list(done), EDITS)
    print(f"\n총 {time.time() - t0:.0f}s  저장 {out}")


if __name__ == "__main__":
    main()
