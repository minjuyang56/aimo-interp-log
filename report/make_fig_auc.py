"""report/fig_auc.png — 사전등록 주지표의 AUC 점 + 라벨 섞기 null 95% 구간 (forest plot). (지휘 #52-4)

표에 들어가는 값은 **전부 experiments/results/ 의 커밋된 로그에서 옮긴 것**이다. 아래 SOURCE 열에 파일명을 적어 둔다.
주지표만 넣는다(탐색 지표 제외). 모델별 점수는 문제 단위로 평균한 뒤 길이 잔차화한 '풀링(잔차)' 값을 쓴다.

사용:  .venv/Scripts/python.exe report/make_fig_auc.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (라벨, AUC, null_lo, null_hi, 출처 파일)
ROWS = [
    ("H3  length only (no training)", 0.331, 0.34, 0.65, "h3_pooled_report.txt"),
    ("H4  MATH-trained probe, layer 18", 0.730, 0.36, 0.75, "h4_external_report.txt"),
    ("H5  per-token NLL, pooled", 0.688, 0.30, 0.71, "h5_report.txt"),
    ("H5r relative familiarity", 0.530, 0.23, 0.76, "h5_relative_report.txt"),
    ("H6  internal shift under perturbation", 0.373, 0.34, 0.69, "h6_report.txt"),
    ("H8  gradient attribution to surface", 0.442, 0.32, 0.68, "h8_report_v2.txt"),
    ("H9  erasure importance (structure share)", 0.596, 0.32, 0.65, "h9_report_v2.txt"),
    ("H10A prompt-state compass", 0.435, 0.20, 0.68, "h10_report_h10_v2.txt"),
    ("H10B solve-trajectory compass", 0.615, 0.28, 0.70, "h10b_report_v2.txt"),
    ("H10B' compass, erasure weights", 0.542, 0.34, 0.66, "h10_report_h9_v2.txt"),
    ("H11b surface knockout, all queries", 0.403, 0.31, 0.69, "h11b_report_v2.txt"),
    ("H12 activation patching of surface", 0.480, 0.31, 0.65, "h12_report_v2.txt"),
    ("H13 surface subspace energy", 0.512, 0.37, 0.64, "h13_report.txt"),
    ("H14 rename length ratio", 0.400, 0.29, 0.70, "h14_report.txt"),
    ("H14b short-horizon logprob gap", 0.427, 0.30, 0.71, "h14b_report.txt"),
    ("H15 rename-able surface count", 0.760, 0.30, 0.70, "h15_report.txt"),
    ("H16 prefill pooled probe", 0.531, 0.31, 0.72, "h16_report.txt"),
    ("H16b prefill probe + 29 synthetic", 0.556, 0.34, 0.66, "h16b_report.txt"),
]


def main():
    rows = list(reversed(ROWS))
    ys = range(len(rows))
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for i, (lab, auc, lo, hi, _) in zip(ys, rows):
        inside = lo <= auc <= hi
        ax.plot([lo, hi], [i, i], color="0.72", lw=5, solid_capstyle="butt", zorder=1)
        ax.plot([auc], [i], marker="o", ms=7, mfc=("white" if inside else "#b2182b"),
                mec=("0.35" if inside else "#b2182b"), mew=1.6, zorder=3)
    ax.axvline(0.5, color="0.45", ls=":", lw=1, zorder=0)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0.15, 0.90)
    ax.set_xlabel("AUC (pre-registered sign; pooled over models, residualised on length)", fontsize=9)
    ax.set_title("Sixteen pre-registered hypotheses against their label-permutation nulls\n"
                 "grey bar = null 95% interval · filled marker = outside the null", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fig_auc.png"
    fig.savefig(out, dpi=200)
    print(f"저장 {out}")
    n_out = sum(1 for _, a, lo, hi, _ in ROWS if not (lo <= a <= hi))
    print(f"항목 {len(ROWS)}개 · null 밖 {n_out}개: "
          f"{[r[0].split()[0] for r in ROWS if not (r[2] <= r[1] <= r[3])]}")


if __name__ == "__main__":
    main()
