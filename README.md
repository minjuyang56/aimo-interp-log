# AIMO Interpretability Challenge — negative results and evaluation methodology

Code and measurement logs for our NeurIPS 2026 AIMO Interpretability Challenge entry.
The technical report is `reports/aimo_interp_report.tex`.

**What this repository is.** A record of ~25 pre-registered hypotheses about predicting whether a
model's correct answer is *robust* or *spurious*, almost all of which failed, plus the checkpoints
that caught three ways of fooling ourselves. The headline is not a score: it is that at the sample
size this task provides (33 problems in the public label set), **no method we built — including the
problem-length rule — is distinguishable from chance by its own shuffle null.**

**Honesty conventions used throughout.** Every hypothesis was written into `CLAUDE.md` with its
primary metric and its sign *before* the measurement, one evaluation run per pre-registration, and
signs were never flipped after seeing results. Anything decided after seeing numbers is labelled
`EXPLORATORY` in both the code and the report. `CLAUDE.md` is the lab notebook and is the single
source of truth for what was fixed when; it is long and in Korean.

---

## 1. Environment

Measured on Windows 11, Python 3.12, one RTX 2000 Ada Laptop (8 GB VRAM).
GPU work that did not fit locally ran on Kaggle (T4 x2) or RunPod (H100 80 GB); those entry points
are under `experiments/kaggle/` and `experiments/runpod/`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -U pip
.venv/Scripts/python.exe -m pip install \
    "torch==2.12.1" --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install \
    transformers accelerate bitsandbytes numpy scipy scikit-learn pandas pyarrow datasets
```

Local experiments load the judged models in 4-bit NF4 (`--quant 4bit`, the default) because 8 GB
cannot hold a 7B model in bf16. `analysis/probe_precision_check.py` measures what that costs:
per-layer cosine 0.997--0.998 and a fold-out score rank correlation of 0.969 against bf16, i.e.
quantisation is not why the probes found nothing.

## 2. Data

* **Organisers' labels.** `aimo-interp/train-main-v2` (82 rows, 36 problems, 5 models) reduced to
  **33 problem-level majority labels** (20 robust / 13 spurious). Download once:
  ```bash
  .venv/Scripts/python.exe -c "import datasets,pathlib; \
    datasets.load_dataset('aimo-interp/train-main-v2',split='train') \
      .to_parquet('experiments/results/train-main-v2.parquet')"
  ```
  The 33-problem reduction lives in `analysis/h21_uncertainty.py::problems()`.
* **Our own labels.** Generated from MATH level 5 + AIME with answer-preserving perturbations; the
  generator is `experiments/kaggle/genlab_v2.py` (same file on Kaggle and RunPod, deliberately, so
  the two label sources stay comparable). Aggregated label JSONs for the final `k=8` sets are
  committed under `experiments/results/runpod/labelgen32k/*/out/`.
* **Not redistributed.** Model weights, and the raw generation texts (85 MB) which are archived
  outside the repository. Every number in the report comes from a committed `.txt`/`.log` under
  `experiments/results/`.

## 3. Reproducing the report's numbers

Each line is standalone and writes its own report file. Runtime is on the local 8 GB machine.

| Report item | Command | Output | Cost |
|---|---|---|---|
| Table 1 — same features, two label sources | `analysis/probe_gen_eval.py --win 2048` | `probe_gen_report*.txt` | $0, minutes |
| | `analysis/org_sub2048_transfer.py` | `org_sub2048_transfer.txt` | $0 |
| | `analysis/org_train_eval.py` | `org_train_report.txt` | $0, ~20 min |
| Table 2 — four classifiers | `analysis/cmp4_eval.py` | `cmp4_report.txt` | $0, ~20 min |
| Lesson (a) length control | `analysis/length_confound.py`, `analysis/h23_length_proxy.py` | `h23_length_proxy.txt` | $0 |
| Lesson (b) label instability | `analysis/temp_sens_eval.py`, `analysis/label_cross_model.py` | `temp_sens_report.txt` | $0 |
| Lesson (c) multivariate misread | `analysis/entlens_eval.py --perms 1000`, `analysis/flipent_eval.py` | `entlens_report.txt`, `flipent_report.txt` | $0 |
| Power / resolution ceiling | `analysis/power_33.py`, `analysis/probe_subset_shift.py`, `analysis/label_audit.py` | `probe_subset_shift.txt`, `label_audit.txt` | $0 |
| H-FIN natural termination | `analysis/hfin_eval.py` then `analysis/hfin_hybrid_eval.py` | `hfin_report.txt`, `hfin_hybrid_report.txt` | $0 |
| Leaderboard facts (aggregate only) | `analysis/codabench_ids.py` | `codabench_ids.txt` | $0 |

All of the above read committed artefacts and need **no GPU**, except the feature extractions they
depend on. To regenerate features from scratch:

```bash
# problem-text features (Table "before solving")
.venv/Scripts/python.exe experiments/probe_pilot.py  --model Skywork/Skywork-OR1-Math-7B
# mid-generation features, window 256/512/2048 in one pass
.venv/Scripts/python.exe experiments/probe_gen.py    --model Skywork/Skywork-OR1-Math-7B
```

Shuffle nulls are 1000 permutations with `numpy.random.default_rng(0)`; evaluators that parallelise
them pre-generate the identical permutation sequence and ship a `--verify-parallel` check that the
serial and parallel null arrays match to 0.0e+00.

## 4. Submissions

```bash
.venv/Scripts/python.exe scripts/package.py hfin --small-only   # -> dist/hfin-small.zip
.venv/Scripts/python.exe analysis/smoke_submission.py dist/hfin-small.zip
.venv/Scripts/python.exe analysis/hfin_smoke.py                 # 17 checks, no GPU needed
```

`solutions/<name>/solution.py` implements the scorer contract
`are_robust(model_id, reasoning_effort="default", problems=None) -> list[bool]`.
Three contract rules cost us whole batches before we wrote them down, and every submission here
obeys them: return **plain Python `bool`** (the scorer tests `type(x) is bool`), never raise on an
unknown `model_id`, and load each model **once per call**. A Small-track zip must contain
`small.txt` at its root; a Main-track zip must not.

Submissions and the leaderboard scores we observed are listed in the report. Where a submission is
exploratory or post-hoc, its own module docstring says so.

## 5. Repository layout

```
analysis/        standalone evaluation scripts; each writes experiments/results/<name>.txt
experiments/     feature extraction, label generation, Kaggle and RunPod entry points
solutions/       submission implementations (one directory per submitted method)
scripts/         packaging
experiments/results/   committed measurement logs -- the provenance for every number
reports/         technical report (LaTeX)
report/          earlier English prose draft and the figure script
docs/            task notes (Korean): what information is available, and the column semantics
CLAUDE.md        lab notebook: pre-registrations, corrections, retractions (Korean)
JOURNAL.html     narrative work log, chapter 1 (Korean; .en.html is the English version)
```

Two task notes are worth reading before the code, both Korean:

* `docs/AVAILABLE.md` — everything a method may legitimately see at the moment `are_robust()` is
  called, split into four layers: the two function arguments, what can be computed inside the
  container, what can be shipped inside the ZIP, and what may only be used offline for training.
* `docs/DATA.md` — the 15 columns of the original dataset, and which of them are *inputs* versus
  *targets only*. The one-line version: in the real call you receive `model_id` and the problem
  text, and nothing else; the other 13 columns are the record of how the label was produced.
  Missing that distinction is how label leakage happens — `permutation_type` separated 28/28 of the
  old validation set before the organisers removed it.

## 6. Release note

This is a **snapshot**. The development repository carries 268 commits whose messages hold the
pre-registration record, but it also tracks 3.5 GB of `.npz` feature caches including six files
above GitHub's 100 MB hard limit, so it cannot be pushed as-is. The snapshot was produced by
`analysis/build_release.py`, which drops every `.npz` and anything over 1 MB and keeps all code, all
measurement logs, the report PDF and every submission ZIP under 1 MB (675 files, 13.2 MB).

Two gates run before publication, and both report **0 hits** for known key formats, 0 credential
files and 0 email addresses: `analysis/release_scan.py` over the development repository's tracked
files, and `analysis/release_scan_tree.py` over this snapshot itself — the second one also reads
*inside* all 49 submission ZIPs, so what was scanned is exactly what was published. Neither prints a
matched value; they report rule name, file and line only.

This tree is **anonymized for review**: author name, GitHub/Kaggle/Codabench handles and the local
OS user name are replaced by `[author-*]` / `[user]` tokens at build time (226 substitutions; the
development repository keeps the originals as its own record). Because a ZIP is copied as bytes and
the filter cannot reach inside it, `build_release.py` scans every submission ZIP's members and
**drops any ZIP that still contains an identity string** rather than repacking it — repacking would
change the sha256 that records which bytes were actually submitted. One ZIP
(`dist/flipent-small.zip`) is excluded for that reason; its source is present and anonymized under
`solutions/flipent/`.

Commit hashes cited in the report refer to the private development history and do not resolve here.
`PROVENANCE.md` maps each cited hash to its full hash, date and commit subject so the citations stay
checkable; the logs those numbers came from are all in `experiments/results/`.

## 7. Citation and related work

The task, the label definition and the two baselines we compare against are from the organisers'
paper, arXiv:2607.13899. Two public competitor repositories informed our reading of the task and
are cited in the report: `byeolki/aimo-interp` (the observation that a prompt-only probe reads
solve rate rather than robustness) and `EconLearn/aimo-interp-difficulty-vs-fragility` (a
non-probing natural-termination rule). We did not copy code from either.

## License

MIT, see `LICENSE`. Third-party data and model licences are listed there.
