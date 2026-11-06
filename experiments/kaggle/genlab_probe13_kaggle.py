"""Kaggle 커널 헤더 (판별 실험, 지휘 #40-1)."""

import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

RUN_MODEL = "__MODEL__"
RUN_LABELS = "__LABELS__"
RUN_K = __K__
RUN_MAX_NEW = __MAXNEW__
RUN_MAX_SEQS = __MAXSEQS__
sys.argv = ["genlab_probe13.py"]
