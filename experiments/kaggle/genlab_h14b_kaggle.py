"""Kaggle 커널 헤더 (H14b 대리지표, 지휘 #44)."""

import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"], check=False)

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

RUN_MODEL = "__MODEL__"
RUN_MAX_NEW = __MAXNEW__
RUN_MAX_SEQS = __MAXSEQS__
sys.argv = ["genlab_h14b.py"]
