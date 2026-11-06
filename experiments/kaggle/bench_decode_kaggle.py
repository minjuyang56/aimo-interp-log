"""Kaggle 커널 헤더 (디코딩 속도 벤치, 지휘 #57b)."""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "pandas", "pyarrow"], check=False)

RUN_MODEL = "__MODEL__"
RUN_SEQS = __SEQS__
RUN_MAX_NEW = __MAXNEW__
sys.argv = ["bench_decode.py"]
