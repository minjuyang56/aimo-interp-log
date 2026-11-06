"""Kaggle 커널 헤더 (vLLM 라벨 생성, 지휘 #30).

vLLM 은 자기 버전에 맞는 torch/transformers 를 끌고 온다. T4(7.5)는 fp16만 되므로 dtype=half.
tensor_parallel_size 는 스크립트가 보이는 GPU 수로 정한다. 멀티프로세스 백엔드는 spawn 고정.
vLLM 설치·초기화가 실패하면 그 모델은 건너뛰고 로그에 이유를 남긴다(옛 HF 커널 genlab.py 로 폴백).
"""

import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"], check=False)

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

RUN_MODELS = ["__MODEL__"]
RUN_N = __N__
RUN_MAX_NEW = __MAXNEW__
RUN_MAX_SEQS = __MAXSEQS__
RUN_N_OFF = __NOFF__
RUN_N_UNOFF = __NUNOFF__
RUN_LEVELS = "__LEVELS__"
RUN_TAG = "__TAG__"
sys.argv = ["genlab_vllm.py", "--out", "/kaggle/working/genlab"]
