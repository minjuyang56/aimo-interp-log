"""Kaggle 커널 헤더 (H14 길이비, 지휘 #43)."""

import os
import subprocess
import sys

# 2026-10-04: Kaggle 이미지가 바뀌어 torch(cu13.0)와 torchaudio(cu12.8)의 CUDA 버전이 어긋났다.
# transformers.audio_utils 가 torchaudio 를 import 하는 순간 RuntimeError 로 죽어서 vllm import 자체가 불가능하다
# (커널 3개가 전부 2분 안에 ERROR). 오디오는 안 쓰므로 **지우고** 간다 — transformers 는 없으면 건너뛴다.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)   # vllm 이 다시 끌어올 수 있다
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"], check=False)
import importlib.util                  # 지워졌는지 **import 하지 않고** 확인한다
                                       # (import 하면 그 CUDA RuntimeError 가 바로 터진다)
if importlib.util.find_spec("torchaudio") is None:
    print("[env] torchaudio 제거 확인", flush=True)
else:
    print("[env] torchaudio 가 아직 있다 → sys.modules 차단으로 ImportError 로 바꿔 둔다", flush=True)
    sys.modules["torchaudio"] = None   # RuntimeError 대신 ImportError → transformers 의 가드가 잡는다

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

RUN_MODEL = "__MODEL__"
RUN_K = __K__
RUN_MAX_NEW = __MAXNEW__
RUN_MAX_SEQS = __MAXSEQS__
RUN_KINDS = "__KINDS__"
sys.argv = ["genlab_h14.py"]
