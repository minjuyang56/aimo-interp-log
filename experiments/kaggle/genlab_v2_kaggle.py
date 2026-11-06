"""Kaggle 커널 헤더 (genlab v2, 지휘 #33)."""

import os
import subprocess
import sys

# 2026-10-04: Kaggle 이미지의 torch(cu13.0) / torchaudio(cu12.8) CUDA 불일치로 transformers.audio_utils 가
# RuntimeError 를 내고, 그 경로가 vllm import 안에 있다(커널 3개가 2분 안에 ERROR). 오디오는 안 쓰므로 지운다.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"], check=False)
import importlib.util                 # import 하면 그 RuntimeError 가 바로 터지므로 find_spec 으로만 확인
if importlib.util.find_spec("torchaudio") is None:
    print("[env] torchaudio 제거 확인", flush=True)
else:
    print("[env] torchaudio 가 남아 있다 → sys.modules 차단", flush=True)
    sys.modules["torchaudio"] = None

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

RUN_MODELS = ["__MODEL__"]
RUN_N = __N__
RUN_K = __K__
RUN_MAX_NEW = __MAXNEW__
RUN_SOURCE = "__SOURCE__"
RUN_MAX_SEQS = __MAXSEQS__
RUN_SCREEN = __SCREEN__
RUN_CAP = __CAP__
RUN_OFFSET = __OFFSET__
# 궤적(greedy 1회 + hidden) 은 쓰지 않는다 — 프로브 특징은 로컬 forward 로 뽑고,
# 파일럿2 에서 이 단계가 OOM 10건을 냈다. 라벨 생성에 불필요하므로 끈다.
sys.argv = ["genlab_v2.py", "--out", "/kaggle/working/genlab", "--no-traj"]
