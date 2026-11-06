"""Kaggle 커널 헤더 (#LIT-PLAN P2·P3 엔트로피 궤적 + logit lens).

**vLLM 을 쓰지 않는다** — 토큰별 전 어휘 엔트로피와 층별 hidden 이 필요하고 그건 HF 경로만 준다.
그래도 torchaudio 제거는 유지한다: Kaggle 이미지의 torch(cu13.0)/torchaudio(cu12.8) 불일치로
`transformers.audio_utils` 가 RuntimeError 를 내는 경로가 transformers 안에도 있다(2026-10-04 사고).
"""

import os
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers",
                "accelerate"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
import importlib.util                 # import 하면 그 RuntimeError 가 바로 터지므로 find_spec 으로만
if importlib.util.find_spec("torchaudio") is None:
    print("[env] torchaudio 제거 확인", flush=True)
else:
    print("[env] torchaudio 가 남아 있다 → sys.modules 차단", flush=True)
    sys.modules["torchaudio"] = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["ENTLENS_MODEL"] = "__MODEL__"
os.environ["ENTLENS_MAXNEW"] = "__MAXNEW__"
os.environ["ENTLENS_BATCH"] = "__BATCH__"
os.environ["ENTLENS_NPROB"] = "__NPROB__"
os.environ["ENTLENS_QUANT"] = "fp16"          # T4 는 Turing — bf16 을 못 쓴다
os.environ["ENTLENS_LOCAL"] = "0"             # Kaggle 은 HF 에서 내려받는다
os.environ["ENTLENS_OUT"] = "/kaggle/working/entlens"
print(f"[env] model {os.environ['ENTLENS_MODEL']} · maxnew {os.environ['ENTLENS_MAXNEW']} · "
      f"batch {os.environ['ENTLENS_BATCH']} · nprob {os.environ['ENTLENS_NPROB']}", flush=True)
