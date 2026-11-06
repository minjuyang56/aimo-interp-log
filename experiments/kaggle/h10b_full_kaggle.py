"""Kaggle 커널 헤더 (H10-B 전체 궤적판, 지휘 #26-4 — 준비만, 실행은 지휘 결정 후).
kaggle_h10b.py --header h10b_full_kaggle.py --suffix=-full 로 올린다. 모델 하나당 커널 하나.
1024토큰판과 차이: MAX_NEW 8192 · CHUNK 128 (청크 수 최대 64). EOS 도달 시 조기 종료되므로 청크 수가 문제마다 다름 → 평가 때 n_chunks 잔차화 필수.
시간 추정: T4×2 fp16 배치 4에서 1024토큰 36문제 ≈ 30–60분 → 8192토큰은 최대 8배(EOS로 실제는 적음). 12시간 커널 한도 안이면 OK, 아니면 --batch 8.
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers==5.13.0", "tokenizers", "accelerate", "safetensors",
                "bitsandbytes", "pandas", "pyarrow"], check=True)

RUN_MODELS = ["__MODEL__"]
RUN_BATCH = __BATCH__
RUN_RESUME = __RESUME__
RUN_MAX_NEW = 8192
RUN_CHUNK = 128
sys.argv = ["h10b_trajectory.py", "--out", "/kaggle/working/h10b_full"]
