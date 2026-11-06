"""Kaggle 커널 헤더 (H10-B) — kaggle_h10b.py 가 뒤에 h10b_trajectory.py 소스를 이어 붙여 kernel.py 로 올린다.
모델 하나당 커널 하나. RUN_MODELS 는 kaggle_h10b.py 가 치환한다. 변형 없음·라벨 생성 없음(벡터 추출만).
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers==5.13.0", "tokenizers", "accelerate", "safetensors",
                "bitsandbytes", "pandas", "pyarrow"], check=True)

RUN_MODELS = ["__MODEL__"]
RUN_BATCH = __BATCH__
RUN_RESUME = __RESUME__      # 예: "/kaggle/input/**/Qwen3_5-4B_v2.npz" (kernel_sources 로 붙인 이전 커널 출력)
sys.argv = ["h10b_trajectory.py", "--out", "/kaggle/working/h10b"]
