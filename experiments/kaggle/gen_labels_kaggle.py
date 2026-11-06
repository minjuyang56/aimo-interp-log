"""Kaggle 커널 헤더 — kaggle_run.py 가 이 파일 뒤에 gen_labels.py 소스를 이어 붙여 kernel.py 로 올린다.

Kaggle script 커널은 CLI 인자를 못 받고 파일 하나만 올라가므로, 여기서 의존성 설치와 sys.argv 를 고정한다.
첫 실행: 라벨 있는 문제 robust 5 + spurious 5 · 변형 4종(정답 불변) × k=4 · Qwen3.5-4B · 8192토큰 · 배치 8.
T4는 bf16 미지원 → load_model 이 fp16으로 자동 전환한다.
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers==5.13.0", "tokenizers", "accelerate", "safetensors",
                "scipy", "pandas", "pyarrow", "scikit-learn"], check=True)   # Kaggle 기본 transformers는 qwen3_5 모름

sys.argv = ["gen_labels.py", "--out", "/kaggle/working/gen", "--balanced", "5", "--variants", "rename,typos,distract,reorder",
            "--k", "4", "--max-new-tokens", "8192", "--batch", "8", "--quant", "none"]
