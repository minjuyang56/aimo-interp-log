"""Kaggle 커널 헤더 (자체 라벨 생성, 지휘 #28). kaggle_h10b.py --header genlab_kaggle.py --body variants_nonofficial.py,genlab.py 로 올린다.
파일럿: RUN_N=10 · RUN_MAX_NEW=4096 · RUN_BATCH=8 · RUN_HOURS 로 안전 상한.
"""

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers==5.13.0", "tokenizers", "accelerate", "safetensors",
                "bitsandbytes", "pandas", "pyarrow"], check=True)

RUN_MODELS = ["__MODEL__"]
RUN_BATCH = __BATCH__
RUN_RESUME = __RESUME__
RUN_N = __N__
RUN_MAX_NEW = __MAXNEW__
RUN_HOURS = __HOURS__
sys.argv = ["genlab.py", "--out", "/kaggle/working/genlab"]
