# Policy training environment

SFT/RL uses a heavier environment than offline rubric checks or dense retrieval.
The existing research environment reports Python 3.12.9, PyTorch 2.10.0+cu128,
vLLM 0.17.0, flash-attn 2.8.3, flash-linear-attention 0.4.2, causal-conv1d
1.6.1, tensordict 0.12.2, and transformers 5.6.0.dev0 from the commit below.
The release's integration tests use that existing environment. Installation
from scratch and GPU training were not rerun for this source release.

The following ordered recipe is carried forward from the research setup.
`requirements-train.txt` is a reference manifest, not a one-shot installation
command: compiled kernels and the vLLM/transformers dependency order matter.

```bash
conda create -n hisearch python=3.12 -y
conda activate hisearch
conda install -c conda-forge cuda-toolkit=12.8 \
  gxx_linux-64=12 gcc_linux-64=12 ripgrep -y
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install psutil packaging ninja wheel setuptools
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 causal-conv1d==1.6.1 --no-build-isolation
python -m pip install flash-linear-attention==0.4.2

# Let vLLM establish its dependency tree before the pinned transformers build.
python -m pip install vllm==0.17.0
python -m pip install tensordict==0.12.2 ray==2.54.0 accelerate deepspeed peft \
  datasets hydra-core omegaconf codetiming dill einops pyarrow pandas wandb \
  sentencepiece tensorboard pylatexenc latex2sympy2_extended math_verify \
  torchdata flashinfer-python httpx
python -m pip install --no-deps \
  huggingface_hub==1.8.0 tokenizers==0.22.2 safetensors==0.7.0
python -m pip install --no-deps \
  'transformers @ git+https://github.com/huggingface/transformers.git@f048e845684894fe60440bb8506f26ffaf7b69ac'
python -m pip install -r sft/data_generation/requirements.txt

export PYTHONPATH="$PWD:$PWD/rl:$PWD/verl:${PYTHONPATH:-}"
```

Activate this environment before using the launchers. If a local `.venv/bin/activate`
exists, the root GSPO launchers use it; otherwise they use the active environment.
The vendored runtime is loaded through `PYTHONPATH`, so installing a separate
upstream verl package is unnecessary. Its packaging metadata predates this
particular stack; do not install its optional dependency extras over the recipe.

The offline workflow only needs `requirements-offline.txt`. Dense retrieval uses
the separate environment in [retrieval/README.md](retrieval/README.md), whose
PyTorch/FAISS versions differ from policy training.

Compiled kernels require a compatible CUDA toolkit/compiler and sufficient
host RAM. Limit build parallelism for your machine. For A100/A800 builds,
`FLASH_ATTN_CUDA_ARCHS=80` can restrict compilation to the target architecture.

The RL launcher deliberately avoids SFT's `expandable_segments` CUDA allocator
setting because vLLM's memory pool can reject it. When moving from SFT to RL,
use a fresh shell or unset `PYTORCH_CUDA_ALLOC_CONF` if it was set manually.
