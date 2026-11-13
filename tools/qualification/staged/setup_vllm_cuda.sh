#!/bin/bash
# Prepare the pinned vLLM 0.30.0+cu129 environment, model snapshot and Metria
# checkout on a fresh GPU host. Mirrors artifacts/qualification/vllm-0.30.0/cuda.
set -euo pipefail
work=${1:?usage: setup_vllm_cuda.sh NEW_WORK_DIR}
mkdir -p "$work"
cd "$work"
log() { printf '\n== %s\n' "$*"; }

log host
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
df -h / | tail -1
python3 --version

log uv
if ! command -v uv >/dev/null; then
  python3 -m pip install --quiet --disable-pip-version-check uv
fi
uv --version

log metria
if [ ! -d metria-src/.git ]; then
  git clone --quiet https://github.com/dipeshbabu/metria.git metria-src
fi
git -C metria-src fetch --quiet origin main
git -C metria-src checkout --quiet --detach origin/main
git -C metria-src rev-parse HEAD

log wheel
wheel="vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
want=e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc
mkdir -p wheels
if [ ! -f "wheels/$wheel" ]; then
  curl -fsSL --proto '=https' -o "wheels/$wheel.part" \
    "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
  mv "wheels/$wheel.part" "wheels/$wheel"
fi
echo "$want  wheels/$wheel" | sha256sum -c -

log venv
# Some GPU images set UV_NO_CACHE=1 and UV_LINK_MODE=copy, which stores every
# package twice and exhausts small disks. Keep one copy in the cache.
unset UV_NO_CACHE
export UV_CACHE_DIR="$work/uv-cache" UV_LINK_MODE=symlink WORK="$work"
[ -x venv/bin/python ] || uv venv --quiet --python 3.12 venv
printf 'setuptools==84.0.0\n' > overrides.txt
uv pip install --quiet --python venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match --override overrides.txt \
  "wheels/$wheel" "torch==2.13.0+cu129" "setuptools==84.0.0"
du -sh "$UV_CACHE_DIR" venv
venv/bin/python - <<'EOF'
import importlib.metadata as m
for name in ("vllm", "torch", "setuptools", "transformers", "triton", "flashinfer-python"):
    try:
        print(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError:
        print(f"{name} not installed")
import torch
print("cuda available:", torch.cuda.is_available(), torch.version.cuda,
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF

log model
venv/bin/python - <<'EOF'
import json, os
from huggingface_hub import snapshot_download
d = json.load(open("metria-src/tools/qualification/vllm-smollm2-135m.json"))
path = snapshot_download(d["model_id"], revision=d["revision"],
                         allow_patterns=sorted(d["files"]), cache_dir=os.environ["WORK"] + "/hf")
open(os.environ["WORK"] + "/model_root", "w").write(path)
print(path)
EOF
df -h / | tail -1
echo SETUP_OK
