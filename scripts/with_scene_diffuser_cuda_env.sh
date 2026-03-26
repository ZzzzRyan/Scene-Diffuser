#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${CONDA_PREFIX:-${SCENE_DIFFUSER_PREFIX:-$ROOT_DIR/.conda/envs/scene-diffuser}}"

if [[ -n "${SCENE_DIFFUSER_PYTHON_SITE:-}" ]]; then
  PYTHON_SITE="$SCENE_DIFFUSER_PYTHON_SITE"
else
  shopt -s nullglob
  _python_site_candidates=("$PREFIX"/lib/python*/site-packages)
  shopt -u nullglob
  if [[ ${#_python_site_candidates[@]} -eq 0 ]]; then
    echo "Python site-packages not found under: $PREFIX" >&2
    exit 1
  fi
  PYTHON_SITE="${_python_site_candidates[0]}"
fi

NVIDIA_SITE="$PYTHON_SITE/nvidia"
TORCH_LIB_DIR="$PYTHON_SITE/torch/lib"

if [[ ! -d "$PREFIX" ]]; then
  echo "Conda env not found: $PREFIX" >&2
  exit 1
fi

_join_by_colon() {
  local result=""
  local item=""
  for item in "$@"; do
    if [[ -z "$item" || ! -e "$item" ]]; then
      continue
    fi
    if [[ -z "$result" ]]; then
      result="$item"
    else
      result="$result:$item"
    fi
  done
  printf '%s' "$result"
}

INCLUDE_PATHS=(
  "$PREFIX/targets/x86_64-linux/include"
  "$NVIDIA_SITE/cuda_runtime/include"
  "$NVIDIA_SITE/cublas/include"
  "$NVIDIA_SITE/cudnn/include"
  "$NVIDIA_SITE/cufft/include"
  "$NVIDIA_SITE/cufile/include"
  "$NVIDIA_SITE/curand/include"
  "$NVIDIA_SITE/cusolver/include"
  "$NVIDIA_SITE/cusparse/include"
  "$NVIDIA_SITE/cusparselt/include"
  "$NVIDIA_SITE/nccl/include"
  "$NVIDIA_SITE/nvjitlink/include"
  "$NVIDIA_SITE/nvshmem/include"
  "$NVIDIA_SITE/nvtx/include"
)

LIB_PATHS=(
  "$TORCH_LIB_DIR"
  "$PREFIX/lib"
  "$PREFIX/targets/x86_64-linux/lib"
  "$PREFIX/nvvm/lib64"
  "$NVIDIA_SITE/cuda_runtime/lib"
  "$NVIDIA_SITE/cublas/lib"
  "$NVIDIA_SITE/cudnn/lib"
  "$NVIDIA_SITE/cufft/lib"
  "$NVIDIA_SITE/cufile/lib"
  "$NVIDIA_SITE/curand/lib"
  "$NVIDIA_SITE/cusolver/lib"
  "$NVIDIA_SITE/cusparse/lib"
  "$NVIDIA_SITE/cusparselt/lib"
  "$NVIDIA_SITE/nccl/lib"
  "$NVIDIA_SITE/nvjitlink/lib"
  "$NVIDIA_SITE/nvshmem/lib"
  "$NVIDIA_SITE/nvtx/lib"
)

INCLUDE_JOINED="$(_join_by_colon "${INCLUDE_PATHS[@]}")"
LIB_JOINED="$(_join_by_colon "${LIB_PATHS[@]}")"

export CUDA_HOME="$PREFIX"
export PATH="$PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export CPATH="$INCLUDE_JOINED${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$INCLUDE_JOINED${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$LIB_JOINED${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$LIB_JOINED${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ $# -eq 0 ]]; then
  exec "${SHELL:-/bin/bash}" -l
fi

exec "$@"
