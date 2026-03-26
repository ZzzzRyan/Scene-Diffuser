#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/install_scene_diffuser_conda_hooks.sh [--prefix /path/to/env]
  bash scripts/install_scene_diffuser_conda_hooks.sh --uninstall [--prefix /path/to/env]

Install or remove conda activate/deactivate hooks that persist the CUDA/PyTorch
runtime paths required by Scene-Diffuser's custom extensions.

By default, the target prefix is taken from the currently activated conda env.
EOF
}

PREFIX="${CONDA_PREFIX:-}"
UNINSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      if [[ $# -lt 2 ]]; then
        echo "--prefix requires a value." >&2
        exit 1
      fi
      PREFIX="$2"
      shift 2
      ;;
    --uninstall)
      UNINSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PREFIX" ]]; then
  echo "No target conda environment found. Activate it first or pass --prefix." >&2
  exit 1
fi

if [[ ! -d "$PREFIX" ]]; then
  echo "Conda prefix does not exist: $PREFIX" >&2
  exit 1
fi

ACTIVATE_DIR="$PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$PREFIX/etc/conda/deactivate.d"
ACTIVATE_FILE="$ACTIVATE_DIR/scene_diffuser_cuda_env.sh"
DEACTIVATE_FILE="$DEACTIVATE_DIR/scene_diffuser_cuda_env.sh"

if [[ "$UNINSTALL" -eq 1 ]]; then
  rm -f "$ACTIVATE_FILE" "$DEACTIVATE_FILE"
  echo "Removed Scene-Diffuser conda hooks from: $PREFIX"
  exit 0
fi

mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat >"$ACTIVATE_FILE" <<'EOF'
#!/usr/bin/env bash

if [[ -n "${SCENE_DIFFUSER_CUDA_HOOK_ACTIVE:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

shopt -s nullglob
_python_site_candidates=("$CONDA_PREFIX"/lib/python*/site-packages)
shopt -u nullglob

if [[ ${#_python_site_candidates[@]} -eq 0 ]]; then
  echo "Scene-Diffuser hook: Python site-packages not found under $CONDA_PREFIX" >&2
  return 1 2>/dev/null || exit 1
fi

PYTHON_SITE="${_python_site_candidates[0]}"
NVIDIA_SITE="$PYTHON_SITE/nvidia"
TORCH_LIB_DIR="$PYTHON_SITE/torch/lib"

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
  "$CONDA_PREFIX/targets/x86_64-linux/include"
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
  "$CONDA_PREFIX/lib"
  "$CONDA_PREFIX/targets/x86_64-linux/lib"
  "$CONDA_PREFIX/nvvm/lib64"
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

export _SCENE_DIFFUSER_CUDA_PREV_CUDA_HOME="${CUDA_HOME-}"
export _SCENE_DIFFUSER_CUDA_PREV_PATH="${PATH-}"
export _SCENE_DIFFUSER_CUDA_PREV_TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-}"
export _SCENE_DIFFUSER_CUDA_PREV_CPATH="${CPATH-}"
export _SCENE_DIFFUSER_CUDA_PREV_CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH-}"
export _SCENE_DIFFUSER_CUDA_PREV_LIBRARY_PATH="${LIBRARY_PATH-}"
export _SCENE_DIFFUSER_CUDA_PREV_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export CPATH="$INCLUDE_JOINED${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$INCLUDE_JOINED${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$LIB_JOINED${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$LIB_JOINED${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SCENE_DIFFUSER_CUDA_HOOK_ACTIVE=1
EOF

cat >"$DEACTIVATE_FILE" <<'EOF'
#!/usr/bin/env bash

if [[ -z "${SCENE_DIFFUSER_CUDA_HOOK_ACTIVE:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_CUDA_HOME+x}" ]]; then
  if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_CUDA_HOME}" ]]; then
    export CUDA_HOME="${_SCENE_DIFFUSER_CUDA_PREV_CUDA_HOME}"
  else
    unset CUDA_HOME
  fi
fi

if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_PATH+x}" ]]; then
  export PATH="${_SCENE_DIFFUSER_CUDA_PREV_PATH}"
fi

if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_TORCH_CUDA_ARCH_LIST+x}" ]]; then
  if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_TORCH_CUDA_ARCH_LIST}" ]]; then
    export TORCH_CUDA_ARCH_LIST="${_SCENE_DIFFUSER_CUDA_PREV_TORCH_CUDA_ARCH_LIST}"
  else
    unset TORCH_CUDA_ARCH_LIST
  fi
fi

if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_CPATH+x}" ]]; then
  if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_CPATH}" ]]; then
    export CPATH="${_SCENE_DIFFUSER_CUDA_PREV_CPATH}"
  else
    unset CPATH
  fi
fi

if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_CPLUS_INCLUDE_PATH+x}" ]]; then
  if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_CPLUS_INCLUDE_PATH}" ]]; then
    export CPLUS_INCLUDE_PATH="${_SCENE_DIFFUSER_CUDA_PREV_CPLUS_INCLUDE_PATH}"
  else
    unset CPLUS_INCLUDE_PATH
  fi
fi

if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_LIBRARY_PATH+x}" ]]; then
  if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_LIBRARY_PATH}" ]]; then
    export LIBRARY_PATH="${_SCENE_DIFFUSER_CUDA_PREV_LIBRARY_PATH}"
  else
    unset LIBRARY_PATH
  fi
fi

if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_LD_LIBRARY_PATH+x}" ]]; then
  if [[ -n "${_SCENE_DIFFUSER_CUDA_PREV_LD_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${_SCENE_DIFFUSER_CUDA_PREV_LD_LIBRARY_PATH}"
  else
    unset LD_LIBRARY_PATH
  fi
fi

unset SCENE_DIFFUSER_CUDA_HOOK_ACTIVE
unset _SCENE_DIFFUSER_CUDA_PREV_CUDA_HOME
unset _SCENE_DIFFUSER_CUDA_PREV_PATH
unset _SCENE_DIFFUSER_CUDA_PREV_TORCH_CUDA_ARCH_LIST
unset _SCENE_DIFFUSER_CUDA_PREV_CPATH
unset _SCENE_DIFFUSER_CUDA_PREV_CPLUS_INCLUDE_PATH
unset _SCENE_DIFFUSER_CUDA_PREV_LIBRARY_PATH
unset _SCENE_DIFFUSER_CUDA_PREV_LD_LIBRARY_PATH
EOF

chmod +x "$ACTIVATE_FILE" "$DEACTIVATE_FILE"

echo "Installed Scene-Diffuser conda hooks into: $PREFIX"
echo "Re-activate the environment to apply them:"
echo "  conda deactivate && conda activate $(basename "$PREFIX")"
