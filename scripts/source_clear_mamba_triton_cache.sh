#!/usr/bin/env bash
# Source this before running Mamba/Triton Python scripts on the cluster:
#
#   source scripts/source_clear_mamba_triton_cache.sh
#   python scripts/your_script.py ...
#
# Override the cache location if needed:
#
#   MAMBA_TRITON_CACHE_ROOT=/some/big/scratch/path source scripts/source_clear_mamba_triton_cache.sh

if (return 0 2>/dev/null); then
  SOURCED_MAMBA_CACHE_SCRIPT=1
else
  SOURCED_MAMBA_CACHE_SCRIPT=0
fi

fail_mamba_cache_setup() {
  echo "Mamba/Triton cache setup failed: $*" >&2
  return 1 2>/dev/null || exit 1
}

CACHE_ROOT="${MAMBA_TRITON_CACHE_ROOT:-/nemo/lab/ulej/home/users/${USER:-wilkino}/tmp/mamba_triton_cache}"

case "$CACHE_ROOT" in
  ""|"/"|"$HOME"|"$HOME/"|"/camp/lab/ulej/home/shared/Oscar/.conda"|"/camp/lab/ulej/home/shared/Oscar/.conda/")
    echo "Refusing to clear unsafe CACHE_ROOT: '$CACHE_ROOT'" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

echo "Clearing Mamba/Triton cache root:"
echo "  $CACHE_ROOT"

echo "Clearing Triton default runtime caches:"
echo "  $HOME/.triton"
echo "  $HOME/.cache/triton"
rm -rf "$HOME"/.triton \
       "$HOME"/.cache/triton || fail_mamba_cache_setup "could not clear Triton default caches"

rm -rf "$CACHE_ROOT"/triton \
       "$CACHE_ROOT"/torch_extensions \
       "$CACHE_ROOT"/xdg \
       "$CACHE_ROOT"/tmp \
       "$CACHE_ROOT"/cuda \
       "$CACHE_ROOT"/pip \
       "$CACHE_ROOT"/hf || fail_mamba_cache_setup "could not clear $CACHE_ROOT"

mkdir -p "$CACHE_ROOT"/triton \
         "$CACHE_ROOT"/torch_extensions \
         "$CACHE_ROOT"/xdg \
         "$CACHE_ROOT"/tmp \
         "$CACHE_ROOT"/cuda \
         "$CACHE_ROOT"/pip \
         "$CACHE_ROOT"/hf || fail_mamba_cache_setup "could not create $CACHE_ROOT"

export MAMBA_TRITON_CACHE_ROOT="$CACHE_ROOT"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"

echo "Exported:"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "  TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "  XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "  TMPDIR=$TMPDIR"
echo "  CUDA_CACHE_PATH=$CUDA_CACHE_PATH"

if [[ "$SOURCED_MAMBA_CACHE_SCRIPT" -eq 0 ]]; then
  echo
  echo "WARNING: you executed this script instead of sourcing it."
  echo "The cache was cleared, but exports will not persist in your shell."
  echo "Use:"
  echo "  source scripts/source_clear_mamba_triton_cache.sh"
fi
