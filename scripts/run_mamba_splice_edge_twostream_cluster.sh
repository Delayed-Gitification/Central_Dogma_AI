#!/usr/bin/env bash
set -euo pipefail

# Triton/Mamba JIT compilation writes cache files on first run. The default
# cache location can hit home/conda disk quota on the cluster, so keep all JIT
# artifacts in a user-writable project/cache area. Override CACHE_ROOT if needed.
CACHE_ROOT="${CACHE_ROOT:-/nemo/lab/ulej/home/users/wilkino/tmp/mamba_splice_cache}"
mkdir -p "$CACHE_ROOT"/{triton,torch,xdg,tmp}

export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"

python scripts/train_mamba_splice_edge_twostream.py \
  --fasta /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa \
  --gtf /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf \
  --chroms chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22 \
  --seq-len 2048 \
  --batch-size 8 \
  --hidden-dim 96 \
  --shared-layers 4 \
  --branch-layers 1 \
  --chunk-size 64 \
  --headdim 8 \
  --d-conv 4 \
  --layerscale-init 0.1 \
  --site-prior 0.001 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --positive-weight 150 \
  --soft-augment-prob 0 \
  --exist-augment-prob 0 \
  --steps 20000 \
  --print-every 100 \
  --val-batches 8 \
  --device cuda \
  --checkpoint-dir checkpoints/mamba_splice_edge_twostream_hard_only_len2048
