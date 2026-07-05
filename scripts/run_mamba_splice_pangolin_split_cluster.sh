#!/usr/bin/env bash
set -euo pipefail

# Keep Mamba/Triton JIT artifacts out of home/conda cache space on the cluster.
CACHE_ROOT="${CACHE_ROOT:-/nemo/lab/ulej/home/users/wilkino/tmp/mamba_splice_pangolin_split_cache}"
mkdir -p "$CACHE_ROOT"/{triton,torch,xdg,tmp}

export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"

python scripts/train_mamba_splice_soft_exist.py \
  --fasta /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa \
  --gtf /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf \
  --train-chroms chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY \
  --val-chroms chr1,chr3,chr5,chr7,chr9 \
  --seq-len 15000 \
  --target-length 5000 \
  --batch-size 2 \
  --hidden-dim 96 \
  --layers 6 \
  --chunk-size 64 \
  --headdim 8 \
  --local-conv-kernel 9 \
  --optimizer adam \
  --lr 1e-3 \
  --weight-decay 0 \
  --grad-clip 0 \
  --positive-weight 1 \
  --lr-milestones 6,7,8,9 \
  --lr-gamma 0.5 \
  --soft-augment-prob 0 \
  --exist-augment-prob 0 \
  --junk-slots-per-base 0 \
  --epochs 10 \
  --steps-per-epoch 1000 \
  --print-every 100 \
  --val-batches 8 \
  --device cuda \
  --checkpoint-dir checkpoints/mamba_splice_pangolin_split_len15000_target5000_h96_l6
