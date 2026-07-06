#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [[ -z "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
  # Slurm batch shells often do not inherit the interactive conda activation.
  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate bipangolin
  set -u
fi

echo "host: $(hostname)"
echo "python: $(command -v python)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python -c 'import torch; print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "torch_cuda", torch.version.cuda, "device_count", torch.cuda.device_count()); torch.cuda.init(); print("cuda init ok", torch.cuda.get_device_name(0))'

# Keep Mamba/Triton JIT artifacts out of home/conda cache space on the cluster.
CACHE_ROOT="${CACHE_ROOT:-/nemo/lab/ulej/home/users/wilkino/tmp/multiscale_mamba_splice_classifier_cache}"
mkdir -p "$CACHE_ROOT"/{triton,torch,xdg,tmp}

export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"

python scripts/train_multiscale_mamba_splice_classifier.py \
  --fasta /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa \
  --gtf /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf \
  --train-chroms chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY \
  --val-chroms chr1,chr3,chr5,chr7,chr9 \
  --seq-len 15000 \
  --target-length 5000 \
  --batch-size 4 \
  --hidden-dim 96 \
  --layers 6 \
  --multiscale-layers 2 \
  --multiscale-stride 8 \
  --chunk-size 64 \
  --headdim 8 \
  --local-conv-kernel 9 \
  --optimizer adamw \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1 \
  --positive-weight 300 \
  --none-weight 1 \
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
  --checkpoint-dir checkpoints/multiscale_mamba_splice_classifier_pangolin_split_len15000_target5000_h96_l6_c2_s8_class_pos300_adamw
