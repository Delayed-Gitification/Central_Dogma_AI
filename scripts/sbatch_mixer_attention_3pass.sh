#!/usr/bin/env bash
#SBATCH --job-name=mixer_attn_3p
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

source ~/.bashrc
conda activate bipangolin
set -eo pipefail

export CACHE_ROOT=/nemo/lab/ulej/home/users/$USER/tmp/synthetic_mamba2_translate_cache
mkdir -p "$CACHE_ROOT"/{triton,torch,xdg,tmp}
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TMPDIR="$CACHE_ROOT/tmp"

cd /nemo/lab/ulej/home/users/wilkino/POSTDOC/software/Dogmamba/Central_Dogma_AI

python -u scripts/train_dense_transition_phase_helper_attention_extra_positional_utr_introns_local_conv_mixers_3pass.py \
  --device cuda \
  --resume-checkpoint checkpoints/curriculum_stage1_mixer_attention/best.pt \
  --checkpoint-dir checkpoints/curriculum_stage1_mixer_attention_3pass \
  --steps 8000 \
  --examples-per-step 64 \
  --validation-examples 512 \
  --eval-batch-size 64 \
  --lr 1e-4 \
  --hidden-dim 96 \
  --conv-layers 4 \
  --use-splice-tracks \
  --materialize-transitions \
  --num-workers 4 \
  --min-utr5-length 1 \
  --max-utr5-length 1 \
  --min-coding-codons 40 \
  --max-coding-codons 140 \
  --min-utr3-length 50 \
  --max-utr3-length 150 \
  --min-exons 2 \
  --max-exons 8 \
  --min-exon-length 6 \
  --min-intron-length 50 \
  --max-intron-length 300 \
  --intron-mod any \
  --no-allow-utr-introns \
  --use-attention-refinement \
  --attention-downsample 8 \
  --attention-layers 2 \
  --attention-heads 4 \
  --delta-logit-scale 1.0 \
  --use-backward-features \
  --extra-attention-layers 2 \
  --extra-attention-init-scale 0.0 \
  --attention-position-encoding sinusoidal \
  --pos-encoding-init-scale 0.0 \
  --use-local-refinement-stem \
  --local-refinement-init-scale 0.001 \
  --mixer-type attention \
  --layer1-loss-weight 0.2 \
  --layer2-loss-weight 1.0 \
  --layer3-loss-weight 1.0 \
  --evidence-loss-weight 10.0 \
  --start-loss-weight 10.0 \
  --stop-loss-weight 10.0 \
  --donor-loss-weight 10.0 \
  --acceptor-loss-weight 10.0
