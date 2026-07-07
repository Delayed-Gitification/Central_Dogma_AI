#!/usr/bin/env bash
#SBATCH --job-name=phase_utr_introns
#SBATCH --partition=gl40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

source ~/.bashrc
conda activate bipangolin
set -eo pipefail

cd /nemo/lab/ulej/home/users/wilkino/POSTDOC/software/Dogmamba/Central_Dogma_AI

python -u scripts/train_dense_transition_phase_helper_attention_extra_positional_utr_introns.py \
  --device cuda \
  --resume-checkpoint checkpoints/dense_transition_phase_helper_attention_utr5_200_extra_attn_positional/best.pt \
  --checkpoint-dir checkpoints/dense_transition_phase_helper_attention_utr5_200_extra_attn_positional_utr_introns_long \
  --steps 100000 \
  --examples-per-step 64 \
  --validation-examples 512 \
  --eval-batch-size 64 \
  --print-every 25 \
  --validate-every 250 \
  --checkpoint-every 250 \
  --hidden-dim 96 \
  --conv-layers 4 \
  --lr 1e-4 \
  --use-splice-tracks \
  --materialize-transitions \
  --num-workers 4 \
  --min-utr5-length 1 \
  --max-utr5-length 200 \
  --min-coding-codons 40 \
  --max-coding-codons 140 \
  --min-utr3-length 100 \
  --max-utr3-length 400 \
  --min-exons 2 \
  --max-exons 8 \
  --min-exon-length 6 \
  --min-intron-length 50 \
  --max-intron-length 300 \
  --intron-mod any \
  --allow-utr-introns \
  --max-utr5-introns 2 \
  --max-utr3-introns 2 \
  --utr-intron-probability 0.75 \
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
  --evidence-loss-weight 10.0 \
  --start-loss-weight 10.0 \
  --stop-loss-weight 10.0 \
  --donor-loss-weight 10.0 \
  --acceptor-loss-weight 10.0
