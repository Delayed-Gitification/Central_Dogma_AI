#!/usr/bin/env bash
set -euo pipefail

python scripts/train_pangolin_style_splice_classifier.py \
  --fasta /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa \
  --gtf /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf \
  --train-chroms chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY \
  --val-chroms chr1,chr3,chr5,chr7,chr9 \
  --seq-len 15000 \
  --batch-size 8 \
  --channels 32 \
  --optimizer adam \
  --lr 1e-3 \
  --weight-decay 0 \
  --grad-clip 0 \
  --positive-weight 1 \
  --none-weight 1 \
  --site-prior 0.3333333333333333 \
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
  --checkpoint-dir checkpoints/pangolin_style_splice_classifier_pangolin_split_len15000_ch32
