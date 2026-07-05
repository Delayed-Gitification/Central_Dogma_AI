#!/usr/bin/env bash
set -euo pipefail

python scripts/train_pangolin_style_splice_classifier.py \
  --fasta /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa \
  --gtf /camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf \
  --chroms chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22 \
  --seq-len 20000 \
  --batch-size 8 \
  --channels 64 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --positive-weight 100 \
  --none-weight 1 \
  --site-prior 0.001 \
  --soft-augment-prob 0 \
  --exist-augment-prob 0 \
  --junk-slots-per-base 0 \
  --steps 20000 \
  --print-every 100 \
  --val-batches 8 \
  --device cuda \
  --checkpoint-dir checkpoints/pangolin_style_splice_classifier_len20000_ch64
