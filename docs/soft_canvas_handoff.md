# Soft Construct Canvas Handoff

This is the next experimental layer after the donor/acceptor-track emit-skip renderer.

Current working renderer baseline:

- `scripts/train_synthetic_splice_official_mamba2_emit_skip_splice_sites.py`
- Inputs are DNA `[B, L, 4]` plus donor/acceptor tracks `[B, L, 2]`.
- It predicts emit probabilities, uses monotonic cumsum-based transcript packing, and calls `fixed_translate_codons()`.
- This baseline works and should not be broken.

## Critical Implementation Order

Do the first pass as **compaction-only** before touching renderer checkpoints.

Pass 1 should prove only:

```text
overcomplete soft DNA canvas
-> differentiable construct compaction
-> compacted soft DNA construct
-> construct/track reconstruction losses
-> hard-decoded compacted construct equals the target genome
```

Do not load a renderer checkpoint in this first pass. Do not debug renderer integration at the same time as canvas compaction. The milestone is:

```text
construct_exact_match = 1.0
track reconstruction near perfect
presence values mostly 0/1
hard decoded compacted construct equals target genome
```

Only after that works should pass 2 feed the compacted soft DNA and compacted donor/acceptor tracks into the frozen emit-skip renderer checkpoint.

## Goal

Create a new script:

```text
scripts/train_synthetic_splice_official_mamba2_soft_canvas.py
```

Reuse code from:

```text
scripts/train_synthetic_splice_official_mamba2_emit_skip_splice_sites.py
```

but keep the working renderer script intact.

## Core Module

Implement a `SoftConstructCanvas(nn.Module)` with direct trainable logits:

```python
self.base_logits      # [B, L_canvas, 4]
self.presence_logits  # [B, L_canvas]
```

Forward pass:

```python
base_probs = self.base_logits.softmax(dim=-1)
presence = torch.sigmoid(self.presence_logits)
soft_rank = torch.cumsum(presence, dim=1)

target_coordinate = torch.arange(output_length, device=device).float() + 1.0
pack_logits = -construct_sharpness * (
    soft_rank[:, None, :] - target_coordinate[None, :, None]
).pow(2)
pack_logits = pack_logits + torch.log(presence.clamp_min(1e-6))[:, None, :]
construct_pack = pack_logits.softmax(dim=-1)

construct_dna = torch.einsum("bol,blc->boc", construct_pack, base_probs)
construct_tracks = torch.einsum("bol,blk->bok", construct_pack, canvas_tracks)
```

Return `construct_dna`, `construct_tracks`, `construct_pack`, `pack_logits`, `base_probs`, `presence`, and diagnostics.

## Canvas Initialisation

For each true genomic base:

- add one real canvas slot
- add `optional_slots_per_base` optional slots after it
- initialise real slots with high presence logits, for example `+4`
- initialise optional slots with low presence logits, for example `-4`
- initialise real base logits near the true base, with noise
- initialise optional base logits randomly
- copy donor/acceptor tracks to real slots and set optional-slot tracks to zero

The compacted output length should initially equal the original hard genome length.

## Losses

Compaction-only pass:

- construct nucleotide reconstruction loss
- donor/acceptor track reconstruction loss
- presence length loss
- presence entropy/binarisation loss
- base entropy/binarisation loss
- pack entropy/sharpness loss

Renderer pass, after compaction-only works:

- downstream amino-acid loss through the frozen renderer
- downstream mature CDS nucleotide loss through the frozen renderer

## Required Diagnostics

Print regularly:

- total loss
- construct nucleotide loss
- track reconstruction loss
- downstream AA/CDS losses, only when renderer is enabled
- presence length loss
- presence entropy
- base entropy
- pack entropy
- construct nucleotide accuracy
- construct exact match
- track reconstruction metric
- downstream token/exact metrics, only when renderer is enabled
- presence mean/min/max
- presence sum mean
- pack confidence
- construct sharpness

Final example diagnostics:

- target genome
- hard decoded compacted genome
- target CDS
- predicted mature CDS, only when renderer is enabled
- target protein
- predicted protein, only when renderer is enabled
- presence pattern summary
- selected canvas slot count
- first mismatch position

## Constraints

- Do not implement the autoencoder/latent DNA decoder yet.
- Do not implement real SpliceAI integration yet.
- Do not replace the working donor/acceptor emit-skip training script.
- Use a fixed synthetic batch for this proof of concept.
- Keep the implementation boring and debuggable.
