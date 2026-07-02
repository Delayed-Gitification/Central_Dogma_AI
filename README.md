# Central Dogma AI

This is a small starting implementation for the "start in the middle" idea: make spliced translation an explicit, typed object before trying to design alternatively spliced vectors.

The package currently provides:

- an exact splice-aware translation compiler;
- synthetic spliced-gene examples with one-hot DNA, exon masks, and amino-acid targets;
- a typed PyTorch transducer scaffold with a selective-scan DNA encoder, splice pointer heads, an explicit codon buffer, a frozen codon-table layer, and isoform consequence outputs;
- tests that check frame handling, split codons, premature stops, and synthetic data consistency.

The design philosophy is deliberately conservative: hard biological invariants are computed exactly, while learned modules are reserved for the fuzzy parts.

## Quick Start

```bash
python -m central_dogma_ai.cli --examples 2 --seed 7
python -m pytest
```

For the optional neural scaffold:

```bash
python -m pip install -e '.[ml]'
python -m central_dogma_ai.train --steps 200 --batch-size 64
```

## Core Representation

The central object is a `SpliceProgram`:

```text
genomic DNA
exon table with genomic coordinates
isoform table with exon paths
```

From that, the compiler can assemble each transcript, track exon phases, split codons across exon junctions, translate ORFs, and flag early stops/NMD-like risk.

The supervised learning examples expose the same biology as tensors/lists:

```text
DNA one-hot:    L x 4
splice tracks:  L x 3  (included, acceptor, donor)
path indices:   T genomic pointers in transcript order
AA target:      C x 21
```

where `C` is the number of complete codons in the spliced transcript and the amino-acid alphabet includes `*` for stop.

## Why This Exists

The eventual ambition is not just to generate DNA that scores well under a predictor. The ambition is to design alternatively spliced vector cassettes where the model has native concepts of exon path, coding frame, exon phase, stop codons, and isoform-specific protein products.

This first implementation is the substrate for that: a reproducible benchmark and code path for asking whether learned components can operate inside an exact splicing/translation grammar.

## Architecture

The optional PyTorch model is a first concrete version of the architecture discussed in the planning thread:

```text
DNA one-hot + splice tracks
-> selective-scan encoder
-> splice pointer heads over genomic positions
-> transcript-order path pointers
-> codon-buffer layer
-> frozen codon-table layer
-> amino-acid probabilities + stop/PTC/NMD-style summaries
```

The current transducer can use exact path pointers or derive them from an exon mask. The pointer heads are present so the next benchmark can train path inference from donor/acceptor/splice-prediction tracks rather than handing the path to the exact branch.