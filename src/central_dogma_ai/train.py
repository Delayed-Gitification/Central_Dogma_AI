"""Train the optional typed splice-translation baseline."""

from __future__ import annotations

import argparse

from central_dogma_ai.synthetic import SyntheticGeneConfig, generate_examples
from central_dogma_ai.torch_model import TypedSpliceTransducer, protein_to_target, require_torch


def _batch_to_tensors(batch):
    torch = require_torch()
    max_length = max(len(example.dna_one_hot) for example in batch)
    dna_rows = []
    mask_rows = []
    track_rows = []
    for example in batch:
        padding = max_length - len(example.dna_one_hot)
        dna_rows.append(example.dna_one_hot + [[0.0, 0.0, 0.0, 0.0]] * padding)
        mask_rows.append(example.exon_mask + [0] * padding)
        track_rows.append(example.splice_tracks + [[0.0, 0.0, 0.0]] * padding)
    dna = torch.tensor(dna_rows, dtype=torch.float32)
    mask = torch.tensor(mask_rows, dtype=torch.bool)
    tracks = torch.tensor(track_rows, dtype=torch.float32)
    target = protein_to_target([example.protein for example in batch])
    return dna, mask, tracks, target


def train(steps: int = 200, batch_size: int = 64, amino_acid_codons: int = 32, seed: int = 1) -> dict[str, float]:
    """Train pointer heads and return exact-product and pointer metrics."""

    torch = require_torch()
    config = SyntheticGeneConfig(
        amino_acid_codons=amino_acid_codons,
        min_exons=3,
        max_exons=5,
        min_intron_length=8,
        max_intron_length=24,
        include_skip_isoform=False,
    )
    model = TypedSpliceTransducer.build(hidden_dim=96, layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    pointer_loss_fn = torch.nn.BCEWithLogitsLoss()

    for step in range(steps):
        examples = generate_examples(batch_size, config=config, seed=seed + step)
        dna, mask, tracks, _target = _batch_to_tensors(examples)
        output = model(dna, splice_tracks=tracks, exon_mask=mask)
        pointer_logits = output["pointer_logits"]
        loss = (
            pointer_loss_fn(pointer_logits["include_logits"], tracks[:, :, 0])
            + pointer_loss_fn(pointer_logits["acceptor_logits"], tracks[:, :, 1])
            + pointer_loss_fn(pointer_logits["donor_logits"], tracks[:, :, 2])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    eval_examples = generate_examples(batch_size, config=config, seed=seed + steps + 1000)
    dna, mask, tracks, target = _batch_to_tensors(eval_examples)
    with torch.no_grad():
        output = model(dna, splice_tracks=tracks, exon_mask=mask)
        predictions = output["amino_acid_probs"].argmax(dim=-1)
        include_predictions = output["pointer_logits"]["include_logits"].sigmoid() >= 0.5
    exact = (predictions == target).all(dim=1).float().mean().item()
    include_accuracy = (include_predictions == mask).float().mean().item()
    return {"exact_match_accuracy": exact, "include_pointer_accuracy": include_accuracy}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amino-acid-codons", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    metrics = train(
        steps=args.steps,
        batch_size=args.batch_size,
        amino_acid_codons=args.amino_acid_codons,
        seed=args.seed,
    )
    for metric_name, value in metrics.items():
        print(f"{metric_name}={value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())