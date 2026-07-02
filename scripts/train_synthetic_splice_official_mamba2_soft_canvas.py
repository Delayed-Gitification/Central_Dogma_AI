from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from central_dogma_ai.biology import (  # noqa: E402
    AMINO_ACIDS,
    AA_TO_INDEX,
    CODONS_BY_AA,
    DNA_BASES,
    DNA_TO_INDEX,
)


NONSTOP_AA = [aa for aa in AMINO_ACIDS if aa != "*"]
MODEL_TRACK_NAMES = ("donor", "acceptor")


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def one_hot_dna(sequence: str) -> torch.Tensor:
    encoded = torch.zeros(len(sequence), 4)
    for index, base in enumerate(sequence):
        encoded[index, DNA_TO_INDEX[base]] = 1.0
    return encoded


def random_protein(codons: int, rng: random.Random, terminal_stop: bool = True) -> str:
    if terminal_stop:
        return "".join(rng.choice(NONSTOP_AA) for _ in range(codons - 1)) + "*"
    return "".join(rng.choice(NONSTOP_AA) for _ in range(codons))


def reverse_translate(protein: str, rng: random.Random) -> str:
    return "".join(rng.choice(CODONS_BY_AA[amino_acid]) for amino_acid in protein)


def random_split_lengths(total_length: int, parts: int, rng: random.Random, min_part: int = 5) -> list[int]:
    if total_length < parts * min_part:
        raise ValueError("total_length is too short for the requested split")
    remaining = total_length
    lengths = []
    for part in range(parts - 1):
        max_length = remaining - min_part * (parts - part - 1)
        length = rng.randint(min_part, max_length)
        lengths.append(length)
        remaining -= length
    lengths.append(remaining)
    return lengths


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def make_synthetic_example(
    protein_codons: int,
    exon_count: int,
    min_intron_length: int,
    max_intron_length: int,
    rng: random.Random,
    min_exon_bases: int = 5,
) -> dict[str, object]:
    protein = random_protein(protein_codons, rng)
    cds = reverse_translate(protein, rng)
    exon_lengths = random_split_lengths(len(cds), exon_count, rng, min_part=min_exon_bases)

    genome_parts = []
    exon_prior = []
    donor_track = []
    acceptor_track = []
    true_transcript_rank = []
    intron_lengths = []

    cds_cursor = 0
    transcript_cursor = 0
    for exon_index, exon_length in enumerate(exon_lengths):
        exon = cds[cds_cursor : cds_cursor + exon_length]
        cds_cursor += exon_length
        for base_index, base in enumerate(exon):
            genome_parts.append(base)
            exon_prior.append(1.0)
            donor_track.append(1.0 if exon_index < exon_count - 1 and base_index == exon_length - 1 else 0.0)
            acceptor_track.append(1.0 if exon_index > 0 and base_index == 0 else 0.0)
            true_transcript_rank.append(float(transcript_cursor))
            transcript_cursor += 1

        if exon_index < exon_count - 1:
            intron_length = rng.randint(min_intron_length, max_intron_length)
            intron_lengths.append(intron_length)
            intron = "GT" + random_dna(intron_length - 4, rng) + "AG"
            for base in intron:
                genome_parts.append(base)
                exon_prior.append(0.0)
                donor_track.append(0.0)
                acceptor_track.append(0.0)
                true_transcript_rank.append(-1.0)

    genome = "".join(genome_parts)
    target = torch.tensor([AA_TO_INDEX[amino_acid] for amino_acid in protein], dtype=torch.long)
    cds_target = torch.tensor([DNA_TO_INDEX[base] for base in cds], dtype=torch.long)
    model_tracks = torch.tensor(list(zip(donor_track, acceptor_track)), dtype=torch.float32)
    emit_target = torch.tensor(exon_prior, dtype=torch.float32)
    annotations = torch.tensor(
        list(zip(exon_prior, donor_track, acceptor_track, true_transcript_rank)),
        dtype=torch.float32,
    )
    return {
        "genome": genome,
        "protein": protein,
        "cds": cds,
        "protein_codons": protein_codons,
        "exon_count": exon_count,
        "exon_lengths": exon_lengths,
        "intron_lengths": intron_lengths,
        "dna": one_hot_dna(genome),
        "tracks": model_tracks,
        "annotations": annotations,
        "emit_target": emit_target,
        "target": target,
        "cds_target": cds_target,
    }


def make_fixed_batch(
    *,
    batch_size: int,
    device: torch.device,
    min_protein_codons: int,
    max_protein_codons: int,
    min_exon_count: int,
    max_exon_count: int,
    min_exon_bases: int,
    min_intron_length: int,
    max_intron_length: int,
    length_bucket_size: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    list[dict[str, object]],
]:
    rng = random.Random(seed)
    examples = []
    for _ in range(batch_size):
        protein_codons = rng.randint(min_protein_codons, max_protein_codons)
        max_allowed_exons = max(1, min(max_exon_count, (protein_codons * 3) // min_exon_bases))
        min_allowed_exons = min(min_exon_count, max_allowed_exons)
        exon_count = rng.randint(min_allowed_exons, max_allowed_exons)
        examples.append(
            make_synthetic_example(
                protein_codons=protein_codons,
                exon_count=exon_count,
                min_intron_length=min_intron_length,
                max_intron_length=max_intron_length,
                rng=rng,
                min_exon_bases=min_exon_bases,
            )
        )

    max_genome_length = round_up_to_multiple(
        max(example["dna"].shape[0] for example in examples),
        length_bucket_size,
    )
    max_target_length = max(example["target"].shape[0] for example in examples)
    max_transcript_bases = max_target_length * 3

    dna_rows = []
    track_rows = []
    genome_mask_rows = []
    target_rows = []
    target_mask_rows = []
    cds_rows = []
    cds_mask_rows = []
    for example in examples:
        dna = example["dna"]
        tracks = example["tracks"]
        target = example["target"]
        cds_target = example["cds_target"]
        genome_padding = max_genome_length - dna.shape[0]
        target_padding = max_target_length - target.shape[0]
        cds_padding = max_transcript_bases - cds_target.shape[0]
        dna_rows.append(torch.cat([dna, torch.zeros(genome_padding, 4)], dim=0))
        track_rows.append(torch.cat([tracks, torch.zeros(genome_padding, tracks.shape[1])], dim=0))
        genome_mask_rows.append(
            torch.cat([torch.ones(dna.shape[0], dtype=torch.bool), torch.zeros(genome_padding, dtype=torch.bool)])
        )
        target_rows.append(torch.cat([target, torch.zeros(target_padding, dtype=target.dtype)], dim=0))
        target_mask_rows.append(
            torch.cat([torch.ones(target.shape[0], dtype=torch.bool), torch.zeros(target_padding, dtype=torch.bool)])
        )
        cds_rows.append(torch.cat([cds_target, torch.zeros(cds_padding, dtype=cds_target.dtype)], dim=0))
        cds_mask_rows.append(
            torch.cat([torch.ones(cds_target.shape[0], dtype=torch.bool), torch.zeros(cds_padding, dtype=torch.bool)])
        )

    return (
        torch.stack(dna_rows).to(device),
        torch.stack(track_rows).to(device),
        torch.stack(genome_mask_rows).to(device),
        torch.stack(target_rows).to(device),
        torch.stack(target_mask_rows).to(device),
        torch.stack(cds_rows).to(device),
        torch.stack(cds_mask_rows).to(device),
        max_transcript_bases,
        examples,
    )


class SoftConstructCanvas(nn.Module):
    def __init__(
        self,
        *,
        base_logits: torch.Tensor,
        presence_logits: torch.Tensor,
        canvas_tracks: torch.Tensor,
        canvas_mask: torch.Tensor,
        construct_sharpness_init: float,
        max_construct_sharpness: float,
    ):
        super().__init__()
        self.base_logits = nn.Parameter(base_logits)
        self.presence_logits = nn.Parameter(presence_logits)
        self.log_construct_sharpness = nn.Parameter(torch.tensor(float(construct_sharpness_init)).log())
        self.max_construct_sharpness = max_construct_sharpness
        self.register_buffer("canvas_tracks", canvas_tracks)
        self.register_buffer("canvas_mask", canvas_mask)

    def forward(self, output_length: int):
        base_probs = self.base_logits.softmax(dim=-1)
        canvas_mask = self.canvas_mask.to(dtype=base_probs.dtype)
        presence = torch.sigmoid(self.presence_logits) * canvas_mask
        soft_rank = torch.cumsum(presence, dim=1)

        target_index = torch.arange(output_length, device=base_probs.device)
        target_coordinate = target_index.to(dtype=base_probs.dtype) + 1.0
        construct_sharpness = F.softplus(self.log_construct_sharpness).clamp(max=self.max_construct_sharpness)
        pack_logits = -construct_sharpness * (
            soft_rank[:, None, :] - target_coordinate[None, :, None]
        ).pow(2)
        pack_logits = pack_logits + torch.log(presence.clamp_min(1e-6))[:, None, :]
        construct_pack = pack_logits.softmax(dim=-1)

        construct_dna = torch.einsum("bol,blc->boc", construct_pack, base_probs)
        construct_tracks = torch.einsum("bol,blk->bok", construct_pack, self.canvas_tracks)
        pack_entropy = -(
            construct_pack.clamp_min(1e-8) * construct_pack.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        pack_confidence = construct_pack.max(dim=-1).values.mean()
        presence_entropy = -(
            presence * presence.clamp_min(1e-8).log()
            + (1.0 - presence) * (1.0 - presence).clamp_min(1e-8).log()
        )
        presence_entropy = (presence_entropy * canvas_mask).sum() / canvas_mask.sum().clamp_min(1.0)
        base_entropy = -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1)
        base_entropy = (base_entropy * canvas_mask).sum() / canvas_mask.sum().clamp_min(1.0)

        return construct_dna, construct_tracks, construct_pack, pack_logits, base_probs, presence, {
            "construct_sharpness": construct_sharpness.detach(),
            "pack_entropy": pack_entropy,
            "pack_confidence": pack_confidence.detach(),
            "presence_entropy": presence_entropy,
            "base_entropy": base_entropy,
            "presence_mean": presence.detach()[self.canvas_mask].mean(),
            "presence_min": presence.detach()[self.canvas_mask].min(),
            "presence_max": presence.detach()[self.canvas_mask].max(),
            "presence_sum_mean": presence.detach().sum(dim=1).mean(),
        }


def make_soft_canvas_initialisation(
    *,
    examples: list[dict[str, object]],
    optional_slots_per_base: int,
    base_logit_init_strength: float,
    presence_present_init: float,
    presence_absent_init: float,
    noise: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_logits_rows = []
    presence_logits_rows = []
    canvas_track_rows = []
    canvas_mask_rows = []
    max_canvas_length = max(len(str(example["genome"])) * (optional_slots_per_base + 1) for example in examples)

    for example in examples:
        base_logits = []
        presence_logits = []
        canvas_tracks = []
        canvas_mask = []
        tracks = example["tracks"]
        for genome_index, base in enumerate(str(example["genome"])):
            real_logits = torch.randn(4) * noise
            real_logits[DNA_TO_INDEX[base]] += base_logit_init_strength
            base_logits.append(real_logits)
            presence_logits.append(torch.tensor(presence_present_init + random.gauss(0.0, noise)))
            canvas_tracks.append(tracks[genome_index].detach().cpu())
            canvas_mask.append(torch.tensor(True))

            for _ in range(optional_slots_per_base):
                base_logits.append(torch.randn(4) * max(noise, 1e-6))
                presence_logits.append(torch.tensor(presence_absent_init + random.gauss(0.0, noise)))
                canvas_tracks.append(torch.zeros(2))
                canvas_mask.append(torch.tensor(True))

        padding = max_canvas_length - len(base_logits)
        if padding:
            base_logits.extend(torch.zeros(4) for _ in range(padding))
            presence_logits.extend(torch.full((), presence_absent_init) for _ in range(padding))
            canvas_tracks.extend(torch.zeros(2) for _ in range(padding))
            canvas_mask.extend(torch.tensor(False) for _ in range(padding))

        base_logits_rows.append(torch.stack(base_logits))
        presence_logits_rows.append(torch.stack(presence_logits))
        canvas_track_rows.append(torch.stack(canvas_tracks))
        canvas_mask_rows.append(torch.stack(canvas_mask))

    return (
        torch.stack(base_logits_rows).to(device),
        torch.stack(presence_logits_rows).to(device),
        torch.stack(canvas_track_rows).to(device),
        torch.stack(canvas_mask_rows).to(device),
    )


def load_renderer(args: argparse.Namespace, device: torch.device) -> nn.Module | None:
    if not args.renderer_checkpoint:
        return None

    scripts_dir = ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import train_synthetic_splice_official_mamba2_emit_skip_splice_sites as renderer_module  # noqa: E402

    renderer = renderer_module.MambaEmitSkipTranslator(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        chunk_size=args.chunk_size,
        headdim=args.headdim,
        use_prior_emit_mask=False,
        max_assignment_sharpness=args.max_assignment_sharpness,
    ).to(device)
    checkpoint_path = Path(args.renderer_checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    renderer.load_state_dict(checkpoint["model_state_dict"])
    if args.freeze_renderer:
        for parameter in renderer.parameters():
            parameter.requires_grad_(False)
        renderer.eval()
    return renderer


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.to(values.dtype)).sum() / mask.sum().clamp_min(1)


def first_mismatch(target: list[int], predicted: list[int]) -> int | None:
    for index, (target_value, predicted_value) in enumerate(zip(target, predicted)):
        if target_value != predicted_value:
            return index
    return None


def hard_decode_construct(
    *,
    pack_logits: torch.Tensor,
    base_probs: torch.Tensor,
    construct_mask: torch.Tensor,
) -> list[str]:
    selected_canvas_positions = pack_logits.argmax(dim=-1)
    hard_canvas_bases = base_probs.argmax(dim=-1)
    decoded = []
    for batch_index in range(pack_logits.shape[0]):
        length = int(construct_mask[batch_index].sum().item())
        selected = selected_canvas_positions[batch_index, :length]
        bases = hard_canvas_bases[batch_index].gather(0, selected)
        decoded.append("".join(DNA_BASES[index] for index in bases.tolist()))
    return decoded


def train(args: argparse.Namespace) -> None:
    if args.optional_slots_per_base < 0:
        raise ValueError("--optional-slots-per-base must be non-negative")
    if args.canvas_steps < 1:
        raise ValueError("--canvas-steps must be positive")
    if args.renderer_checkpoint and args.compaction_only:
        raise ValueError("--compaction-only cannot be combined with --renderer-checkpoint")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    (
        target_dna,
        target_tracks,
        construct_mask,
        target_aa,
        target_aa_mask,
        target_cds,
        target_cds_mask,
        transcript_bases,
        examples,
    ) = make_fixed_batch(
        batch_size=args.batch_size,
        device=device,
        min_protein_codons=args.min_protein_codons,
        max_protein_codons=args.max_protein_codons,
        min_exon_count=args.min_exon_count,
        max_exon_count=args.max_exon_count,
        min_exon_bases=args.min_exon_bases,
        min_intron_length=args.min_intron_length,
        max_intron_length=args.max_intron_length,
        length_bucket_size=args.length_bucket_size,
        seed=args.seed,
    )
    output_length = target_dna.shape[1]
    target_bases = target_dna.argmax(dim=-1)

    base_logits, presence_logits, canvas_tracks, canvas_mask = make_soft_canvas_initialisation(
        examples=examples,
        optional_slots_per_base=args.optional_slots_per_base,
        base_logit_init_strength=args.base_logit_init_strength,
        presence_present_init=args.presence_present_init,
        presence_absent_init=args.presence_absent_init,
        noise=args.canvas_noise,
        device=device,
    )
    canvas = SoftConstructCanvas(
        base_logits=base_logits,
        presence_logits=presence_logits,
        canvas_tracks=canvas_tracks,
        canvas_mask=canvas_mask,
        construct_sharpness_init=args.construct_sharpness_init,
        max_construct_sharpness=args.max_construct_sharpness,
    ).to(device)
    if args.load_canvas_checkpoint:
        canvas_checkpoint_path = Path(args.load_canvas_checkpoint).expanduser()
        if not canvas_checkpoint_path.is_absolute():
            canvas_checkpoint_path = ROOT / canvas_checkpoint_path
        try:
            canvas_checkpoint = torch.load(canvas_checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            canvas_checkpoint = torch.load(canvas_checkpoint_path, map_location=device)
        canvas.load_state_dict(canvas_checkpoint["canvas_state_dict"])
        print(f"loaded canvas checkpoint: {canvas_checkpoint_path}")

    renderer = load_renderer(args, device)
    renderer_enabled = renderer is not None and not args.compaction_only
    optimizer_parameters = list(canvas.parameters())
    if renderer is not None and not args.freeze_renderer:
        optimizer_parameters.extend(renderer.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.canvas_learning_rate)

    print("soft canvas optimisation")
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}; construct length: {output_length}; transcript_bases: {transcript_bases}")
    print(f"canvas slots: {canvas_mask.shape[1]}; optional_slots_per_base: {args.optional_slots_per_base}")
    print("mode: renderer integration" if renderer_enabled else "mode: compaction-only")
    if renderer_enabled:
        print(f"renderer checkpoint: {args.renderer_checkpoint}; freeze_renderer={args.freeze_renderer}")

    final_metrics = {}
    for step in range(args.canvas_steps):
        optimizer.zero_grad(set_to_none=True)
        construct_dna, construct_tracks, construct_pack, pack_logits, base_probs, presence, diagnostics = canvas(output_length)

        per_nt_loss = F.nll_loss(
            torch.log(construct_dna.clamp_min(1e-8)).reshape(-1, 4),
            target_bases.reshape(-1),
            reduction="none",
        ).reshape_as(target_bases)
        construct_nt_loss = masked_mean(per_nt_loss, construct_mask)

        per_track_loss = F.binary_cross_entropy(
            construct_tracks.clamp(1e-6, 1.0 - 1e-6),
            target_tracks,
            reduction="none",
        ).mean(dim=-1)
        track_loss = masked_mean(per_track_loss, construct_mask)

        target_lengths = construct_mask.sum(dim=1).to(presence.dtype)
        presence_length_loss = (presence.sum(dim=1) - target_lengths).pow(2).mean()
        pack_entropy = diagnostics["pack_entropy"]
        presence_entropy = diagnostics["presence_entropy"]
        base_entropy = diagnostics["base_entropy"]

        downstream_aa_loss = torch.zeros((), device=device)
        downstream_nt_loss = torch.zeros((), device=device)
        downstream_token_accuracy = 0.0
        downstream_exact = 0.0
        downstream_nt_accuracy = 0.0
        downstream_nt_exact = 0.0
        if renderer_enabled:
            renderer_construct_dna = construct_dna * construct_mask[..., None].to(construct_dna.dtype)
            renderer_construct_tracks = construct_tracks * construct_mask[..., None].to(construct_tracks.dtype)
            amino_acid_probs, transcript_base_probs, _assignment, _assignment_logits, _renderer_diag = renderer(
                renderer_construct_dna,
                renderer_construct_tracks,
                transcript_bases=transcript_bases,
            )
            aa_loss_rows = F.nll_loss(
                torch.log(amino_acid_probs.clamp_min(1e-8)).reshape(-1, len(AMINO_ACIDS)),
                target_aa.reshape(-1),
                reduction="none",
            ).reshape_as(target_aa)
            downstream_aa_loss = masked_mean(aa_loss_rows, target_aa_mask)
            nt_loss_rows = F.nll_loss(
                torch.log(transcript_base_probs.clamp_min(1e-8)).reshape(-1, len(DNA_BASES)),
                target_cds.reshape(-1),
                reduction="none",
            ).reshape_as(target_cds)
            downstream_nt_loss = masked_mean(nt_loss_rows, target_cds_mask)
            with torch.no_grad():
                predicted_aa = amino_acid_probs.argmax(dim=-1)
                predicted_cds = transcript_base_probs.argmax(dim=-1)
                downstream_token_accuracy = float(((predicted_aa == target_aa) & target_aa_mask).sum().item()) / max(
                    1, int(target_aa_mask.sum().item())
                )
                downstream_exact = float(((predicted_aa == target_aa) | ~target_aa_mask).all(dim=1).float().mean().item())
                downstream_nt_accuracy = float(((predicted_cds == target_cds) & target_cds_mask).sum().item()) / max(
                    1, int(target_cds_mask.sum().item())
                )
                downstream_nt_exact = float(
                    ((predicted_cds == target_cds) | ~target_cds_mask).all(dim=1).float().mean().item()
                )

        loss = (
            args.construct_nt_loss_weight * construct_nt_loss
            + args.track_loss_weight * track_loss
            + args.length_loss_weight * presence_length_loss
            + args.presence_entropy_weight * presence_entropy
            + args.base_entropy_weight * base_entropy
            + args.pack_entropy_weight * pack_entropy
            + args.downstream_aa_loss_weight * downstream_aa_loss
            + args.downstream_nt_loss_weight * downstream_nt_loss
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            hard_construct = construct_dna.argmax(dim=-1)
            construct_correct = (hard_construct == target_bases) & construct_mask
            construct_nt_accuracy = float(construct_correct.sum().item()) / max(1, int(construct_mask.sum().item()))
            construct_exact = float(((hard_construct == target_bases) | ~construct_mask).all(dim=1).float().mean().item())
            track_mse = masked_mean((construct_tracks - target_tracks).pow(2).mean(dim=-1), construct_mask)
            track_binary = (construct_tracks > 0.5).to(target_tracks.dtype)
            track_position_correct = (track_binary == target_tracks).all(dim=-1) & construct_mask
            track_accuracy = float(track_position_correct.sum().item()) / max(
                1, int(construct_mask.sum().item())
            )
            final_metrics = {
                "loss": float(loss.item()),
                "construct_nt_loss": float(construct_nt_loss.item()),
                "track_loss": float(track_loss.item()),
                "presence_length_loss": float(presence_length_loss.item()),
                "presence_entropy": float(presence_entropy.item()),
                "base_entropy": float(base_entropy.item()),
                "pack_entropy": float(pack_entropy.item()),
                "construct_nt_accuracy": construct_nt_accuracy,
                "construct_exact_match": construct_exact,
                "track_mse": float(track_mse.item()),
                "track_accuracy": track_accuracy,
                "downstream_aa_loss": float(downstream_aa_loss.item()),
                "downstream_nt_loss": float(downstream_nt_loss.item()),
                "downstream_token_accuracy": downstream_token_accuracy,
                "downstream_exact_match": downstream_exact,
                "downstream_nt_accuracy": downstream_nt_accuracy,
                "downstream_nt_exact_match": downstream_nt_exact,
                "presence_mean": float(diagnostics["presence_mean"].item()),
                "presence_min": float(diagnostics["presence_min"].item()),
                "presence_max": float(diagnostics["presence_max"].item()),
                "presence_sum_mean": float(diagnostics["presence_sum_mean"].item()),
                "pack_confidence": float(diagnostics["pack_confidence"].item()),
                "construct_sharpness": float(diagnostics["construct_sharpness"].item()),
            }

        if step % args.print_every == 0 or step == args.canvas_steps - 1:
            print(
                f"\nstep {step:05d} loss {final_metrics['loss']:.4f} | "
                f"construct nt {final_metrics['construct_nt_loss']:.4f} "
                f"acc {final_metrics['construct_nt_accuracy']:.3f} "
                f"exact {final_metrics['construct_exact_match']:.3f} | "
                f"track {final_metrics['track_loss']:.4f} mse {final_metrics['track_mse']:.5f} "
                f"acc {final_metrics['track_accuracy']:.3f}"
            )
            print(
                f"presence len {final_metrics['presence_length_loss']:.4f} "
                f"entropy {final_metrics['presence_entropy']:.4f} "
                f"sum {final_metrics['presence_sum_mean']:.2f} "
                f"range {final_metrics['presence_min']:.3f}-{final_metrics['presence_max']:.3f} | "
                f"base_entropy {final_metrics['base_entropy']:.4f} "
                f"pack_entropy {final_metrics['pack_entropy']:.4f} "
                f"pack_conf {final_metrics['pack_confidence']:.3f} "
                f"sharp {final_metrics['construct_sharpness']:.3f}"
            )
            if renderer_enabled:
                print(
                    f"downstream aa {final_metrics['downstream_aa_loss']:.4f} "
                    f"tok {final_metrics['downstream_token_accuracy']:.3f} "
                    f"exact {final_metrics['downstream_exact_match']:.3f} | "
                    f"cds {final_metrics['downstream_nt_loss']:.4f} "
                    f"nt {final_metrics['downstream_nt_accuracy']:.3f} "
                    f"nt_exact {final_metrics['downstream_nt_exact_match']:.3f}"
                )

    print("\nFinal metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")

    with torch.no_grad():
        construct_dna, construct_tracks, construct_pack, pack_logits, base_probs, presence, _diagnostics = canvas(output_length)
        decoded_constructs = hard_decode_construct(
            pack_logits=pack_logits,
            base_probs=base_probs,
            construct_mask=construct_mask,
        )
        predicted_cds = None
        predicted_protein = None
        if renderer_enabled:
            renderer_construct_dna = construct_dna * construct_mask[..., None].to(construct_dna.dtype)
            renderer_construct_tracks = construct_tracks * construct_mask[..., None].to(construct_tracks.dtype)
            amino_acid_probs, transcript_base_probs, _assignment, _assignment_logits, _renderer_diag = renderer(
                renderer_construct_dna,
                renderer_construct_tracks,
                transcript_bases=transcript_bases,
            )
            eval_aa_length = int(target_aa_mask[0].sum().item())
            eval_cds_length = int(target_cds_mask[0].sum().item())
            predicted_protein = "".join(
                AMINO_ACIDS[index] for index in amino_acid_probs.argmax(dim=-1)[0, :eval_aa_length].tolist()
            )
            predicted_cds = "".join(
                DNA_BASES[index] for index in transcript_base_probs.argmax(dim=-1)[0, :eval_cds_length].tolist()
            )
        first_example = examples[0]
        target_genome = str(first_example["genome"])
        decoded = decoded_constructs[0]
        target_indices = [DNA_TO_INDEX[base] for base in target_genome]
        decoded_indices = [DNA_TO_INDEX[base] for base in decoded]
        mismatch = first_mismatch(target_indices, decoded_indices)
        selected_slots = pack_logits.argmax(dim=-1)[0, : len(target_genome)]
        unique_selected = int(torch.unique(selected_slots).numel())
        selected_presence = presence[0, selected_slots].detach().cpu()

    print("\nHeld-out canvas example:")
    print("target genome: ", target_genome)
    print("decoded genome:", decoded)
    print("target cds:    ", first_example["cds"])
    if predicted_cds is not None:
        print("predicted cds: ", predicted_cds)
    print("target protein:", first_example["protein"])
    if predicted_protein is not None:
        print("pred protein:  ", predicted_protein)
    print("selected canvas slots:", unique_selected)
    print("selected presence mean:", float(selected_presence.mean().item()))
    print("selected presence min:", float(selected_presence.min().item()))
    print("selected presence max:", float(selected_presence.max().item()))
    print("first mismatch:", mismatch if mismatch is not None else "none")

    if args.save_canvas_checkpoint:
        canvas_checkpoint_path = Path(args.save_canvas_checkpoint).expanduser()
        if not canvas_checkpoint_path.is_absolute():
            canvas_checkpoint_path = ROOT / canvas_checkpoint_path
        canvas_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "canvas_state_dict": canvas.state_dict(),
                "args": vars(args),
                "final_metrics": final_metrics,
            },
            canvas_checkpoint_path,
        )
        print(f"saved canvas checkpoint: {canvas_checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise an overcomplete soft construct canvas before optional Dogmamba renderer integration."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-protein-codons", type=int, default=24)
    parser.add_argument("--max-protein-codons", type=int, default=48)
    parser.add_argument("--min-exon-count", type=int, default=1)
    parser.add_argument("--max-exon-count", type=int, default=3)
    parser.add_argument("--min-exon-bases", type=int, default=9)
    parser.add_argument("--min-intron-length", type=int, default=10)
    parser.add_argument("--max-intron-length", type=int, default=80)
    parser.add_argument("--length-bucket-size", type=int, default=128)
    parser.add_argument("--renderer-checkpoint", default="")
    parser.add_argument("--freeze-renderer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compaction-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load-canvas-checkpoint", default="")
    parser.add_argument("--save-canvas-checkpoint", default="")
    parser.add_argument("--optional-slots-per-base", type=int, default=2)
    parser.add_argument("--canvas-steps", type=int, default=2_000)
    parser.add_argument("--canvas-learning-rate", type=float, default=3e-2)
    parser.add_argument("--presence-entropy-weight", type=float, default=0.01)
    parser.add_argument("--base-entropy-weight", type=float, default=0.01)
    parser.add_argument("--pack-entropy-weight", type=float, default=0.01)
    parser.add_argument("--length-loss-weight", type=float, default=0.001)
    parser.add_argument("--construct-nt-loss-weight", type=float, default=1.0)
    parser.add_argument("--track-loss-weight", type=float, default=1.0)
    parser.add_argument("--downstream-aa-loss-weight", type=float, default=1.0)
    parser.add_argument("--downstream-nt-loss-weight", type=float, default=0.5)
    parser.add_argument("--construct-sharpness-init", type=float, default=4.0)
    parser.add_argument("--max-construct-sharpness", type=float, default=50.0)
    parser.add_argument("--base-logit-init-strength", type=float, default=5.0)
    parser.add_argument("--presence-present-init", type=float, default=4.0)
    parser.add_argument("--presence-absent-init", type=float, default=-4.0)
    parser.add_argument("--canvas-noise", type=float, default=0.25)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=8)
    parser.add_argument("--max-assignment-sharpness", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
