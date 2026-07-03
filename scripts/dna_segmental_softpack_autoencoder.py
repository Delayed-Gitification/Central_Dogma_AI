from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DNA_BASES = "ACGT"
BASE_TO_INDEX = {base: index for index, base in enumerate(DNA_BASES)}


def logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def sequence_to_tensor(sequence: str) -> torch.Tensor:
    return torch.tensor([BASE_TO_INDEX[base] for base in sequence], dtype=torch.long)


def tensor_to_sequence(tensor: torch.Tensor) -> str:
    return "".join(DNA_BASES[index] for index in tensor.detach().cpu().tolist())


def random_dna_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def homopolymer_block(rng: random.Random, length: int) -> str:
    return rng.choice(DNA_BASES) * length


def dinucleotide_repeat_block(rng: random.Random, length: int) -> str:
    motif = rng.choice(DNA_BASES) + rng.choice(DNA_BASES)
    return (motif * ((length + 1) // 2))[:length]


def pyrimidine_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("CTTCC") for _ in range(length))


def motif_repeat_block(rng: random.Random, length: int) -> str:
    motif_length = rng.randint(3, 6)
    motif = random_dna_block(rng, motif_length)
    return (motif * ((length + motif_length - 1) // motif_length))[:length]


def generate_segmental_sequence_with_blocks(seq_len: int, rng: random.Random) -> tuple[str, list[int], list[str]]:
    builders = [
        ("random", random_dna_block, 3, 8),
        ("homopolymer", homopolymer_block, 16, 48),
        ("dinucleotide", dinucleotide_repeat_block, 14, 40),
        ("pyrimidine", pyrimidine_rich_block, 8, 28),
        ("motif", motif_repeat_block, 10, 36),
    ]
    parts = []
    block_lengths = []
    block_types = []
    total_length = 0
    while total_length < seq_len:
        remaining = seq_len - total_length
        block_type, builder, min_len, max_len = rng.choice(builders)
        length = min(remaining, rng.randint(min_len, max_len))
        part = builder(rng, length)
        parts.append(part)
        block_lengths.append(len(part))
        block_types.append(block_type)
        total_length += len(part)
    return "".join(parts)[:seq_len], block_lengths, block_types


def generate_segmental_sequence(seq_len: int, rng: random.Random) -> str:
    sequence, _, _ = generate_segmental_sequence_with_blocks(seq_len, rng)
    return sequence


def make_batch(batch_size: int, seq_len: int, rng: random.Random, device: torch.device) -> torch.Tensor:
    batch = torch.stack([sequence_to_tensor(generate_segmental_sequence(seq_len, rng)) for _ in range(batch_size)])
    return batch.to(device)


def make_batch_with_blocks(
    batch_size: int,
    seq_len: int,
    num_segments: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[list[str]]]:
    sequences = []
    all_block_lengths = []
    all_block_types = []
    block_length_targets = torch.zeros(batch_size, num_segments, dtype=torch.float32)
    block_length_mask = torch.zeros(batch_size, num_segments, dtype=torch.float32)
    for batch_index in range(batch_size):
        sequence, block_lengths, block_types = generate_segmental_sequence_with_blocks(seq_len, rng)
        sequences.append(sequence_to_tensor(sequence))
        all_block_lengths.append(block_lengths)
        all_block_types.append(block_types)
        usable_blocks = min(num_segments, len(block_lengths))
        if usable_blocks > 0:
            block_length_targets[batch_index, :usable_blocks] = torch.tensor(block_lengths[:usable_blocks], dtype=torch.float32)
            block_length_mask[batch_index, :usable_blocks] = 1.0
    return (
        torch.stack(sequences).to(device),
        block_length_targets.to(device),
        block_length_mask.to(device),
        all_block_lengths,
        all_block_types,
    )


class QuerySegmentEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_segments: int,
        latent_dim: int,
        hidden_dim: int,
        layers: int,
        query_position_bias: float,
        query_position_width: float,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("--encoder-layers must be at least 1.")
        if query_position_width <= 0:
            raise ValueError("--query-position-width must be positive.")
        self.query_position_bias = query_position_bias
        self.query_position_width = query_position_width
        blocks = []
        in_channels = 5
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.GELU(),
                ]
            )
            in_channels = hidden_dim
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(hidden_dim)
        self.segment_queries = nn.Parameter(torch.empty(num_segments, hidden_dim))
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.segment_embedding = nn.Parameter(torch.zeros(num_segments, latent_dim))
        nn.init.normal_(self.segment_queries, mean=0.0, std=hidden_dim**-0.5)
        nn.init.normal_(self.segment_embedding, mean=0.0, std=0.02)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(target, num_classes=4).to(dtype=torch.float32)
        positions = torch.linspace(-1.0, 1.0, target.shape[1], device=target.device, dtype=one_hot.dtype)
        positions = positions[None, :, None].expand(target.shape[0], -1, -1)
        encoder_input = torch.cat([one_hot, positions], dim=-1)
        features = self.net(encoder_input.transpose(1, 2)).transpose(1, 2)
        features = self.norm(features)

        logits = torch.einsum("bld,md->bml", features, self.segment_queries) / math.sqrt(float(features.shape[-1]))
        if self.query_position_bias > 0:
            centres = torch.linspace(-1.0, 1.0, self.segment_queries.shape[0], device=target.device, dtype=features.dtype)
            position_values = positions[:, :, 0]
            distance = position_values[:, None, :] - centres[None, :, None]
            logits = logits - self.query_position_bias * distance.pow(2) / (2.0 * self.query_position_width**2)

        attention = logits.softmax(dim=-1)
        context = torch.einsum("bml,bld->bmd", attention, features)
        return self.to_latent(context) + self.segment_embedding[None, :, :]


class SegmentalSoftpackAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        num_segments: int,
        latent_dim: int,
        max_slots_per_segment: int,
        gate_temperature: float,
        pack_temperature: float,
        encoder_hidden_dim: int = 96,
        encoder_layers: int = 2,
        query_position_bias: float = 1.5,
        query_position_width: float = 0.35,
        decoder_hidden_dim: int = 64,
        slot_dim: int = 16,
        initial_active_fraction: float = 1.0,
        initial_length_jitter: float = 0.05,
        initial_usage_jitter: float = 0.25,
    ):
        super().__init__()
        if not 0.0 < initial_active_fraction <= 1.0:
            raise ValueError("--initial-active-fraction must be in (0, 1].")
        self.num_segments = num_segments
        self.latent_dim = latent_dim
        self.max_slots_per_segment = max_slots_per_segment
        self.gate_temperature = gate_temperature
        self.pack_temperature = pack_temperature

        self.encoder = QuerySegmentEncoder(
            num_segments=num_segments,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
            layers=encoder_layers,
            query_position_bias=query_position_bias,
            query_position_width=query_position_width,
        )
        self.slot_embedding = nn.Embedding(max_slots_per_segment, slot_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + slot_dim + 1, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, 4),
        )
        self.length_head = nn.Linear(latent_dim, 1)
        self.usage_head = nn.Linear(latent_dim, 1)
        initial_raw_length = seq_len / float(num_segments * initial_active_fraction)
        initial_raw_length = min(max(initial_raw_length, 1e-3), max_slots_per_segment - 1e-3)
        self.segment_length_logit_bias = nn.Parameter(torch.empty(num_segments))
        self.segment_usage_logit_bias = nn.Parameter(torch.empty(num_segments))
        nn.init.zeros_(self.length_head.weight)
        nn.init.zeros_(self.length_head.bias)
        nn.init.normal_(self.usage_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.usage_head.bias)
        nn.init.normal_(
            self.segment_length_logit_bias,
            mean=logit(initial_raw_length / float(max_slots_per_segment)),
            std=initial_length_jitter,
        )
        nn.init.normal_(
            self.segment_usage_logit_bias,
            mean=logit(initial_active_fraction),
            std=initial_usage_jitter,
        )

    def encode(self, target: torch.Tensor) -> torch.Tensor:
        return self.encoder(target)

    def segment_outputs(
        self,
        z: torch.Tensor,
        length_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_segments, _ = z.shape
        slots = torch.arange(self.max_slots_per_segment, device=z.device)
        slot_embed = self.slot_embedding(slots)
        relative_position = ((slots.to(dtype=z.dtype) + 0.5) / float(self.max_slots_per_segment))[None, None, :, None]

        z_expanded = z[:, :, None, :].expand(-1, -1, self.max_slots_per_segment, -1)
        slot_expanded = slot_embed[None, None, :, :].expand(batch_size, num_segments, -1, -1)
        relative_expanded = relative_position.expand(batch_size, num_segments, -1, -1)
        decoder_input = torch.cat([z_expanded, slot_expanded, relative_expanded], dim=-1)
        base_logits = self.decoder(decoder_input)
        base_probs = base_logits.softmax(dim=-1)

        raw_length_logits = self.length_head(z).squeeze(-1) + self.segment_length_logit_bias[None, :]
        usage_logits = self.usage_head(z).squeeze(-1) + self.segment_usage_logit_bias[None, :]
        raw_lengths = self.max_slots_per_segment * torch.sigmoid(raw_length_logits)
        segment_usage = torch.sigmoid(usage_logits)
        lengths = raw_lengths * segment_usage
        if length_delta is not None:
            lengths = (lengths + length_delta).clamp(0.0, float(self.max_slots_per_segment))
        slot_centres = slots.to(dtype=z.dtype) + 0.5
        keep = torch.sigmoid((lengths[..., None] - slot_centres[None, None, :]) / self.gate_temperature)
        return base_logits, base_probs, lengths, keep, segment_usage

    def render_from_latents(
        self,
        z: torch.Tensor,
        out_len: int,
        length_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base_logits, base_probs_segmented, lengths, keep_segmented, segment_usage = self.segment_outputs(
            z,
            length_delta=length_delta,
        )
        batch_size = z.shape[0]
        num_slots = self.num_segments * self.max_slots_per_segment
        base_probs = base_probs_segmented.reshape(batch_size, num_slots, 4)
        keep = keep_segmented.reshape(batch_size, num_slots)

        end = torch.cumsum(keep, dim=1)
        start = end - keep
        coords = torch.arange(out_len, device=z.device, dtype=z.dtype) + 0.5
        weights = torch.sigmoid((coords[None, None, :] - start[:, :, None]) / self.pack_temperature) - torch.sigmoid(
            (coords[None, None, :] - end[:, :, None]) / self.pack_temperature
        )
        weights = weights.clamp_min(0.0)
        weights_norm = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        soft_dna = torch.einsum("bjl,bjc->blc", weights_norm, base_probs)

        weight_probs = weights_norm.clamp_min(1e-8)
        base_probs_clamped = base_probs.clamp_min(1e-8)
        pack_entropy = -(weight_probs * weight_probs.log()).sum(dim=1).mean()
        base_entropy = -(base_probs_clamped * base_probs_clamped.log()).sum(dim=-1).mean()
        return {
            "base_logits": base_logits,
            "base_probs_segmented": base_probs_segmented,
            "base_probs": base_probs,
            "lengths": lengths,
            "segment_usage": segment_usage,
            "keep_segmented": keep_segmented,
            "keep": keep,
            "total_len": keep.sum(dim=1),
            "weights": weights_norm,
            "soft_dna": soft_dna,
            "pack_entropy": pack_entropy,
            "base_entropy": base_entropy,
            "pack_confidence": weights_norm.max(dim=1).values.mean(),
        }

    def forward(self, target: torch.Tensor, out_len: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(target)
        return z, self.render_from_latents(z, out_len)


def reconstruction_metrics(soft_dna: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = soft_dna.argmax(dim=-1)
    accuracy = (predicted == target).float().mean()
    exact = (predicted == target).all(dim=1).float().mean()
    return {
        "accuracy": float(accuracy.item()),
        "exact": float(exact.item()),
    }


def loss_for_batch(
    *,
    model: SegmentalSoftpackAutoencoder,
    target: torch.Tensor,
    seq_len: int,
    block_lengths: torch.Tensor | None,
    block_length_mask: torch.Tensor | None,
    length_weight: float,
    block_length_weight: float,
    latent_l2_weight: float,
    sharp_weight: float,
    active_segment_weight: float,
    active_segment_threshold: float,
    active_segment_temperature: float,
    usage_sharp_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    z, rendered = model(target, seq_len)
    soft_dna = rendered["soft_dna"].clamp_min(1e-8)
    recon_loss = -soft_dna.gather(-1, target[..., None]).squeeze(-1).log().mean()
    length_loss = F.smooth_l1_loss(rendered["total_len"], torch.full_like(rendered["total_len"], float(seq_len)))
    if block_length_weight > 0 and block_lengths is not None and block_length_mask is not None:
        block_lengths = block_lengths.to(dtype=rendered["lengths"].dtype)
        block_length_mask = block_length_mask.to(dtype=rendered["lengths"].dtype)
        block_length_error = F.smooth_l1_loss(rendered["lengths"], block_lengths, reduction="none")
        block_length_loss = (block_length_error * block_length_mask).sum() / block_length_mask.sum().clamp_min(1.0)
    else:
        block_length_loss = rendered["lengths"].new_zeros(())
    latent_l2 = z.pow(2).mean()
    sharp_loss = (rendered["keep"] * (1.0 - rendered["keep"])).mean()
    length_active = torch.sigmoid((rendered["lengths"] - active_segment_threshold) / active_segment_temperature)
    active_segments = rendered["segment_usage"] * length_active
    active_segment_loss = active_segments.sum(dim=1).mean()
    usage_sharp_loss = (rendered["segment_usage"] * (1.0 - rendered["segment_usage"])).sum(dim=1).mean()
    loss = (
        recon_loss
        + length_weight * length_loss
        + block_length_weight * block_length_loss
        + latent_l2_weight * latent_l2
        + sharp_weight * sharp_loss
        + active_segment_weight * active_segment_loss
        + usage_sharp_weight * usage_sharp_loss
    )
    return loss, {
        **rendered,
        "z": z.detach(),
        "recon_loss": recon_loss.detach(),
        "length_loss": length_loss.detach(),
        "block_length_loss": block_length_loss.detach(),
        "latent_l2": latent_l2.detach(),
        "sharp_loss": sharp_loss.detach(),
        "active_segment_loss": active_segment_loss.detach(),
        "active_segments": active_segments.detach(),
        "length_active": length_active.detach(),
        "usage_sharp_loss": usage_sharp_loss.detach(),
    }


def hard_decode(
    model: SegmentalSoftpackAutoencoder,
    z: torch.Tensor,
    out_len: int | None = None,
    length_delta: torch.Tensor | None = None,
) -> str:
    rendered = model.render_from_latents(z, out_len or 1, length_delta=length_delta)
    if out_len is None:
        out_len = max(1, int(round(float(rendered["total_len"][0].detach().cpu().item()))))
        rendered = model.render_from_latents(z, out_len, length_delta=length_delta)
    return tensor_to_sequence(rendered["soft_dna"].argmax(dim=-1)[0])


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, first_char in enumerate(first, start=1):
        current = [i]
        for j, second_char in enumerate(second, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (first_char != second_char),
                )
            )
        previous = current
    return previous[-1]


def changed_positions(first: str, second: str) -> list[int]:
    return [index for index, (left, right) in enumerate(zip(first, second)) if left != right]


def pick_device(args: argparse.Namespace) -> torch.device:
    if args.mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --mps, but MPS is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def current_sharp_weight(step: int, steps: int, sharp_weight: float, warmup_frac: float) -> float:
    if sharp_weight <= 0:
        return 0.0
    warmup_steps = int(steps * warmup_frac)
    if step < warmup_steps:
        return 0.0
    ramp_steps = max(1, steps - warmup_steps)
    return sharp_weight * min(1.0, float(step - warmup_steps + 1) / float(ramp_steps))


def summarise_lengths(lengths: torch.Tensor, total_len: torch.Tensor) -> str:
    flat_lengths = lengths.detach().flatten()
    zero_segments = (lengths.detach() < 1.0).float().sum(dim=1).mean()
    return (
        f"total_len {total_len.mean().item():.1f}/{total_len.std(unbiased=False).item():.1f} "
        f"range {total_len.min().item():.1f}-{total_len.max().item():.1f} | "
        f"seg_len {flat_lengths.mean().item():.1f}/{flat_lengths.std(unbiased=False).item():.1f} "
        f"range {flat_lengths.min().item():.1f}-{flat_lengths.max().item():.1f} | "
        f"near_zero {zero_segments.item():.2f}"
    )


def summarise_active_segments(active_segments: torch.Tensor) -> str:
    active_count = active_segments.detach().sum(dim=1)
    return (
        f"active {active_count.mean().item():.1f}/{active_count.std(unbiased=False).item():.1f} "
        f"range {active_count.min().item():.1f}-{active_count.max().item():.1f}"
    )


def summarise_segment_usage(segment_usage: torch.Tensor) -> str:
    usage = segment_usage.detach()
    return (
        f"usage {usage.mean().item():.2f}/{usage.std(unbiased=False).item():.2f} "
        f"range {usage.min().item():.2f}-{usage.max().item():.2f}"
    )


def summarise_true_blocks(block_lengths: list[list[int]]) -> str:
    counts = torch.tensor([len(lengths) for lengths in block_lengths], dtype=torch.float32)
    flat = torch.tensor([length for lengths in block_lengths for length in lengths], dtype=torch.float32)
    return (
        f"true_blocks {counts.mean().item():.1f}/{counts.std(unbiased=False).item():.1f} "
        f"count_range {counts.min().item():.0f}-{counts.max().item():.0f} | "
        f"true_block_len {flat.mean().item():.1f}/{flat.std(unbiased=False).item():.1f} "
        f"range {flat.min().item():.0f}-{flat.max().item():.0f}"
    )


def block_length_correlation(
    predicted_lengths: torch.Tensor,
    block_lengths: list[list[int]],
) -> float:
    predicted_values = []
    target_values = []
    for batch_index, lengths in enumerate(block_lengths):
        usable_blocks = min(predicted_lengths.shape[1], len(lengths))
        if usable_blocks == 0:
            continue
        predicted_values.append(predicted_lengths[batch_index, :usable_blocks].detach().cpu())
        target_values.append(torch.tensor(lengths[:usable_blocks], dtype=torch.float32))
    if not predicted_values:
        return 0.0
    predicted = torch.cat(predicted_values)
    target = torch.cat(target_values)
    predicted = predicted - predicted.mean()
    target = target - target.mean()
    denom = predicted.norm() * target.norm()
    if denom.item() == 0.0:
        return 0.0
    return float((predicted * target).sum().div(denom).item())


def tensor_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() < 2 or second.numel() < 2:
        return 0.0
    first = first.detach().float().cpu().flatten()
    second = second.detach().float().cpu().flatten()
    first = first - first.mean()
    second = second - second.mean()
    denom = first.norm() * second.norm()
    if denom.item() == 0.0:
        return 0.0
    return float((first * second).sum().div(denom).item())


def summarise_segment_entropy(rendered: dict[str, torch.Tensor], target: torch.Tensor) -> str:
    lengths = rendered["lengths"].detach()
    weights = rendered["weights"].detach()
    batch_size, num_segments = lengths.shape
    max_slots_per_segment = weights.shape[1] // num_segments
    segment_weights = weights.reshape(batch_size, num_segments, max_slots_per_segment, target.shape[1]).sum(dim=2)
    segment_mass = segment_weights.sum(dim=-1)

    one_hot = F.one_hot(target, num_classes=4).to(dtype=segment_weights.dtype)
    base_mass = torch.einsum("bml,blc->bmc", segment_weights, one_hot)
    base_probs = base_mass / (base_mass.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1) / math.log(4.0)

    mask = segment_mass > 0.25
    if not mask.any():
        return "seg_entropy n/a"

    length_values = lengths[mask].detach().float().cpu()
    entropy_values = entropy[mask].detach().float().cpu()
    order = torch.argsort(length_values)
    group_size = max(1, int(math.ceil(float(order.numel()) * 0.25)))
    short_entropy = entropy_values[order[:group_size]].mean()
    long_entropy = entropy_values[order[-group_size:]].mean()
    return (
        f"seg_entropy {entropy_values.mean().item():.3f} "
        f"len_entropy_corr {tensor_correlation(length_values, entropy_values):.3f} "
        f"short_entropy {short_entropy.item():.3f} long_entropy {long_entropy.item():.3f}"
    )


def format_metrics(prefix: str, loss: torch.Tensor, rendered: dict[str, torch.Tensor], target: torch.Tensor) -> str:
    metrics = reconstruction_metrics(rendered["soft_dna"], target)
    return (
        f"{prefix:<5} loss {loss.item():.4f} ce {rendered['recon_loss'].item():.4f} "
        f"acc {metrics['accuracy']:.3f} exact {metrics['exact']:.3f} "
        f"len {rendered['length_loss'].item():.3f} "
        f"block_len {rendered['block_length_loss'].item():.3f} "
        f"active_loss {rendered['active_segment_loss'].item():.3f} "
        f"usage_sharp {rendered['usage_sharp_loss'].item():.3f} "
        f"base_ent {rendered['base_entropy'].item():.3f} "
        f"pack_ent {rendered['pack_entropy'].item():.3f} "
        f"pack_conf {rendered['pack_confidence'].item():.3f}"
    )


def evaluate_fresh_batches(
    *,
    model: SegmentalSoftpackAutoencoder,
    rng: random.Random,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    batches: int,
    num_segments: int,
    length_weight: float,
    block_length_weight: float,
    latent_l2_weight: float,
    sharp_weight: float,
    active_segment_weight: float,
    active_segment_threshold: float,
    active_segment_temperature: float,
    usage_sharp_weight: float,
) -> tuple[float, dict[str, float], dict[str, torch.Tensor], torch.Tensor, list[list[int]]]:
    total_loss = 0.0
    total_accuracy = 0.0
    total_exact = 0.0
    last_rendered: dict[str, torch.Tensor] | None = None
    last_batch: torch.Tensor | None = None
    last_block_lengths_list: list[list[int]] | None = None
    for _ in range(batches):
        batch, block_lengths, block_mask, block_lengths_list, _ = make_batch_with_blocks(
            batch_size,
            seq_len,
            num_segments,
            rng,
            device,
        )
        loss, rendered = loss_for_batch(
            model=model,
            target=batch,
            seq_len=seq_len,
            block_lengths=block_lengths,
            block_length_mask=block_mask,
            length_weight=length_weight,
            block_length_weight=block_length_weight,
            latent_l2_weight=latent_l2_weight,
            sharp_weight=sharp_weight,
            active_segment_weight=active_segment_weight,
            active_segment_threshold=active_segment_threshold,
            active_segment_temperature=active_segment_temperature,
            usage_sharp_weight=usage_sharp_weight,
        )
        metrics = reconstruction_metrics(rendered["soft_dna"], batch)
        total_loss += loss.item()
        total_accuracy += metrics["accuracy"]
        total_exact += metrics["exact"]
        last_rendered = rendered
        last_batch = batch
        last_block_lengths_list = block_lengths_list
    assert last_rendered is not None and last_batch is not None and last_block_lengths_list is not None
    return (
        total_loss / float(batches),
        {
            "accuracy": total_accuracy / float(batches),
            "exact": total_exact / float(batches),
        },
        last_rendered,
        last_batch,
        last_block_lengths_list,
    )


def run_diagnostics(
    *,
    model: SegmentalSoftpackAutoencoder,
    rng: random.Random,
    device: torch.device,
    seq_len: int,
) -> None:
    model.eval()
    with torch.no_grad():
        target_tensor = make_batch(1, seq_len, rng, device)
        z = model.encode(target_tensor)
        original = hard_decode(model, z, out_len=seq_len)
        target = tensor_to_sequence(target_tensor[0])

        content_perturbed_z = z.clone()
        content_perturbed_z[:, 0, :] = content_perturbed_z[:, 0, :] + torch.randn_like(content_perturbed_z[:, 0, :]) * 0.5
        perturbed = hard_decode(model, content_perturbed_z, out_len=seq_len)
        changed = changed_positions(original, perturbed)

        length_delta = torch.zeros(1, model.num_segments, device=device)
        segment_index = min(1, model.num_segments - 1)
        length_delta[:, segment_index] = 10.0
        length_modified = hard_decode(model, z, out_len=seq_len + 10, length_delta=length_delta)
        rendered = model.render_from_latents(z, seq_len)
        lengths = rendered["lengths"][0].detach().cpu()
        segment_usage = rendered["segment_usage"][0].detach().cpu()

    print("\nDiagnostics:")
    print("target:       ", target)
    print("reconstructed:", original)
    print("perturbed:    ", perturbed)
    print("length +10:   ", length_modified)
    print("segment lengths:", ", ".join(f"{value:.1f}" for value in lengths.tolist()))
    print("segment usage:  ", ", ".join(f"{value:.2f}" for value in segment_usage.tolist()))
    print("perturb edit distance:", edit_distance(original, perturbed))
    print("perturb changed positions:", changed[:30], "..." if len(changed) > 30 else "")
    model.train()


def train(args: argparse.Namespace) -> None:
    if args.num_segments * args.max_slots_per_segment < args.seq_len:
        raise ValueError("--num-segments * --max-slots-per-segment must be at least --seq-len.")
    if args.gate_temperature <= 0 or args.pack_temperature <= 0:
        raise ValueError("--gate-temperature and --pack-temperature must be positive.")
    if args.batch_size <= 0 or args.val_batches <= 0:
        raise ValueError("--batch-size and --val-batches must be positive.")
    if args.active_segment_temperature <= 0:
        raise ValueError("--active-segment-temperature must be positive.")
    if args.block_length_weight < 0:
        raise ValueError("--block-length-weight must be non-negative.")
    if args.initial_length_jitter < 0 or args.initial_usage_jitter < 0:
        raise ValueError("--initial-length-jitter and --initial-usage-jitter must be non-negative.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_rng = random.Random(args.seed)
    val_rng = random.Random(args.seed + 1_000_000)
    diagnostic_rng = random.Random(args.seed + 2_000_000)

    model = SegmentalSoftpackAutoencoder(
        seq_len=args.seq_len,
        num_segments=args.num_segments,
        latent_dim=args.latent_dim,
        max_slots_per_segment=args.max_slots_per_segment,
        gate_temperature=args.gate_temperature,
        pack_temperature=args.pack_temperature,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_layers=args.encoder_layers,
        query_position_bias=args.query_position_bias,
        query_position_width=args.query_position_width,
        decoder_hidden_dim=args.decoder_hidden_dim,
        initial_active_fraction=args.initial_active_fraction,
        initial_length_jitter=args.initial_length_jitter,
        initial_usage_jitter=args.initial_usage_jitter,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")

    param_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"fresh synthetic batches; seq_len={args.seq_len}; batch_size={args.batch_size}")
    print(
        f"segments={args.num_segments}; latent_dim={args.latent_dim}; "
        f"max_slots_per_segment={args.max_slots_per_segment}; params={param_count}"
    )
    print(
        f"query_position_bias={args.query_position_bias}; "
        f"initial_active_fraction={args.initial_active_fraction}; "
        f"active_segment_weight={args.active_segment_weight}; "
        f"block_length_weight={args.block_length_weight}"
    )

    for step in range(args.steps):
        batch, block_lengths, block_mask, _, _ = make_batch_with_blocks(
            args.batch_size,
            args.seq_len,
            args.num_segments,
            train_rng,
            device,
        )
        sharp_weight = current_sharp_weight(step, args.steps, args.sharp_weight, args.sharp_warmup_frac)
        usage_sharp_weight = current_sharp_weight(
            step,
            args.steps,
            args.usage_sharp_weight,
            args.usage_sharp_warmup_frac,
        )
        loss, rendered = loss_for_batch(
            model=model,
            target=batch,
            seq_len=args.seq_len,
            block_lengths=block_lengths,
            block_length_mask=block_mask,
            length_weight=args.length_weight,
            block_length_weight=args.block_length_weight,
            latent_l2_weight=args.latent_l2_weight,
            sharp_weight=sharp_weight,
            active_segment_weight=args.active_segment_weight,
            active_segment_threshold=args.active_segment_threshold,
            active_segment_temperature=args.active_segment_temperature,
            usage_sharp_weight=usage_sharp_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_rendered, val_batch, val_block_lengths = evaluate_fresh_batches(
                    model=model,
                    rng=val_rng,
                    device=device,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    batches=args.val_batches,
                    num_segments=args.num_segments,
                    length_weight=args.length_weight,
                    block_length_weight=args.block_length_weight,
                    latent_l2_weight=args.latent_l2_weight,
                    sharp_weight=sharp_weight,
                    active_segment_weight=args.active_segment_weight,
                    active_segment_threshold=args.active_segment_threshold,
                    active_segment_temperature=args.active_segment_temperature,
                    usage_sharp_weight=usage_sharp_weight,
                )
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "step": step,
                        "validation_loss": best_val_loss,
                    },
                    checkpoint_dir / "best.pt",
                )

            print(
                f"\nstep {step:06d} sharp_w {sharp_weight:.2e} usage_sharp_w {usage_sharp_weight:.2e} "
                f"val_loss {val_loss:.4f} val_acc {val_metrics['accuracy']:.3f} val_exact {val_metrics['exact']:.3f}"
            )
            print(format_metrics("train", loss, rendered, batch))
            print(format_metrics("val", torch.tensor(val_loss), val_rendered, val_batch))
            print(summarise_lengths(val_rendered["lengths"], val_rendered["total_len"]))
            print(summarise_active_segments(val_rendered["active_segments"]))
            print(summarise_segment_usage(val_rendered["segment_usage"]))
            print(summarise_true_blocks(val_block_lengths))
            print(f"block_len_corr {block_length_correlation(val_rendered['lengths'], val_block_lengths):.3f}")
            print(summarise_segment_entropy(val_rendered, val_batch))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": args.steps - 1,
            "validation_loss": best_val_loss,
        },
        checkpoint_dir / "latest.pt",
    )
    run_diagnostics(model=model, rng=diagnostic_rng, device=device, seq_len=args.seq_len)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fresh-batch segmental SoftPack DNA autoencoder.")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--num-segments", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--max-slots-per-segment", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--length-weight", type=float, default=0.05)
    parser.add_argument("--block-length-weight", type=float, default=0.0)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=0.2)
    parser.add_argument("--pack-temperature", type=float, default=0.1)
    parser.add_argument("--sharp-weight", type=float, default=0.002)
    parser.add_argument("--sharp-warmup-frac", type=float, default=0.2)
    parser.add_argument("--active-segment-weight", type=float, default=0.0)
    parser.add_argument("--active-segment-threshold", type=float, default=1.0)
    parser.add_argument("--active-segment-temperature", type=float, default=0.5)
    parser.add_argument("--usage-sharp-weight", type=float, default=0.0)
    parser.add_argument("--usage-sharp-warmup-frac", type=float, default=0.2)
    parser.add_argument("--encoder-hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--query-position-bias", type=float, default=1.5)
    parser.add_argument("--query-position-width", type=float, default=0.35)
    parser.add_argument("--initial-active-fraction", type=float, default=1.0)
    parser.add_argument("--initial-length-jitter", type=float, default=0.05)
    parser.add_argument("--initial-usage-jitter", type=float, default=0.25)
    parser.add_argument("--decoder-hidden-dim", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="checkpoints/dna_segmental_softpack_autoencoder")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
