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


def sequence_to_tensor(sequence: str) -> torch.Tensor:
    return torch.tensor([BASE_TO_INDEX[base] for base in sequence], dtype=torch.long)


def tensor_to_sequence(tensor: torch.Tensor) -> str:
    return "".join(DNA_BASES[index] for index in tensor.detach().cpu().tolist())


def random_dna_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def gc_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("GGGCCAT") for _ in range(length))


def at_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("AAATTGC") for _ in range(length))


def pyrimidine_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("CTTCC") for _ in range(length))


def purine_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("AAGGA") for _ in range(length))


def homopolymer_block(rng: random.Random, length: int) -> str:
    return rng.choice(DNA_BASES) * length


def dinucleotide_repeat_block(rng: random.Random, length: int) -> str:
    motif = rng.choice(DNA_BASES) + rng.choice(DNA_BASES)
    return (motif * ((length + 1) // 2))[:length]


def motif_repeat_block(rng: random.Random, length: int) -> str:
    motif_length = rng.randint(3, 6)
    motif = random_dna_block(rng, motif_length)
    return (motif * ((length + motif_length - 1) // motif_length))[:length]


BLOCK_BUILDERS = {
    "random": random_dna_block,
    "gc": gc_rich_block,
    "at": at_rich_block,
    "pyrimidine": pyrimidine_rich_block,
    "purine": purine_rich_block,
    "homopolymer": homopolymer_block,
    "dinucleotide": dinucleotide_repeat_block,
    "motif": motif_repeat_block,
}


TRAIN_FAMILIES = (
    "random",
    "gc",
    "at",
    "pyrimidine",
    "purine",
    "homopolymer",
    "dinucleotide",
    "motif",
    "hybrid",
)


def block_length_for_family(family: str, rng: random.Random, remaining: int) -> int:
    if family == "random":
        length = rng.randint(3, 12)
    elif family in {"gc", "at", "pyrimidine", "purine"}:
        length = rng.randint(6, 28)
    elif family == "homopolymer":
        length = rng.randint(8, 72)
    elif family == "dinucleotide":
        length = rng.randint(6, 72)
    elif family == "motif":
        length = rng.randint(8, 72)
    else:
        length = rng.randint(3, 72)
    return min(remaining, length)


def generate_controlled_sequence(
    *,
    seq_len: int,
    block_type: str,
    block_len: int,
    rng: random.Random,
) -> str:
    builder = BLOCK_BUILDERS[block_type]
    parts = []
    total = 0
    while total < seq_len:
        length = min(block_len, seq_len - total)
        parts.append(builder(rng, length))
        total += length
    return "".join(parts)


def generate_family_sequence(seq_len: int, family: str, rng: random.Random) -> tuple[str, str]:
    parts = []
    total = 0
    while total < seq_len:
        remaining = seq_len - total
        block_type = rng.choice(tuple(BLOCK_BUILDERS)) if family == "hybrid" else family
        length = block_length_for_family(block_type, rng, remaining)
        parts.append(BLOCK_BUILDERS[block_type](rng, length))
        total += length
    return "".join(parts), family


def generate_variable_sequence(
    *,
    max_seq_len: int,
    min_seq_len: int,
    short_max_len_frac: float,
    long_min_len_frac: float,
    rng: random.Random,
) -> tuple[str, str]:
    short_max_len = max(min_seq_len, int(round(max_seq_len * short_max_len_frac)))
    long_min_len = min(max_seq_len, max(min_seq_len, int(round(max_seq_len * long_min_len_frac))))
    length_mode = rng.random()
    if length_mode < 0.4:
        seq_len = rng.randint(min_seq_len, short_max_len)
    elif length_mode < 0.8:
        seq_len = rng.randint(long_min_len, max_seq_len)
    else:
        seq_len = rng.randint(min_seq_len, max_seq_len)
    return generate_family_sequence(seq_len, rng.choice(TRAIN_FAMILIES), rng)


def make_batch(
    *,
    batch_size: int,
    max_seq_len: int,
    min_seq_len: int,
    short_max_len_frac: float,
    long_min_len_frac: float,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    target = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.float32)
    kinds = []
    for batch_index in range(batch_size):
        sequence, kind = generate_variable_sequence(
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            short_max_len_frac=short_max_len_frac,
            long_min_len_frac=long_min_len_frac,
            rng=rng,
        )
        sequence_tensor = sequence_to_tensor(sequence)
        seq_len = len(sequence)
        target[batch_index, :seq_len] = sequence_tensor
        mask[batch_index, :seq_len] = 1.0
        lengths[batch_index] = seq_len
        kinds.append(kind)
    return target.to(device), mask.to(device), lengths.to(device), kinds


def make_probe_batch(
    *,
    max_seq_len: int,
    seq_lengths: list[int],
    block_types: list[str],
    block_lengths: list[int],
    examples_per_condition: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[str, int, int]]]:
    total_examples = len(seq_lengths) * len(block_types) * len(block_lengths) * examples_per_condition
    target = torch.zeros(total_examples, max_seq_len, dtype=torch.long)
    mask = torch.zeros(total_examples, max_seq_len, dtype=torch.float32)
    lengths = torch.zeros(total_examples, dtype=torch.float32)
    labels: list[tuple[str, int, int]] = []
    index = 0
    for block_type in block_types:
        for seq_len in seq_lengths:
            for block_len in block_lengths:
                for _ in range(examples_per_condition):
                    sequence = generate_controlled_sequence(
                        seq_len=seq_len,
                        block_type=block_type,
                        block_len=min(block_len, seq_len),
                        rng=rng,
                    )
                    sequence_tensor = sequence_to_tensor(sequence)
                    target[index, :seq_len] = sequence_tensor
                    mask[index, :seq_len] = 1.0
                    lengths[index] = seq_len
                    labels.append((block_type, seq_len, block_len))
                    index += 1
    return target.to(device), mask.to(device), lengths.to(device), labels


class AdaptiveTokenizerAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len: int,
        max_tokens: int,
        latent_dim: int,
        max_slots_per_token: int,
        hidden_dim: int,
        encoder_layers: int,
        decoder_hidden_dim: int,
        slot_dim: int,
        token_rank_temperature: float,
        token_usage_temperature: float,
        gate_temperature: float,
        pack_temperature: float,
        initial_token_stride: float,
    ):
        super().__init__()
        if max_tokens <= 0:
            raise ValueError("--max-tokens must be positive.")
        if initial_token_stride <= 0:
            raise ValueError("--initial-token-stride must be positive.")
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.max_slots_per_token = max_slots_per_token
        self.token_rank_temperature = token_rank_temperature
        self.token_usage_temperature = token_usage_temperature
        self.gate_temperature = gate_temperature
        self.pack_temperature = pack_temperature

        blocks = []
        in_channels = 6
        for _ in range(encoder_layers):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.GELU(),
                ]
            )
            in_channels = hidden_dim
        self.encoder = nn.Sequential(*blocks)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.token_head = nn.Linear(hidden_dim, 1)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.slot_embedding = nn.Embedding(max_slots_per_token, slot_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + slot_dim + 1, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, 4),
        )
        self.length_head = nn.Linear(latent_dim, 1)

        initial_token_prob = min(0.95, max(0.01, 1.0 / initial_token_stride))
        nn.init.constant_(self.token_head.bias, math.log(initial_token_prob / (1.0 - initial_token_prob)))
        nn.init.normal_(self.token_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.length_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.length_head.bias, math.log(8.0 / max(1.0, max_slots_per_token - 8.0)))

    def encode_features(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(target, num_classes=4).to(dtype=torch.float32)
        mask = mask.to(dtype=one_hot.dtype)
        positions = torch.linspace(-1.0, 1.0, target.shape[1], device=target.device, dtype=one_hot.dtype)
        positions = positions[None, :, None].expand(target.shape[0], -1, -1)
        encoder_input = torch.cat([one_hot * mask[..., None], positions, mask[..., None]], dim=-1)
        features = self.encoder(encoder_input.transpose(1, 2)).transpose(1, 2)
        return self.encoder_norm(features)

    def tokenize(self, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_features(target, mask)
        token_logits = self.token_head(features).squeeze(-1)
        token_prob = torch.sigmoid(token_logits) * mask

        # Ensure every non-empty sequence can open at least one token while keeping
        # the remaining token count differentiable.
        first_position = F.one_hot(torch.zeros(target.shape[0], dtype=torch.long, device=target.device), target.shape[1])
        token_prob = torch.maximum(token_prob, first_position.to(dtype=token_prob.dtype) * mask[:, :1])

        token_centres = torch.cumsum(token_prob, dim=1) - 0.5 * token_prob
        token_ids = torch.arange(self.max_tokens, device=target.device, dtype=features.dtype) + 0.5
        assignment_logits = -(
            token_centres[:, None, :] - token_ids[None, :, None]
        ).pow(2) / max(self.token_rank_temperature, 1e-6)
        assignment_logits = assignment_logits.masked_fill(mask[:, None, :] <= 0, -1e4)
        token_weights = assignment_logits.softmax(dim=-1)

        token_mass = token_prob.sum(dim=1)
        token_usage = torch.sigmoid(
            (token_mass[:, None] - token_ids[None, :]) / max(self.token_usage_temperature, 1e-6)
        )
        token_weights = token_weights * token_usage[:, :, None]
        token_weights_norm = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        pooled = torch.einsum("bkl,blh->bkh", token_weights_norm, features)
        latents = self.to_latent(pooled)
        return {
            "features": features,
            "latents": latents,
            "token_logits": token_logits,
            "token_prob": token_prob,
            "token_mass": token_mass,
            "token_usage": token_usage,
            "token_weights": token_weights,
        }

    def decode(self, latents: torch.Tensor, token_usage: torch.Tensor, out_len: int) -> dict[str, torch.Tensor]:
        batch_size = latents.shape[0]
        slots = torch.arange(self.max_slots_per_token, device=latents.device)
        slot_embed = self.slot_embedding(slots)
        relative_position = ((slots.to(dtype=latents.dtype) + 0.5) / float(self.max_slots_per_token))[None, None, :, None]

        latent_expanded = latents[:, :, None, :].expand(-1, -1, self.max_slots_per_token, -1)
        slot_expanded = slot_embed[None, None, :, :].expand(batch_size, self.max_tokens, -1, -1)
        relative_expanded = relative_position.expand(batch_size, self.max_tokens, -1, -1)
        decoder_input = torch.cat([latent_expanded, slot_expanded, relative_expanded], dim=-1)
        base_logits_segmented = self.decoder(decoder_input)
        base_probs_segmented = base_logits_segmented.softmax(dim=-1)

        raw_lengths = self.max_slots_per_token * torch.sigmoid(self.length_head(latents).squeeze(-1))
        lengths = raw_lengths * token_usage
        slot_centres = slots.to(dtype=latents.dtype) + 0.5
        keep_segmented = torch.sigmoid((lengths[..., None] - slot_centres[None, None, :]) / self.gate_temperature)

        num_slots = self.max_tokens * self.max_slots_per_token
        base_probs = base_probs_segmented.reshape(batch_size, num_slots, 4)
        keep = keep_segmented.reshape(batch_size, num_slots)
        end = torch.cumsum(keep, dim=1)
        start = end - keep
        coords = torch.arange(out_len, device=latents.device, dtype=latents.dtype) + 0.5
        weights = torch.sigmoid((coords[None, None, :] - start[:, :, None]) / self.pack_temperature) - torch.sigmoid(
            (coords[None, None, :] - end[:, :, None]) / self.pack_temperature
        )
        weights = weights.clamp_min(0.0)
        weights_norm = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        soft_dna = torch.einsum("bjl,bjc->blc", weights_norm, base_probs)

        return {
            "base_logits_segmented": base_logits_segmented,
            "base_probs_segmented": base_probs_segmented,
            "lengths": lengths,
            "keep_segmented": keep_segmented,
            "keep": keep,
            "total_len": keep.sum(dim=1),
            "weights": weights_norm,
            "soft_dna": soft_dna,
            "base_entropy": -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1).mean(),
            "pack_entropy": -(weights_norm.clamp_min(1e-8) * weights_norm.clamp_min(1e-8).log()).sum(dim=1).mean(),
            "pack_confidence": weights_norm.max(dim=1).values.mean(),
        }

    def forward(self, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        tokenized = self.tokenize(target, mask)
        decoded = self.decode(tokenized["latents"], tokenized["token_usage"], target.shape[1])
        return {**tokenized, **decoded}


def reconstruction_metrics(soft_dna: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    predicted = soft_dna.argmax(dim=-1)
    correct = predicted == target
    accuracy = (correct.float() * mask).sum() / mask.sum().clamp_min(1.0)
    exact = (correct | (mask <= 0)).all(dim=1).float().mean()
    return {"accuracy": float(accuracy.item()), "exact": float(exact.item())}


def loss_for_batch(
    *,
    model: AdaptiveTokenizerAutoencoder,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_lengths: torch.Tensor,
    length_weight: float,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
    latent_l2_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rendered = model(target, mask)
    soft_dna = rendered["soft_dna"].clamp_min(1e-8)
    token_nll = -soft_dna.gather(-1, target[..., None]).squeeze(-1).log()
    recon_loss = (token_nll * mask).sum() / mask.sum().clamp_min(1.0)
    length_loss = F.smooth_l1_loss(rendered["total_len"], target_lengths)
    token_count = rendered["token_usage"].sum(dim=1)
    token_cost = token_count.mean()
    token_sharp = (rendered["token_prob"] * (1.0 - rendered["token_prob"]) * mask).sum() / mask.sum().clamp_min(1.0)
    decoder_sharp = (rendered["keep"] * (1.0 - rendered["keep"])).mean()
    latent_l2 = rendered["latents"].pow(2).mean()
    loss = (
        recon_loss
        + length_weight * length_loss
        + token_cost_weight * token_cost
        + token_sharp_weight * token_sharp
        + decoder_sharp_weight * decoder_sharp
        + latent_l2_weight * latent_l2
    )
    return loss, {
        **rendered,
        "recon_loss": recon_loss.detach(),
        "length_loss": length_loss.detach(),
        "token_cost": token_cost.detach(),
        "token_sharp": token_sharp.detach(),
        "decoder_sharp": decoder_sharp.detach(),
        "latent_l2": latent_l2.detach(),
        "token_count": token_count.detach(),
    }


def pick_device(args: argparse.Namespace) -> torch.device:
    if args.mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --mps, but MPS is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def current_weight(step: int, steps: int, weight: float, warmup_frac: float) -> float:
    if weight <= 0:
        return 0.0
    if warmup_frac <= 0:
        return weight
    warmup_steps = int(steps * warmup_frac)
    if step < warmup_steps:
        return 0.0
    ramp_steps = max(1, steps - warmup_steps)
    return weight * min(1.0, float(step - warmup_steps + 1) / float(ramp_steps))


def parse_int_list(values: str) -> list[int]:
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def parse_str_list(values: str) -> list[str]:
    parsed = [value.strip() for value in values.split(",") if value.strip()]
    unknown = [value for value in parsed if value not in BLOCK_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown block type(s): {unknown}. Known: {sorted(BLOCK_BUILDERS)}")
    return parsed


def summarise_batch(rendered: dict[str, torch.Tensor], target_lengths: torch.Tensor) -> str:
    token_count = rendered["token_count"]
    token_density = token_count / target_lengths.clamp_min(1.0)
    lengths = rendered["lengths"]
    active = (lengths > 1.0).float().sum(dim=1)
    return (
        f"target_len {target_lengths.mean().item():.1f}/{target_lengths.std(unbiased=False).item():.1f} "
        f"range {target_lengths.min().item():.0f}-{target_lengths.max().item():.0f} | "
        f"out {rendered['total_len'].mean().item():.1f}/{rendered['total_len'].std(unbiased=False).item():.1f} "
        f"range {rendered['total_len'].min().item():.1f}-{rendered['total_len'].max().item():.1f} | "
        f"tokens {token_count.mean().item():.2f}/{token_count.std(unbiased=False).item():.2f} "
        f"range {token_count.min().item():.1f}-{token_count.max().item():.1f} | "
        f"tok_per_base {token_density.mean().item():.3f} | "
        f"active_emit {active.mean().item():.1f}/{active.std(unbiased=False).item():.1f}"
    )


def compact_status(
    *,
    step: int,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
    train_loss: torch.Tensor,
    train_rendered: dict[str, torch.Tensor],
    train_target: torch.Tensor,
    train_mask: torch.Tensor,
    train_lengths: torch.Tensor,
    val_loss: float,
    val_rendered: dict[str, torch.Tensor],
    val_target: torch.Tensor,
    val_mask: torch.Tensor,
    val_lengths: torch.Tensor,
) -> list[str]:
    train_metrics = reconstruction_metrics(train_rendered["soft_dna"], train_target, train_mask)
    val_metrics = reconstruction_metrics(val_rendered["soft_dna"], val_target, val_mask)
    train_tokens = train_rendered["token_count"]
    val_tokens = val_rendered["token_count"]
    return [
        (
            f"\nstep {step:06d} | "
            f"val loss {val_loss:.4f} acc {val_metrics['accuracy']:.3f} exact {val_metrics['exact']:.3f} | "
            f"token_w {token_cost_weight:.2e} sharp_w {token_sharp_weight:.2e}/{decoder_sharp_weight:.2e}"
        ),
        (
            f"train loss {train_loss.item():.4f} ce {train_rendered['recon_loss'].item():.4f} "
            f"acc {train_metrics['accuracy']:.3f} exact {train_metrics['exact']:.3f} "
            f"len_loss {train_rendered['length_loss'].item():.3f} "
            f"tokens {train_tokens.mean().item():.2f}/{train_tokens.std(unbiased=False).item():.2f} "
            f"target_len {train_lengths.mean().item():.1f}/{train_lengths.std(unbiased=False).item():.1f} "
            f"out {train_rendered['total_len'].mean().item():.1f}/{train_rendered['total_len'].std(unbiased=False).item():.1f}"
        ),
        (
            f"val   ce {val_rendered['recon_loss'].item():.4f} "
            f"len_loss {val_rendered['length_loss'].item():.3f} "
            f"tokens {val_tokens.mean().item():.2f}/{val_tokens.std(unbiased=False).item():.2f} "
            f"range {val_tokens.min().item():.1f}-{val_tokens.max().item():.1f} "
            f"tok/base {(val_tokens / val_lengths.clamp_min(1.0)).mean().item():.3f} "
            f"target_len {val_lengths.mean().item():.1f}/{val_lengths.std(unbiased=False).item():.1f} "
            f"out {val_rendered['total_len'].mean().item():.1f}/{val_rendered['total_len'].std(unbiased=False).item():.1f}"
        ),
    ]


def format_metrics(prefix: str, loss: torch.Tensor, rendered: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor) -> str:
    metrics = reconstruction_metrics(rendered["soft_dna"], target, mask)
    return (
        f"{prefix:<5} loss {loss.item():.4f} ce {rendered['recon_loss'].item():.4f} "
        f"acc {metrics['accuracy']:.3f} exact {metrics['exact']:.3f} "
        f"len {rendered['length_loss'].item():.3f} "
        f"token_cost {rendered['token_cost'].item():.3f} "
        f"tok_sharp {rendered['token_sharp'].item():.3f} "
        f"dec_sharp {rendered['decoder_sharp'].item():.3f} "
        f"base_ent {rendered['base_entropy'].item():.3f} "
        f"pack_ent {rendered['pack_entropy'].item():.3f} "
        f"pack_conf {rendered['pack_confidence'].item():.3f}"
    )


def mean_for_indices(values: torch.Tensor, indices: list[int], device: torch.device) -> float:
    if not indices:
        return float("nan")
    index_tensor = torch.tensor(indices, device=device)
    return float(values[index_tensor].mean().item())


def fmt_mean(values: torch.Tensor, indices: list[int], device: torch.device, precision: int = 2) -> str:
    if not indices:
        return "na"
    return f"{mean_for_indices(values, indices, device):.{precision}f}"


def evaluate(
    *,
    model: AdaptiveTokenizerAutoencoder,
    rng: random.Random,
    device: torch.device,
    args: argparse.Namespace,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
) -> tuple[float, dict[str, float], dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    total_loss = 0.0
    total_acc = 0.0
    total_exact = 0.0
    last_rendered: dict[str, torch.Tensor] | None = None
    last_target: torch.Tensor | None = None
    last_mask: torch.Tensor | None = None
    last_lengths: torch.Tensor | None = None
    for _ in range(args.val_batches):
        target, mask, lengths, _ = make_batch(
            batch_size=args.batch_size,
            max_seq_len=args.seq_len,
            min_seq_len=args.variable_min_len,
            short_max_len_frac=args.short_max_len_frac,
            long_min_len_frac=args.long_min_len_frac,
            rng=rng,
            device=device,
        )
        loss, rendered = loss_for_batch(
            model=model,
            target=target,
            mask=mask,
            target_lengths=lengths,
            length_weight=args.length_weight,
            token_cost_weight=token_cost_weight,
            token_sharp_weight=token_sharp_weight,
            decoder_sharp_weight=decoder_sharp_weight,
            latent_l2_weight=args.latent_l2_weight,
        )
        metrics = reconstruction_metrics(rendered["soft_dna"], target, mask)
        total_loss += loss.item()
        total_acc += metrics["accuracy"]
        total_exact += metrics["exact"]
        last_rendered = rendered
        last_target = target
        last_mask = mask
        last_lengths = lengths
    assert last_rendered is not None and last_target is not None and last_mask is not None and last_lengths is not None
    return (
        total_loss / float(args.val_batches),
        {"accuracy": total_acc / float(args.val_batches), "exact": total_exact / float(args.val_batches)},
        last_rendered,
        last_target,
        last_mask,
        last_lengths,
    )


def controlled_probe(
    *,
    model: AdaptiveTokenizerAutoencoder,
    rng: random.Random,
    device: torch.device,
    args: argparse.Namespace,
) -> list[str]:
    if args.probe_examples <= 0:
        return []
    seq_lengths = parse_int_list(args.probe_seq_lengths)
    block_lengths = parse_int_list(args.probe_block_lengths)
    block_types = parse_str_list(args.probe_block_types)
    target, mask, lengths, labels = make_probe_batch(
        max_seq_len=args.seq_len,
        seq_lengths=[min(args.seq_len, value) for value in seq_lengths],
        block_types=block_types,
        block_lengths=block_lengths,
        examples_per_condition=args.probe_examples,
        rng=rng,
        device=device,
    )
    with torch.no_grad():
        rendered = model(target, mask)
    predicted = rendered["soft_dna"].argmax(dim=-1)
    correct = predicted == target
    per_acc = (correct.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    per_exact = (correct | (mask <= 0)).all(dim=1).float()
    token_count = rendered["token_usage"].sum(dim=1)
    active_emit = (rendered["lengths"] > 1.0).float().sum(dim=1)

    label_to_indices: dict[tuple[str, int, int], list[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    if not args.verbose_probe:
        lines = [
            "probe summary (L=target length; tok=latent tokens; base=base accuracy; exact=whole-seq exact; out=rendered length):"
        ]
        for seq_len in seq_lengths:
            clipped_seq_len = min(args.seq_len, seq_len)
            parts = []
            for block_type in block_types:
                indices = [
                    index
                    for key, key_indices in label_to_indices.items()
                    if key[0] == block_type and key[1] == clipped_seq_len
                    for index in key_indices
                ]
                short_name = {
                    "random": "rand",
                    "gc": "gc",
                    "at": "at",
                    "pyrimidine": "pyr",
                    "purine": "pur",
                    "homopolymer": "hpoly",
                    "dinucleotide": "di",
                    "motif": "motif",
                }.get(block_type, block_type)
                parts.append(
                    f"{short_name} tok {fmt_mean(token_count, indices, device)} "
                    f"base {fmt_mean(per_acc, indices, device, 3)} "
                    f"exact {fmt_mean(per_exact, indices, device, 3)} "
                    f"out {fmt_mean(rendered['total_len'], indices, device, 1)}"
                )
            lines.append(f"probe L{clipped_seq_len:03d} | " + " | ".join(parts))
        return lines

    lines = ["probe detail:"]
    for block_type in block_types:
        for seq_len in seq_lengths:
            parts = []
            for block_len in block_lengths:
                key = (block_type, min(args.seq_len, seq_len), block_len)
                indices = label_to_indices.get(key, [])
                if not indices:
                    continue
                index_tensor = torch.tensor(indices, device=device)
                first_index = indices[0]
                token_probs = rendered["token_prob"][first_index, : min(args.seq_len, seq_len)].detach().cpu()
                lengths_text = ",".join(f"{value:.1f}" for value in rendered["lengths"][first_index].detach().cpu().tolist())
                top_token_prob = token_probs.topk(min(8, token_probs.numel())).values.mean().item()
                parts.append(
                    f"b{block_len:02d} acc{per_acc[index_tensor].mean().item():.2f} "
                    f"ex{per_exact[index_tensor].mean().item():.2f} "
                    f"out{rendered['total_len'][index_tensor].mean().item():.1f} "
                    f"tok{token_count[index_tensor].mean().item():.1f} "
                    f"act{active_emit[index_tensor].mean().item():.1f} "
                    f"top_p{top_token_prob:.2f} "
                    f"lens[{lengths_text}]"
                )
            if parts:
                lines.append(f"probe {block_type:<12} L{min(args.seq_len, seq_len):03d} " + " | ".join(parts))
    return lines


def run_diagnostics(model: AdaptiveTokenizerAutoencoder, rng: random.Random, device: torch.device, args: argparse.Namespace) -> None:
    model.eval()
    with torch.no_grad():
        target, mask, lengths, kinds = make_batch(
            batch_size=1,
            max_seq_len=args.seq_len,
            min_seq_len=args.variable_min_len,
            short_max_len_frac=args.short_max_len_frac,
            long_min_len_frac=args.long_min_len_frac,
            rng=rng,
            device=device,
        )
        rendered = model(target, mask)
        seq_len = int(lengths[0].item())
        decoded = rendered["soft_dna"].argmax(dim=-1)[0, :seq_len]
        print("\nDiagnostics:")
        print("kind:", kinds[0], "length:", seq_len)
        print("target:       ", tensor_to_sequence(target[0, :seq_len]))
        print("reconstructed:", tensor_to_sequence(decoded))
        print("token count:", f"{rendered['token_usage'].sum(dim=1)[0].item():.2f}")
        print("token probs:", ", ".join(f"{value:.2f}" for value in rendered["token_prob"][0, :seq_len].detach().cpu().tolist()[:80]))
        print("emit lengths:", ", ".join(f"{value:.1f}" for value in rendered["lengths"][0].detach().cpu().tolist()))
    model.train()


def build_model(args: argparse.Namespace) -> AdaptiveTokenizerAutoencoder:
    return AdaptiveTokenizerAutoencoder(
        max_seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        latent_dim=args.latent_dim,
        max_slots_per_token=args.max_slots_per_token,
        hidden_dim=args.encoder_hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        slot_dim=args.slot_dim,
        token_rank_temperature=args.token_rank_temperature,
        token_usage_temperature=args.token_usage_temperature,
        gate_temperature=args.gate_temperature,
        pack_temperature=args.pack_temperature,
        initial_token_stride=args.initial_token_stride,
    )


def train(args: argparse.Namespace) -> None:
    if args.max_tokens * args.max_slots_per_token < args.seq_len:
        raise ValueError("--max-tokens * --max-slots-per-token must be at least --seq-len.")
    if args.variable_min_len <= 0 or args.variable_min_len > args.seq_len:
        raise ValueError("--variable-min-len must be in [1, seq_len].")
    if args.batch_size <= 0 or args.val_batches <= 0:
        raise ValueError("--batch-size and --val-batches must be positive.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args)
    train_rng = random.Random(args.seed)
    val_rng = random.Random(args.seed + 1_000_000)
    probe_rng = random.Random(args.seed + 2_000_000)
    diagnostic_rng = random.Random(args.seed + 3_000_000)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")

    print(f"device: {device}")
    print(
        f"adaptive tokenizer; seq_len={args.seq_len}; max_tokens={args.max_tokens}; "
        f"latent_dim={args.latent_dim}; params={sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(
        f"token_cost={args.token_cost_weight}; token_sharp={args.token_sharp_weight}; "
        f"length_weight={args.length_weight}; initial_stride={args.initial_token_stride}"
    )

    for step in range(args.steps):
        target, mask, lengths, _ = make_batch(
            batch_size=args.batch_size,
            max_seq_len=args.seq_len,
            min_seq_len=args.variable_min_len,
            short_max_len_frac=args.short_max_len_frac,
            long_min_len_frac=args.long_min_len_frac,
            rng=train_rng,
            device=device,
        )
        token_cost_weight = current_weight(step, args.steps, args.token_cost_weight, args.token_cost_warmup_frac)
        token_sharp_weight = current_weight(step, args.steps, args.token_sharp_weight, args.token_sharp_warmup_frac)
        decoder_sharp_weight = current_weight(step, args.steps, args.decoder_sharp_weight, args.decoder_sharp_warmup_frac)
        loss, rendered = loss_for_batch(
            model=model,
            target=target,
            mask=mask,
            target_lengths=lengths,
            length_weight=args.length_weight,
            token_cost_weight=token_cost_weight,
            token_sharp_weight=token_sharp_weight,
            decoder_sharp_weight=decoder_sharp_weight,
            latent_l2_weight=args.latent_l2_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_rendered, val_target, val_mask, val_lengths = evaluate(
                    model=model,
                    rng=val_rng,
                    device=device,
                    args=args,
                    token_cost_weight=token_cost_weight,
                    token_sharp_weight=token_sharp_weight,
                    decoder_sharp_weight=decoder_sharp_weight,
                )
                probe_lines = controlled_probe(model=model, rng=probe_rng, device=device, args=args)
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

            for line in compact_status(
                step=step,
                token_cost_weight=token_cost_weight,
                token_sharp_weight=token_sharp_weight,
                decoder_sharp_weight=decoder_sharp_weight,
                train_loss=loss,
                train_rendered=rendered,
                train_target=target,
                train_mask=mask,
                train_lengths=lengths,
                val_loss=val_loss,
                val_rendered=val_rendered,
                val_target=val_target,
                val_mask=val_mask,
                val_lengths=val_lengths,
            ):
                print(line)
            if args.verbose_metrics:
                print(format_metrics("train", loss, rendered, target, mask))
                print(format_metrics("val", torch.tensor(val_loss), val_rendered, val_target, val_mask))
                print(summarise_batch(val_rendered, val_lengths))
            for line in probe_lines:
                print(line)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": args.steps - 1,
            "validation_loss": best_val_loss,
        },
        checkpoint_dir / "latest.pt",
    )
    run_diagnostics(model, diagnostic_rng, device, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive-token DNA SoftPack autoencoder.")
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--variable-min-len", type=int, default=20)
    parser.add_argument("--short-max-len-frac", type=float, default=0.4)
    parser.add_argument("--long-min-len-frac", type=float, default=0.75)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--max-slots-per-token", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--length-weight", type=float, default=0.2)
    parser.add_argument("--token-cost-weight", type=float, default=0.01)
    parser.add_argument("--token-cost-warmup-frac", type=float, default=0.1)
    parser.add_argument("--token-sharp-weight", type=float, default=0.001)
    parser.add_argument("--token-sharp-warmup-frac", type=float, default=0.1)
    parser.add_argument("--decoder-sharp-weight", type=float, default=0.001)
    parser.add_argument("--decoder-sharp-warmup-frac", type=float, default=0.1)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)
    parser.add_argument("--token-rank-temperature", type=float, default=0.35)
    parser.add_argument("--token-usage-temperature", type=float, default=0.5)
    parser.add_argument("--gate-temperature", type=float, default=0.2)
    parser.add_argument("--pack-temperature", type=float, default=0.1)
    parser.add_argument("--initial-token-stride", type=float, default=8.0)
    parser.add_argument("--encoder-hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-hidden-dim", type=int, default=96)
    parser.add_argument("--slot-dim", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--probe-examples", type=int, default=4)
    parser.add_argument("--probe-seq-lengths", default="20,40,75,100")
    parser.add_argument("--probe-block-lengths", default="4,16,48")
    parser.add_argument("--probe-block-types", default="random,gc,at,homopolymer,dinucleotide,motif")
    parser.add_argument("--verbose-probe", action="store_true")
    parser.add_argument("--verbose-metrics", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/dna_adaptive_tokenizer_autoencoder")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
