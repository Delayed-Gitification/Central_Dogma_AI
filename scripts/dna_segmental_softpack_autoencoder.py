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


def generate_segmental_sequence(seq_len: int, rng: random.Random) -> str:
    builders = [
        random_dna_block,
        homopolymer_block,
        dinucleotide_repeat_block,
        pyrimidine_rich_block,
        motif_repeat_block,
    ]
    parts = []
    total_length = 0
    while total_length < seq_len:
        remaining = seq_len - total_length
        length = min(remaining, rng.randint(5, 32))
        part = rng.choice(builders)(rng, length)
        parts.append(part)
        total_length += len(part)
    return "".join(parts)[:seq_len]


def make_batch(batch_size: int, seq_len: int, rng: random.Random, device: torch.device) -> torch.Tensor:
    batch = torch.stack([sequence_to_tensor(generate_segmental_sequence(seq_len, rng)) for _ in range(batch_size)])
    return batch.to(device)


class ConvSegmentEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_segments: int,
        latent_dim: int,
        hidden_dim: int,
        layers: int,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("--encoder-layers must be at least 1.")
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
        self.pool = nn.AdaptiveAvgPool1d(num_segments)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.segment_embedding = nn.Parameter(torch.zeros(num_segments, latent_dim))
        nn.init.normal_(self.segment_embedding, mean=0.0, std=0.02)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(target, num_classes=4).to(dtype=torch.float32)
        positions = torch.linspace(-1.0, 1.0, target.shape[1], device=target.device, dtype=one_hot.dtype)
        positions = positions[None, :, None].expand(target.shape[0], -1, -1)
        encoder_input = torch.cat([one_hot, positions], dim=-1)
        hidden = self.net(encoder_input.transpose(1, 2))
        pooled = self.pool(hidden).transpose(1, 2)
        return self.to_latent(pooled) + self.segment_embedding[None, :, :]


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
        decoder_hidden_dim: int = 64,
        slot_dim: int = 16,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.latent_dim = latent_dim
        self.max_slots_per_segment = max_slots_per_segment
        self.gate_temperature = gate_temperature
        self.pack_temperature = pack_temperature

        self.encoder = ConvSegmentEncoder(
            num_segments=num_segments,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
            layers=encoder_layers,
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
        initial_length = seq_len / float(num_segments)
        nn.init.zeros_(self.length_head.weight)
        nn.init.constant_(self.length_head.bias, logit(initial_length / float(max_slots_per_segment)))

    def encode(self, target: torch.Tensor) -> torch.Tensor:
        return self.encoder(target)

    def segment_outputs(
        self,
        z: torch.Tensor,
        length_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        lengths = self.max_slots_per_segment * torch.sigmoid(self.length_head(z).squeeze(-1))
        if length_delta is not None:
            lengths = (lengths + length_delta).clamp(0.0, float(self.max_slots_per_segment))
        slot_centres = slots.to(dtype=z.dtype) + 0.5
        keep = torch.sigmoid((lengths[..., None] - slot_centres[None, None, :]) / self.gate_temperature)
        return base_logits, base_probs, lengths, keep

    def render_from_latents(
        self,
        z: torch.Tensor,
        out_len: int,
        length_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        base_logits, base_probs_segmented, lengths, keep_segmented = self.segment_outputs(z, length_delta=length_delta)
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
    length_weight: float,
    latent_l2_weight: float,
    sharp_weight: float,
    active_segment_weight: float,
    active_segment_threshold: float,
    active_segment_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    z, rendered = model(target, seq_len)
    soft_dna = rendered["soft_dna"].clamp_min(1e-8)
    recon_loss = -soft_dna.gather(-1, target[..., None]).squeeze(-1).log().mean()
    length_loss = F.smooth_l1_loss(rendered["total_len"], torch.full_like(rendered["total_len"], float(seq_len)))
    latent_l2 = z.pow(2).mean()
    sharp_loss = (rendered["keep"] * (1.0 - rendered["keep"])).mean()
    active_segments = torch.sigmoid((rendered["lengths"] - active_segment_threshold) / active_segment_temperature)
    active_segment_loss = active_segments.mean()
    loss = (
        recon_loss
        + length_weight * length_loss
        + latent_l2_weight * latent_l2
        + sharp_weight * sharp_loss
        + active_segment_weight * active_segment_loss
    )
    return loss, {
        **rendered,
        "z": z.detach(),
        "recon_loss": recon_loss.detach(),
        "length_loss": length_loss.detach(),
        "latent_l2": latent_l2.detach(),
        "sharp_loss": sharp_loss.detach(),
        "active_segment_loss": active_segment_loss.detach(),
        "active_segments": active_segments.detach(),
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


def format_metrics(prefix: str, loss: torch.Tensor, rendered: dict[str, torch.Tensor], target: torch.Tensor) -> str:
    metrics = reconstruction_metrics(rendered["soft_dna"], target)
    return (
        f"{prefix:<5} loss {loss.item():.4f} ce {rendered['recon_loss'].item():.4f} "
        f"acc {metrics['accuracy']:.3f} exact {metrics['exact']:.3f} "
        f"len {rendered['length_loss'].item():.3f} "
        f"active_loss {rendered['active_segment_loss'].item():.3f} "
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
    length_weight: float,
    latent_l2_weight: float,
    sharp_weight: float,
    active_segment_weight: float,
    active_segment_threshold: float,
    active_segment_temperature: float,
) -> tuple[float, dict[str, float], dict[str, torch.Tensor], torch.Tensor]:
    total_loss = 0.0
    total_accuracy = 0.0
    total_exact = 0.0
    last_rendered: dict[str, torch.Tensor] | None = None
    last_batch: torch.Tensor | None = None
    for _ in range(batches):
        batch = make_batch(batch_size, seq_len, rng, device)
        loss, rendered = loss_for_batch(
            model=model,
            target=batch,
            seq_len=seq_len,
            length_weight=length_weight,
            latent_l2_weight=latent_l2_weight,
            sharp_weight=sharp_weight,
            active_segment_weight=active_segment_weight,
            active_segment_threshold=active_segment_threshold,
            active_segment_temperature=active_segment_temperature,
        )
        metrics = reconstruction_metrics(rendered["soft_dna"], batch)
        total_loss += loss.item()
        total_accuracy += metrics["accuracy"]
        total_exact += metrics["exact"]
        last_rendered = rendered
        last_batch = batch
    assert last_rendered is not None and last_batch is not None
    return (
        total_loss / float(batches),
        {
            "accuracy": total_accuracy / float(batches),
            "exact": total_exact / float(batches),
        },
        last_rendered,
        last_batch,
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

    print("\nDiagnostics:")
    print("target:       ", target)
    print("reconstructed:", original)
    print("perturbed:    ", perturbed)
    print("length +10:   ", length_modified)
    print("segment lengths:", ", ".join(f"{value:.1f}" for value in lengths.tolist()))
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
        decoder_hidden_dim=args.decoder_hidden_dim,
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

    for step in range(args.steps):
        batch = make_batch(args.batch_size, args.seq_len, train_rng, device)
        sharp_weight = current_sharp_weight(step, args.steps, args.sharp_weight, args.sharp_warmup_frac)
        loss, rendered = loss_for_batch(
            model=model,
            target=batch,
            seq_len=args.seq_len,
            length_weight=args.length_weight,
            latent_l2_weight=args.latent_l2_weight,
            sharp_weight=sharp_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_rendered, val_batch = evaluate_fresh_batches(
                    model=model,
                    rng=val_rng,
                    device=device,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    batches=args.val_batches,
                    length_weight=args.length_weight,
                    latent_l2_weight=args.latent_l2_weight,
                    sharp_weight=sharp_weight,
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
                f"\nstep {step:06d} sharp_w {sharp_weight:.2e} "
                f"val_loss {val_loss:.4f} val_acc {val_metrics['accuracy']:.3f} val_exact {val_metrics['exact']:.3f}"
            )
            print(format_metrics("train", loss, rendered, batch))
            print(format_metrics("val", torch.tensor(val_loss), val_rendered, val_batch))
            print(summarise_lengths(val_rendered["lengths"], val_rendered["total_len"]))

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
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=0.2)
    parser.add_argument("--pack-temperature", type=float, default=0.1)
    parser.add_argument("--sharp-weight", type=float, default=0.002)
    parser.add_argument("--sharp-warmup-frac", type=float, default=0.2)
    parser.add_argument("--encoder-hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=2)
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
