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
    while sum(len(part) for part in parts) < seq_len:
        remaining = seq_len - sum(len(part) for part in parts)
        length = min(remaining, rng.randint(5, 32))
        parts.append(rng.choice(builders)(rng, length))
    return "".join(parts)[:seq_len]


def make_dataset(num_sequences: int, seq_len: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    return torch.stack([sequence_to_tensor(generate_segmental_sequence(seq_len, rng)) for _ in range(num_sequences)])


class SegmentalSoftpackAutodecoder(nn.Module):
    def __init__(
        self,
        *,
        num_train: int,
        seq_len: int,
        num_segments: int,
        latent_dim: int,
        max_slots_per_segment: int,
        gate_temperature: float,
        pack_temperature: float,
        hidden_dim: int = 64,
        slot_dim: int = 16,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.latent_dim = latent_dim
        self.max_slots_per_segment = max_slots_per_segment
        self.gate_temperature = gate_temperature
        self.pack_temperature = pack_temperature

        self.latent_table = nn.Embedding(num_train, num_segments * latent_dim)
        nn.init.normal_(self.latent_table.weight, mean=0.0, std=0.02)

        self.slot_embedding = nn.Embedding(max_slots_per_segment, slot_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + slot_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        self.length_head = nn.Linear(latent_dim, 1)
        initial_length = seq_len / float(num_segments)
        nn.init.zeros_(self.length_head.weight)
        nn.init.constant_(self.length_head.bias, logit(initial_length / float(max_slots_per_segment)))

    def latents_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.latent_table(indices).reshape(indices.shape[0], self.num_segments, self.latent_dim)

    def segment_outputs(
        self,
        z: torch.Tensor,
        length_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_segments, latent_dim = z.shape
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

        pack_entropy = -(weights_norm.clamp_min(1e-8) * weights_norm.clamp_min(1e-8).log()).sum(dim=1).mean()
        base_entropy = -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
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

    def forward(self, indices: torch.Tensor, out_len: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.latents_for_indices(indices)
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
    model: SegmentalSoftpackAutodecoder,
    indices: torch.Tensor,
    target: torch.Tensor,
    seq_len: int,
    length_weight: float,
    latent_l2_weight: float,
    sharp_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    z, rendered = model(indices, seq_len)
    soft_dna = rendered["soft_dna"].clamp_min(1e-8)
    recon_loss = -soft_dna.gather(-1, target[..., None]).squeeze(-1).log().mean()
    length_loss = F.smooth_l1_loss(rendered["total_len"], torch.full_like(rendered["total_len"], float(seq_len)))
    latent_l2 = z.pow(2).mean()
    sharp_loss = (rendered["keep"] * (1.0 - rendered["keep"])).mean()
    loss = recon_loss + length_weight * length_loss + latent_l2_weight * latent_l2 + sharp_weight * sharp_loss
    return loss, {
        **rendered,
        "recon_loss": recon_loss.detach(),
        "length_loss": length_loss.detach(),
        "latent_l2": latent_l2.detach(),
        "sharp_loss": sharp_loss.detach(),
    }


def hard_decode(
    model: SegmentalSoftpackAutodecoder,
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


def fit_latents_to_targets(
    model: SegmentalSoftpackAutodecoder,
    target_seq: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
) -> dict[str, float]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    z = torch.randn(
        target_seq.shape[0],
        model.num_segments,
        model.latent_dim,
        device=target_seq.device,
    ) * 0.02
    z.requires_grad_(True)
    optimizer = torch.optim.AdamW([z], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rendered = model.render_from_latents(z, target_seq.shape[1])
        soft_dna = rendered["soft_dna"].clamp_min(1e-8)
        recon_loss = -soft_dna.gather(-1, target_seq[..., None]).squeeze(-1).log().mean()
        length_loss = F.smooth_l1_loss(rendered["total_len"], torch.full_like(rendered["total_len"], float(target_seq.shape[1])))
        loss = recon_loss + 0.05 * length_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rendered = model.render_from_latents(z, target_seq.shape[1])
        soft_dna = rendered["soft_dna"].clamp_min(1e-8)
        recon_loss = -soft_dna.gather(-1, target_seq[..., None]).squeeze(-1).log().mean()
        metrics = reconstruction_metrics(soft_dna, target_seq)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {
        "ce": float(recon_loss.item()),
        "accuracy": metrics["accuracy"],
        "exact": metrics["exact"],
    }


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


def run_diagnostics(
    *,
    model: SegmentalSoftpackAutodecoder,
    train_data: torch.Tensor,
    device: torch.device,
    seq_len: int,
) -> None:
    model.eval()
    with torch.no_grad():
        index = torch.tensor([0], dtype=torch.long, device=device)
        z = model.latents_for_indices(index)
        original = hard_decode(model, z, out_len=seq_len)
        target = tensor_to_sequence(train_data[0])

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
    if args.num_train <= 0 or args.num_val <= 0:
        raise ValueError("--num-train and --num-val must be positive.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data = make_dataset(args.num_train, args.seq_len, args.seed).to(device)
    val_data = make_dataset(args.num_val, args.seq_len, args.seed + 1_000_000).to(device)

    model = SegmentalSoftpackAutodecoder(
        num_train=args.num_train,
        seq_len=args.seq_len,
        num_segments=args.num_segments,
        latent_dim=args.latent_dim,
        max_slots_per_segment=args.max_slots_per_segment,
        gate_temperature=args.gate_temperature,
        pack_temperature=args.pack_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = float("inf")

    param_count = sum(parameter.numel() for name, parameter in model.named_parameters() if "latent_table" not in name)
    print(f"device: {device}")
    print(f"train/val: {args.num_train}/{args.num_val}; seq_len={args.seq_len}")
    print(
        f"segments={args.num_segments}; latent_dim={args.latent_dim}; "
        f"max_slots_per_segment={args.max_slots_per_segment}; decoder_params={param_count}"
    )

    for step in range(args.steps):
        batch_indices = torch.randint(0, args.num_train, (args.batch_size,), device=device)
        batch = train_data[batch_indices]
        sharp_weight = current_sharp_weight(step, args.steps, args.sharp_weight, args.sharp_warmup_frac)
        loss, rendered = loss_for_batch(
            model=model,
            indices=batch_indices,
            target=batch,
            seq_len=args.seq_len,
            length_weight=args.length_weight,
            latent_l2_weight=args.latent_l2_weight,
            sharp_weight=sharp_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "step": step,
                    "loss": best_loss,
                },
                checkpoint_dir / "best.pt",
            )

        if step % args.print_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                metrics = reconstruction_metrics(rendered["soft_dna"], batch)
                base_entropy = rendered["base_entropy"].item()
                print(
                    f"\nstep {step:06d} loss {loss.item():.4f} "
                    f"ce {rendered['recon_loss'].item():.4f} "
                    f"acc {metrics['accuracy']:.3f} exact {metrics['exact']:.3f} "
                    f"len_loss {rendered['length_loss'].item():.3f} "
                    f"sharp_w {sharp_weight:.2e} sharp {rendered['sharp_loss'].item():.4f}"
                )
                print(summarise_lengths(rendered["lengths"], rendered["total_len"]))
                print(
                    f"base_entropy {base_entropy:.3f} "
                    f"pack_entropy {rendered['pack_entropy'].item():.3f} "
                    f"pack_conf {rendered['pack_confidence'].item():.3f}"
                )

    val_fit = fit_latents_to_targets(
        model,
        val_data[: min(args.num_val, args.val_fit_examples)],
        steps=args.val_fit_steps,
        lr=args.val_fit_lr,
    )
    print("\nHeld-out latent fitting:")
    print(f"ce {val_fit['ce']:.4f} acc {val_fit['accuracy']:.3f} exact {val_fit['exact']:.3f}")
    run_diagnostics(model=model, train_data=train_data, device=device, seq_len=args.seq_len)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmental SoftPack DNA auto-decoder.")
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--num-train", type=int, default=4096)
    parser.add_argument("--num-val", type=int, default=256)
    parser.add_argument("--num-segments", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--max-slots-per-segment", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--length-weight", type=float, default=0.05)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)
    parser.add_argument("--gate-temperature", type=float, default=0.5)
    parser.add_argument("--pack-temperature", type=float, default=0.5)
    parser.add_argument("--sharp-weight", type=float, default=1e-3)
    parser.add_argument("--sharp-warmup-frac", type=float, default=0.2)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="checkpoints/dna_segmental_softpack_autodecoder")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-fit-steps", type=int, default=300)
    parser.add_argument("--val-fit-lr", type=float, default=0.05)
    parser.add_argument("--val-fit-examples", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
