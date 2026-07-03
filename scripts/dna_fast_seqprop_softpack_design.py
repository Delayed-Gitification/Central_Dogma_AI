from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DNA_BASES = "ACGT"
DNA_TO_INDEX = {base: index for index, base in enumerate(DNA_BASES)}


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def clean_dna(sequence: str) -> str:
    cleaned = "".join(base for base in sequence.upper() if base in DNA_TO_INDEX)
    if not cleaned:
        raise ValueError("DNA sequence is empty after filtering to A/C/G/T.")
    return cleaned


def dna_to_tensor(sequence: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([DNA_TO_INDEX[base] for base in clean_dna(sequence)], dtype=torch.long, device=device)


def tensor_to_dna(indices: torch.Tensor) -> str:
    return "".join(DNA_BASES[int(index)] for index in indices.detach().cpu().tolist())


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested --device mps, but PyTorch MPS is not available.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return torch.device(requested)


def straight_through_one_hot(probs: torch.Tensor, *, sample: bool) -> torch.Tensor:
    if sample:
        flat = probs.reshape(-1, probs.shape[-1])
        sampled = torch.multinomial(flat.clamp_min(1e-8), num_samples=1).squeeze(-1)
        hard = F.one_hot(sampled, num_classes=probs.shape[-1]).to(dtype=probs.dtype).reshape_as(probs)
    else:
        hard = F.one_hot(probs.argmax(dim=-1), num_classes=probs.shape[-1]).to(dtype=probs.dtype)
    return hard + probs - probs.detach()


def straight_through_bernoulli(probs: torch.Tensor, *, threshold: float) -> torch.Tensor:
    hard = (probs >= threshold).to(dtype=probs.dtype)
    return hard + probs - probs.detach()


class FastSeqPropSoftPackDesigner(nn.Module):
    def __init__(
        self,
        *,
        batch_size: int,
        max_slots: int,
        expected_length: float,
        base_init_std: float,
        presence_init_mode: str,
        presence_present_init: float,
        presence_absent_init: float,
        device: torch.device,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_slots = max_slots
        self.raw_base_logits = nn.Parameter(torch.randn(batch_size, max_slots, 4, device=device) * base_init_std)
        if presence_init_mode == "uniform":
            expected_fraction = min(0.98, max(0.02, expected_length / float(max_slots)))
            presence_bias = math.log(expected_fraction / (1.0 - expected_fraction))
            presence_logits = torch.full((batch_size, max_slots), presence_bias, device=device)
        elif presence_init_mode == "even":
            present_count = min(max_slots, max(1, int(round(expected_length))))
            presence_logits = torch.full((batch_size, max_slots), presence_absent_init, device=device)
            if present_count == 1:
                selected = torch.tensor([0], device=device)
            else:
                selected = torch.linspace(0, max_slots - 1, present_count, device=device).round().long().unique()
            presence_logits[:, selected] = presence_present_init
        else:
            raise ValueError(f"Unknown presence_init_mode: {presence_init_mode}")
        self.raw_presence_logits = nn.Parameter(presence_logits + torch.randn(batch_size, max_slots, device=device) * 0.01)
        self.gamma = nn.Parameter(torch.ones(4, device=device))
        self.beta = nn.Parameter(torch.zeros(4, device=device))
        self.log_pack_sharpness = nn.Parameter(torch.tensor(math.log(4.0), device=device))

    def normalised_base_logits(self) -> torch.Tensor:
        logits = self.raw_base_logits
        mean = logits.mean(dim=1, keepdim=True)
        std = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        logits = (logits - mean) / std
        return logits * self.gamma[None, None, :] + self.beta[None, None, :]

    def forward(
        self,
        *,
        output_length: int,
        base_temperature: float,
        presence_temperature: float,
        hard_bases: bool,
        sample_bases: bool,
        hard_presence: bool,
        presence_threshold: float,
        max_pack_sharpness: float,
    ) -> dict[str, torch.Tensor]:
        base_logits = self.normalised_base_logits() / max(base_temperature, 1e-6)
        base_probs = base_logits.softmax(dim=-1)
        if hard_bases:
            bases_for_pack = straight_through_one_hot(base_probs, sample=sample_bases)
        else:
            bases_for_pack = base_probs

        presence_prob = torch.sigmoid(self.raw_presence_logits / max(presence_temperature, 1e-6))
        if hard_presence:
            presence_for_pack = straight_through_bernoulli(presence_prob, threshold=presence_threshold)
        else:
            presence_for_pack = presence_prob

        soft_rank = torch.cumsum(presence_for_pack, dim=1)
        target_coordinate = torch.arange(output_length, device=base_probs.device, dtype=base_probs.dtype) + 1.0
        pack_sharpness = F.softplus(self.log_pack_sharpness).clamp(max=max_pack_sharpness)
        pack_logits = -pack_sharpness * (soft_rank[:, None, :] - target_coordinate[None, :, None]).pow(2)
        pack_logits = pack_logits + torch.log(presence_for_pack.clamp_min(1e-6))[:, None, :]
        pack = pack_logits.softmax(dim=-1)
        soft_dna = torch.einsum("bol,blc->boc", pack, bases_for_pack)

        relaxed_rank = torch.cumsum(presence_prob, dim=1)
        relaxed_pack_logits = -pack_sharpness * (relaxed_rank[:, None, :] - target_coordinate[None, :, None]).pow(2)
        relaxed_pack_logits = relaxed_pack_logits + torch.log(presence_prob.clamp_min(1e-6))[:, None, :]
        relaxed_pack = relaxed_pack_logits.softmax(dim=-1)
        soft_dna_relaxed = torch.einsum("bol,blc->boc", relaxed_pack, base_probs)

        base_entropy = -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        presence_entropy = -(
            presence_prob.clamp_min(1e-8) * presence_prob.clamp_min(1e-8).log()
            + (1.0 - presence_prob).clamp_min(1e-8) * (1.0 - presence_prob).clamp_min(1e-8).log()
        ).mean()
        pack_entropy = -(pack.clamp_min(1e-8) * pack.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return {
            "base_logits": base_logits,
            "base_probs": base_probs,
            "bases_for_pack": bases_for_pack,
            "presence_prob": presence_prob,
            "presence_for_pack": presence_for_pack,
            "soft_rank": soft_rank,
            "pack_logits": pack_logits,
            "pack": pack,
            "soft_dna": soft_dna,
            "soft_dna_relaxed": soft_dna_relaxed,
            "relaxed_pack": relaxed_pack,
            "pack_sharpness": pack_sharpness.detach(),
            "base_entropy": base_entropy,
            "presence_entropy": presence_entropy,
            "pack_entropy": pack_entropy,
            "pack_confidence": pack.max(dim=-1).values.mean().detach(),
            "presence_sum": presence_for_pack.sum(dim=1).detach(),
            "presence_prob_sum": presence_prob.sum(dim=1).detach(),
            "presence_min": presence_prob.min().detach(),
            "presence_max": presence_prob.max().detach(),
        }


def reconstruction_loss(
    rendered: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    target_lengths: torch.Tensor,
    length_weight: float,
    presence_cost_weight: float,
    base_entropy_weight: float,
    presence_entropy_weight: float,
    pack_entropy_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    soft_dna = rendered["soft_dna_relaxed"].clamp_min(1e-8)
    token_nll = -soft_dna.gather(-1, target[..., None]).squeeze(-1).log()
    nt_loss = token_nll.mean()
    length_loss = F.smooth_l1_loss(rendered["presence_for_pack"].sum(dim=1), target_lengths)
    presence_cost = rendered["presence_prob"].mean()
    loss = (
        nt_loss
        + length_weight * length_loss
        + presence_cost_weight * presence_cost
        + base_entropy_weight * rendered["base_entropy"]
        + presence_entropy_weight * rendered["presence_entropy"]
        + pack_entropy_weight * rendered["pack_entropy"]
    )
    return loss, {
        "nt_loss": nt_loss.detach(),
        "length_loss": length_loss.detach(),
        "presence_cost": presence_cost.detach(),
    }


def motif_loss(
    rendered: dict[str, torch.Tensor],
    motif: torch.Tensor,
    *,
    target_length: float,
    length_weight: float,
    presence_cost_weight: float,
    base_entropy_weight: float,
    presence_entropy_weight: float,
    pack_entropy_weight: float,
    motif_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    soft_dna = rendered["soft_dna_relaxed"].clamp_min(1e-8)
    log_probs = soft_dna.log()
    motif_length = motif.numel()
    output_length = soft_dna.shape[1]
    if motif_length > output_length:
        raise ValueError("--motif cannot be longer than --output-len.")
    scores = []
    for start in range(output_length - motif_length + 1):
        rows = [log_probs[:, start + offset, motif[offset]] for offset in range(motif_length)]
        scores.append(torch.stack(rows, dim=0).sum(dim=0))
    score_matrix = torch.stack(scores, dim=1)
    motif_score = motif_temperature * torch.logsumexp(score_matrix / max(motif_temperature, 1e-6), dim=1)
    length_target = torch.full_like(rendered["presence_for_pack"].sum(dim=1), float(target_length))
    length_loss = F.smooth_l1_loss(rendered["presence_for_pack"].sum(dim=1), length_target)
    presence_cost = rendered["presence_prob"].mean()
    loss = (
        -motif_score.mean()
        + length_weight * length_loss
        + presence_cost_weight * presence_cost
        + base_entropy_weight * rendered["base_entropy"]
        + presence_entropy_weight * rendered["presence_entropy"]
        + pack_entropy_weight * rendered["pack_entropy"]
    )
    return loss, {
        "motif_score": motif_score.detach().mean(),
        "motif_best_start": score_matrix.detach().argmax(dim=1).float().mean(),
        "length_loss": length_loss.detach(),
        "presence_cost": presence_cost.detach(),
    }


def reconstruction_metrics(soft_dna: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = soft_dna.argmax(dim=-1)
    correct = predicted == target
    return {
        "nt_acc": float(correct.float().mean().item()),
        "exact": float(correct.all(dim=1).float().mean().item()),
    }


def make_reconstruction_targets(args: argparse.Namespace, rng: random.Random, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    if args.target_dna:
        target = clean_dna(args.target_dna)
        sequences = [target for _ in range(args.batch_size)]
    else:
        sequences = [random_dna(args.target_len, rng) for _ in range(args.batch_size)]
    target_tensor = torch.stack([dna_to_tensor(sequence, device) for sequence in sequences], dim=0)
    return target_tensor, sequences


def decode_batch(rendered: dict[str, torch.Tensor], output_length: int) -> list[str]:
    predicted = rendered["soft_dna"].argmax(dim=-1)
    return [tensor_to_dna(predicted[row, :output_length]) for row in range(predicted.shape[0])]


def format_presence(values: torch.Tensor, limit: int = 80) -> str:
    shown = values.detach().cpu().tolist()[:limit]
    return ", ".join(f"{value:.2f}" for value in shown)


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    if args.mode == "reconstruct":
        target, sequences = make_reconstruction_targets(args, rng, device)
        output_length = target.shape[1]
        target_lengths = torch.full((args.batch_size,), float(output_length), device=device)
        expected_length = float(output_length)
    else:
        motif = dna_to_tensor(args.motif, device)
        output_length = args.output_len
        target = None
        sequences = []
        target_lengths = torch.full((args.batch_size,), float(args.target_len), device=device)
        expected_length = float(args.target_len)

    if args.max_slots < output_length:
        raise ValueError("--max-slots should be at least the output/target length for this first benchmark.")

    designer = FastSeqPropSoftPackDesigner(
        batch_size=args.batch_size,
        max_slots=args.max_slots,
        expected_length=expected_length,
        base_init_std=args.base_init_std,
        presence_init_mode=args.presence_init_mode,
        presence_present_init=args.presence_present_init,
        presence_absent_init=args.presence_absent_init,
        device=device,
    )
    optimizer = torch.optim.AdamW(designer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    final_rendered: dict[str, torch.Tensor] | None = None
    final_metrics: dict[str, float] = {}

    print("Fast SeqProp + existence gates + SoftPack design")
    print(f"device: {device}")
    print(
        f"mode={args.mode}; batch={args.batch_size}; max_slots={args.max_slots}; "
        f"output_len={output_length}; expected_len={expected_length:.1f}"
    )
    print(
        f"hard_bases={args.hard_bases}; sample_bases={args.sample_bases}; "
        f"hard_presence={args.hard_presence}; lr={args.lr}"
    )
    if args.mode == "reconstruct":
        print("target[0]:", sequences[0])
    else:
        print("motif:", clean_dna(args.motif))

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        rendered = designer(
            output_length=output_length,
            base_temperature=args.base_temperature,
            presence_temperature=args.presence_temperature,
            hard_bases=args.hard_bases,
            sample_bases=args.sample_bases,
            hard_presence=args.hard_presence,
            presence_threshold=args.presence_threshold,
            max_pack_sharpness=args.max_pack_sharpness,
        )
        if args.mode == "reconstruct":
            assert target is not None
            loss, loss_metrics = reconstruction_loss(
                rendered,
                target,
                target_lengths=target_lengths,
                length_weight=args.length_weight,
                presence_cost_weight=args.presence_cost_weight,
                base_entropy_weight=args.base_entropy_weight,
                presence_entropy_weight=args.presence_entropy_weight,
                pack_entropy_weight=args.pack_entropy_weight,
            )
            recon = reconstruction_metrics(rendered["soft_dna"], target)
            final_metrics = {**{key: float(value.item()) for key, value in loss_metrics.items()}, **recon}
        else:
            loss, loss_metrics = motif_loss(
                rendered,
                motif,
                target_length=float(args.target_len),
                length_weight=args.length_weight,
                presence_cost_weight=args.presence_cost_weight,
                base_entropy_weight=args.base_entropy_weight,
                presence_entropy_weight=args.presence_entropy_weight,
                pack_entropy_weight=args.pack_entropy_weight,
                motif_temperature=args.motif_temperature,
            )
            final_metrics = {key: float(value.item()) for key, value in loss_metrics.items()}

        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(designer.parameters(), args.grad_clip)
        optimizer.step()
        final_rendered = rendered

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            if args.save_best:
                torch.save(
                    {
                        "model_state_dict": designer.state_dict(),
                        "args": vars(args),
                        "step": step,
                        "loss": best_loss,
                    },
                    checkpoint_dir / "best.pt",
                )

        if step % args.print_every == 0 or step == args.steps - 1:
            decoded = decode_batch(rendered, output_length)
            metric_text = " ".join(f"{key} {value:.4f}" for key, value in final_metrics.items())
            print(
                f"\nstep {step:05d} loss {loss.item():.4f} best {best_loss:.4f} | {metric_text}"
            )
            print(
                f"presence sum {rendered['presence_sum'].mean().item():.2f}/"
                f"{rendered['presence_sum'].std(unbiased=False).item():.2f} "
                f"prob_sum {rendered['presence_prob_sum'].mean().item():.2f} "
                f"range {rendered['presence_min'].item():.3f}-{rendered['presence_max'].item():.3f} | "
                f"entropy base {rendered['base_entropy'].item():.3f} "
                f"presence {rendered['presence_entropy'].item():.3f} "
                f"pack {rendered['pack_entropy'].item():.3f} "
                f"pack_conf {rendered['pack_confidence'].item():.3f} "
                f"sharp {rendered['pack_sharpness'].item():.3f}"
            )
            if args.mode == "reconstruct":
                print("target: ", sequences[0])
            print("decoded:", decoded[0])

    assert final_rendered is not None
    decoded = decode_batch(final_rendered, output_length)
    print("\nFinal:")
    print("decoded[0]:", decoded[0])
    if args.mode == "reconstruct":
        print("target[0]: ", sequences[0])
    print("presence probs[0]:", format_presence(final_rendered["presence_prob"][0]))
    print("base argmax[0]:", tensor_to_dna(final_rendered["base_probs"][0].argmax(dim=-1)))
    torch.save(
        {
            "model_state_dict": designer.state_dict(),
            "args": vars(args),
            "step": args.steps - 1,
            "loss": best_loss,
        },
        checkpoint_dir / "latest.pt",
    )
    print(f"saved latest checkpoint: {checkpoint_dir / 'latest.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast SeqProp base logits plus existence gates and SoftPack.")
    parser.add_argument("--mode", choices=("reconstruct", "motif"), default="reconstruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-slots", type=int, default=160)
    parser.add_argument("--target-len", type=int, default=100)
    parser.add_argument("--output-len", type=int, default=100)
    parser.add_argument("--target-dna", default="")
    parser.add_argument("--motif", default="TATAAA")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-init-std", type=float, default=0.01)
    parser.add_argument("--presence-init-mode", choices=("even", "uniform"), default="even")
    parser.add_argument("--presence-present-init", type=float, default=4.0)
    parser.add_argument("--presence-absent-init", type=float, default=-4.0)
    parser.add_argument("--base-temperature", type=float, default=1.0)
    parser.add_argument("--presence-temperature", type=float, default=1.0)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--hard-bases", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-bases", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hard-presence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--length-weight", type=float, default=0.1)
    parser.add_argument("--presence-cost-weight", type=float, default=0.0)
    parser.add_argument("--base-entropy-weight", type=float, default=0.001)
    parser.add_argument("--presence-entropy-weight", type=float, default=0.001)
    parser.add_argument("--pack-entropy-weight", type=float, default=0.001)
    parser.add_argument("--motif-temperature", type=float, default=0.5)
    parser.add_argument("--max-pack-sharpness", type=float, default=80.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints/dna_fast_seqprop_softpack_design")
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
