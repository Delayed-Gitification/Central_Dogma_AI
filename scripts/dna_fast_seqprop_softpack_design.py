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


def hard_concrete_gate(
    logits: torch.Tensor,
    *,
    temperature: float,
    gamma: float,
    zeta: float,
    sample: bool,
    hard: bool,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sample:
        uniform = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
        logistic_noise = uniform.log() - (1.0 - uniform).log()
        stretched = torch.sigmoid((logits + logistic_noise) / max(temperature, 1e-6))
    else:
        stretched = torch.sigmoid(logits / max(temperature, 1e-6))
    relaxed = (stretched * (zeta - gamma) + gamma).clamp(0.0, 1.0)
    if hard:
        relaxed = straight_through_bernoulli(relaxed, threshold=threshold)
    l0_prob = torch.sigmoid(logits - temperature * math.log(-gamma / zeta))
    return relaxed, l0_prob


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
        initial_base_logits: torch.Tensor | None = None,
        initial_presence_logits: torch.Tensor | None = None,
        device: torch.device,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_slots = max_slots
        if initial_base_logits is None:
            base_logits = torch.randn(batch_size, max_slots, 4, device=device) * base_init_std
        else:
            base_logits = initial_base_logits.to(device=device, dtype=torch.float32)
            if base_logits.shape != (batch_size, max_slots, 4):
                raise ValueError(f"initial_base_logits has shape {tuple(base_logits.shape)}, expected {(batch_size, max_slots, 4)}")
        self.raw_base_logits = nn.Parameter(base_logits)
        if initial_presence_logits is None:
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
        else:
            presence_logits = initial_presence_logits.to(device=device, dtype=torch.float32)
            if presence_logits.shape != (batch_size, max_slots):
                raise ValueError(f"initial_presence_logits has shape {tuple(presence_logits.shape)}, expected {(batch_size, max_slots)}")
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
        presence_gate: str,
        sample_presence: bool,
        presence_threshold: float,
        hard_concrete_gamma: float,
        hard_concrete_zeta: float,
        max_pack_sharpness: float,
    ) -> dict[str, torch.Tensor]:
        base_logits = self.normalised_base_logits() / max(base_temperature, 1e-6)
        base_probs = base_logits.softmax(dim=-1)
        if hard_bases:
            bases_for_pack = straight_through_one_hot(base_probs, sample=sample_bases)
        else:
            bases_for_pack = base_probs

        if presence_gate == "sigmoid":
            presence_prob = torch.sigmoid(self.raw_presence_logits / max(presence_temperature, 1e-6))
            presence_for_pack = presence_prob
            l0_prob = presence_prob
            if hard_presence:
                presence_for_pack = straight_through_bernoulli(presence_prob, threshold=presence_threshold)
        elif presence_gate == "hard_concrete":
            presence_for_pack, l0_prob = hard_concrete_gate(
                self.raw_presence_logits,
                temperature=presence_temperature,
                gamma=hard_concrete_gamma,
                zeta=hard_concrete_zeta,
                sample=sample_presence and self.training,
                hard=hard_presence,
                threshold=presence_threshold,
            )
            presence_prob = l0_prob
        else:
            raise ValueError(f"Unknown presence_gate: {presence_gate}")

        soft_rank = torch.cumsum(presence_for_pack, dim=1)
        target_coordinate = torch.arange(output_length, device=base_probs.device, dtype=base_probs.dtype) + 1.0
        pack_sharpness = F.softplus(self.log_pack_sharpness).clamp(max=max_pack_sharpness)
        pack_logits = -pack_sharpness * (soft_rank[:, None, :] - target_coordinate[None, :, None]).pow(2)
        pack_logits = pack_logits + torch.log(presence_for_pack.clamp_min(1e-6))[:, None, :]
        pack = pack_logits.softmax(dim=-1)
        soft_dna = torch.einsum("bol,blc->boc", pack, bases_for_pack)

        relaxed_presence = presence_prob if presence_gate == "sigmoid" else hard_concrete_gate(
            self.raw_presence_logits,
            temperature=presence_temperature,
            gamma=hard_concrete_gamma,
            zeta=hard_concrete_zeta,
            sample=False,
            hard=False,
            threshold=presence_threshold,
        )[0]
        relaxed_rank = torch.cumsum(relaxed_presence, dim=1)
        relaxed_pack_logits = -pack_sharpness * (relaxed_rank[:, None, :] - target_coordinate[None, :, None]).pow(2)
        relaxed_pack_logits = relaxed_pack_logits + torch.log(relaxed_presence.clamp_min(1e-6))[:, None, :]
        relaxed_pack = relaxed_pack_logits.softmax(dim=-1)
        soft_dna_relaxed = torch.einsum("bol,blc->boc", relaxed_pack, base_probs)

        base_entropy = -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        presence_entropy = -(
            l0_prob.clamp_min(1e-8) * l0_prob.clamp_min(1e-8).log()
            + (1.0 - l0_prob).clamp_min(1e-8) * (1.0 - l0_prob).clamp_min(1e-8).log()
        ).mean()
        pack_entropy = -(pack.clamp_min(1e-8) * pack.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return {
            "base_logits": base_logits,
            "base_probs": base_probs,
            "bases_for_pack": bases_for_pack,
            "presence_prob": presence_prob,
            "presence_l0_prob": l0_prob,
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
            "presence_prob_sum": l0_prob.sum(dim=1).detach(),
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
    presence_cost = rendered["presence_l0_prob"].mean()
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
    presence_cost = rendered["presence_l0_prob"].mean()
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
        "example0_acc": float(correct[0].float().mean().item()),
        "example0_exact": float(correct[0].all().float().item()),
        "exact_count": float(correct.all(dim=1).float().sum().item()),
    }


def make_reconstruction_targets(args: argparse.Namespace, rng: random.Random, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    if args.target_dna:
        target = clean_dna(args.target_dna)
        sequences = [target for _ in range(args.batch_size)]
    else:
        sequences = [random_dna(args.target_len, rng) for _ in range(args.batch_size)]
    target_tensor = torch.stack([dna_to_tensor(sequence, device) for sequence in sequences], dim=0)
    return target_tensor, sequences


def make_indel_pair(args: argparse.Namespace, rng: random.Random) -> tuple[str, str]:
    source = clean_dna(args.source_dna) if args.source_dna else random_dna(args.source_len, rng)
    delete_start = max(0, min(args.indel_delete_start, len(source)))
    delete_end = max(delete_start, min(delete_start + args.indel_delete_len, len(source)))
    target = source[:delete_start] + source[delete_end:]
    insert_sequence = clean_dna(args.indel_insert)
    insert_at = max(0, min(args.indel_insert_at, len(target)))
    target = target[:insert_at] + insert_sequence + target[insert_at:]
    return source, target


def make_source_canvas_initialisation(
    *,
    source: str,
    batch_size: int,
    max_slots: int,
    optional_slots_per_base: int,
    base_logit_init_strength: float,
    presence_present_init: float,
    presence_absent_init: float,
    base_init_std: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows_base = []
    rows_presence = []
    real_slot_mask = []
    real_slot_base = []
    for _ in range(batch_size):
        base_logits = []
        presence_logits = []
        slot_mask = []
        slot_base = []
        for base in source:
            real_logits = torch.randn(4, device=device) * base_init_std
            real_logits[DNA_TO_INDEX[base]] += base_logit_init_strength
            base_logits.append(real_logits)
            presence_logits.append(torch.tensor(presence_present_init, device=device))
            slot_mask.append(torch.tensor(True, device=device))
            slot_base.append(torch.tensor(DNA_TO_INDEX[base], device=device))
            for _slot in range(optional_slots_per_base):
                base_logits.append(torch.randn(4, device=device) * base_init_std)
                presence_logits.append(torch.tensor(presence_absent_init, device=device))
                slot_mask.append(torch.tensor(False, device=device))
                slot_base.append(torch.tensor(0, device=device))
        padding = max_slots - len(base_logits)
        if padding < 0:
            raise ValueError(
                f"Source canvas needs {len(base_logits)} slots, but --max-slots={max_slots}. "
                "Increase --max-slots or reduce --optional-slots-per-base."
            )
        for _pad in range(padding):
            base_logits.append(torch.randn(4, device=device) * base_init_std)
            presence_logits.append(torch.tensor(presence_absent_init, device=device))
            slot_mask.append(torch.tensor(False, device=device))
            slot_base.append(torch.tensor(0, device=device))
        rows_base.append(torch.stack(base_logits))
        rows_presence.append(torch.stack(presence_logits))
        real_slot_mask.append(torch.stack(slot_mask))
        real_slot_base.append(torch.stack(slot_base))
    return torch.stack(rows_base), torch.stack(rows_presence), torch.stack(real_slot_mask), torch.stack(real_slot_base)


def decode_batch(rendered: dict[str, torch.Tensor], output_length: int) -> list[str]:
    predicted = rendered["soft_dna"].argmax(dim=-1)
    return [tensor_to_dna(predicted[row, :output_length]) for row in range(predicted.shape[0])]


def format_presence(values: torch.Tensor, limit: int = 80) -> str:
    shown = values.detach().cpu().tolist()[:limit]
    return ", ".join(f"{value:.2f}" for value in shown)


def selected_slot_summary(presence: torch.Tensor, *, threshold: float = 0.5, limit: int = 120) -> str:
    selected = torch.nonzero(presence.detach().cpu() >= threshold, as_tuple=False).flatten().tolist()
    shown = selected[:limit]
    suffix = " ..." if len(selected) > limit else ""
    return f"{len(selected)} active; slots {shown}{suffix}"


def source_slot_labels(source: str, optional_slots_per_base: int, max_slots: int) -> list[str]:
    labels = []
    for index, base in enumerate(source):
        labels.append(f"src{index}:{base}")
        for optional_index in range(optional_slots_per_base):
            labels.append(f"opt{index}.{optional_index}")
    while len(labels) < max_slots:
        labels.append(f"pad{len(labels)}")
    return labels[:max_slots]


def active_slot_trace(
    *,
    presence: torch.Tensor,
    base_probs: torch.Tensor,
    labels: list[str],
    threshold: float = 0.5,
    limit: int = 120,
) -> str:
    selected = torch.nonzero(presence.detach().cpu() >= threshold, as_tuple=False).flatten().tolist()
    parts = []
    base_indices = base_probs.detach().cpu().argmax(dim=-1)
    for slot in selected[:limit]:
        label = labels[slot] if slot < len(labels) else f"slot{slot}"
        parts.append(f"{slot}:{label}->{DNA_BASES[int(base_indices[slot])]}")
    suffix = " ..." if len(selected) > limit else ""
    return " | ".join(parts) + suffix


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    source_sequence = ""
    initial_base_logits = None
    initial_presence_logits = None
    source_real_slot_mask = None
    source_real_slot_base = None
    slot_labels = [f"slot{index}" for index in range(args.max_slots)]

    if args.mode == "reconstruct":
        target, sequences = make_reconstruction_targets(args, rng, device)
        output_length = target.shape[1]
        target_lengths = torch.full((args.batch_size,), float(output_length), device=device)
        expected_length = float(output_length)
    elif args.mode == "indel":
        source_sequence, target_sequence = make_indel_pair(args, rng)
        sequences = [target_sequence for _ in range(args.batch_size)]
        target = torch.stack([dna_to_tensor(target_sequence, device) for _ in range(args.batch_size)], dim=0)
        output_length = target.shape[1]
        target_lengths = torch.full((args.batch_size,), float(output_length), device=device)
        expected_length = float(len(source_sequence))
        source_canvas_slots = len(source_sequence) * (args.optional_slots_per_base + 1)
        if args.max_slots < source_canvas_slots:
            args.max_slots = source_canvas_slots
        slot_labels = source_slot_labels(source_sequence, args.optional_slots_per_base, args.max_slots)
        (
            initial_base_logits,
            initial_presence_logits,
            source_real_slot_mask,
            source_real_slot_base,
        ) = make_source_canvas_initialisation(
            source=source_sequence,
            batch_size=args.batch_size,
            max_slots=args.max_slots,
            optional_slots_per_base=args.optional_slots_per_base,
            base_logit_init_strength=args.base_logit_init_strength,
            presence_present_init=args.presence_present_init,
            presence_absent_init=args.presence_absent_init,
            base_init_std=args.base_init_std,
            device=device,
        )
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
        initial_base_logits=initial_base_logits,
        initial_presence_logits=initial_presence_logits,
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
        f"presence_gate={args.presence_gate}; hard_presence={args.hard_presence}; "
        f"sample_presence={args.sample_presence}; lr={args.lr}"
    )
    if args.mode == "reconstruct":
        print("target[0]:", sequences[0])
    elif args.mode == "indel":
        print("source:", source_sequence)
        print("target:", sequences[0])
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
            presence_gate=args.presence_gate,
            sample_presence=args.sample_presence,
            presence_threshold=args.presence_threshold,
            hard_concrete_gamma=args.hard_concrete_gamma,
            hard_concrete_zeta=args.hard_concrete_zeta,
            max_pack_sharpness=args.max_pack_sharpness,
        )
        if args.mode in {"reconstruct", "indel"}:
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
            if args.mode == "indel" and args.source_base_anchor_weight > 0:
                assert source_real_slot_mask is not None and source_real_slot_base is not None
                anchor_log_probs = rendered["base_probs"].clamp_min(1e-8).log()
                source_base_nll = -anchor_log_probs.gather(-1, source_real_slot_base[..., None]).squeeze(-1)
                source_mask = source_real_slot_mask.to(dtype=source_base_nll.dtype)
                source_anchor_loss = (
                    source_base_nll * source_mask
                ).sum() / source_mask.sum().clamp_min(1.0)
                loss = loss + args.source_base_anchor_weight * source_anchor_loss
                final_metrics["source_anchor_loss"] = float(source_anchor_loss.detach().item())
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
            if args.mode in {"reconstruct", "indel"}:
                print("target: ", sequences[0])
                if args.mode == "indel":
                    print("source: ", source_sequence)
                print(
                    f"example0 acc {final_metrics.get('example0_acc', 0.0):.3f} "
                    f"exact {final_metrics.get('example0_exact', 0.0):.0f}; "
                    f"batch exact {final_metrics.get('exact_count', 0.0):.0f}/{args.batch_size}"
                )
            print("decoded:", decoded[0])

    assert final_rendered is not None
    decoded = decode_batch(final_rendered, output_length)
    print("\nFinal:")
    print("decoded[0]:", decoded[0])
    if args.mode in {"reconstruct", "indel"}:
        print("target[0]: ", sequences[0])
    if args.mode == "indel":
        print("source:    ", source_sequence)
    print("presence probs[0]:", format_presence(final_rendered["presence_prob"][0]))
    print("selected slots[0]:", selected_slot_summary(final_rendered["presence_prob"][0]))
    print("base argmax[0]:", tensor_to_dna(final_rendered["base_probs"][0].argmax(dim=-1)))
    with torch.no_grad():
        designer.eval()
        hard_rendered = designer(
            output_length=output_length,
            base_temperature=args.base_temperature,
            presence_temperature=args.presence_temperature,
            hard_bases=True,
            sample_bases=False,
            hard_presence=True,
            presence_gate=args.presence_gate,
            sample_presence=False,
            presence_threshold=args.presence_threshold,
            hard_concrete_gamma=args.hard_concrete_gamma,
            hard_concrete_zeta=args.hard_concrete_zeta,
            max_pack_sharpness=args.max_pack_sharpness,
        )
        hard_decoded = decode_batch(hard_rendered, output_length)
        print("hard decoded[0]:", hard_decoded[0])
        print("hard active slots[0]:", selected_slot_summary(hard_rendered["presence_for_pack"][0]))
        if args.mode == "indel":
            print(
                "hard active trace[0]:",
                active_slot_trace(
                    presence=hard_rendered["presence_for_pack"][0],
                    base_probs=hard_rendered["base_probs"][0],
                    labels=slot_labels,
                ),
            )
        if args.mode in {"reconstruct", "indel"}:
            assert target is not None
            hard_metrics = reconstruction_metrics(hard_rendered["soft_dna"], target)
            print(
                f"hard example0 acc {hard_metrics['example0_acc']:.3f} "
                f"exact {hard_metrics['example0_exact']:.0f}; "
                f"batch exact {hard_metrics['exact_count']:.0f}/{args.batch_size}"
            )
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
    parser.add_argument("--mode", choices=("reconstruct", "indel", "motif"), default="reconstruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-slots", type=int, default=160)
    parser.add_argument("--target-len", type=int, default=100)
    parser.add_argument("--output-len", type=int, default=100)
    parser.add_argument("--target-dna", default="")
    parser.add_argument("--source-dna", default="")
    parser.add_argument("--source-len", type=int, default=100)
    parser.add_argument("--optional-slots-per-base", type=int, default=1)
    parser.add_argument("--indel-insert", default="GATTACA")
    parser.add_argument("--indel-insert-at", type=int, default=50)
    parser.add_argument("--indel-delete-start", type=int, default=25)
    parser.add_argument("--indel-delete-len", type=int, default=5)
    parser.add_argument("--source-base-anchor-weight", type=float, default=1.0)
    parser.add_argument("--motif", default="TATAAA")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--base-init-std", type=float, default=0.01)
    parser.add_argument("--presence-init-mode", choices=("even", "uniform"), default="even")
    parser.add_argument("--base-logit-init-strength", type=float, default=5.0)
    parser.add_argument("--presence-present-init", type=float, default=4.0)
    parser.add_argument("--presence-absent-init", type=float, default=-4.0)
    parser.add_argument("--base-temperature", type=float, default=1.0)
    parser.add_argument("--presence-temperature", type=float, default=1.0)
    parser.add_argument("--presence-gate", choices=("sigmoid", "hard_concrete"), default="sigmoid")
    parser.add_argument("--sample-presence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--hard-concrete-gamma", type=float, default=-0.1)
    parser.add_argument("--hard-concrete-zeta", type=float, default=1.1)
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
