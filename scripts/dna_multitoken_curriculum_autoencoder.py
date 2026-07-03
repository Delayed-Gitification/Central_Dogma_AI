#!/usr/bin/env python3
"""Stage-2 multi-token DNA SoftPack curriculum.

This script starts from a trained single-token primitive model and teaches a
multi-token encoder to compose those primitives. The decoder/length head can be
frozen so the learned single-token manifold is treated as a stable DNA decoder
while the tokenizer/encoder learns to emit ordered primitive latents.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dna_single_token_entropy_curriculum_autoencoder as base


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device)


def args_from_checkpoint(checkpoint: dict) -> argparse.Namespace:
    loaded = argparse.Namespace()
    for key, value in checkpoint["args"].items():
        setattr(loaded, key, value)
    return loaded


def single_generation_kwargs(args: argparse.Namespace) -> dict:
    return {
        "curriculum_mode": args.curriculum_mode,
        "max_seq_len": args.component_max_len,
        "min_seq_len": args.component_min_len,
        "bit_budget": args.single_token_bit_budget,
        "kmer_sizes": base.parse_kmer_sizes(args.single_token_kmer_sizes),
        "entropy_percentile": args.single_token_entropy_percentile,
        "families": base.parse_family_list(args.single_token_families),
        "max_tries": args.single_token_max_tries,
        "program_max_motifs": args.program_max_motifs,
        "program_max_motif_len": args.program_max_motif_len,
        "program_repeat_lambda": args.program_repeat_lambda,
        "program_long_repeat_lambda": args.program_long_repeat_lambda,
        "program_long_repeat_prob": args.program_long_repeat_prob,
        "program_single_repeat_prob": args.program_single_repeat_prob,
    }


def make_multitoken_batch(
    *,
    batch_size: int,
    component_count: int,
    seq_len: int,
    component_max_len: int,
    rng: random.Random,
    device: torch.device,
    generation_kwargs: dict,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[list[str]],
]:
    target = torch.zeros(batch_size, seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.float32)
    component_target = torch.zeros(batch_size, component_count, component_max_len, dtype=torch.long)
    component_mask = torch.zeros(batch_size, component_count, component_max_len, dtype=torch.float32)
    component_lengths = torch.zeros(batch_size, component_count, dtype=torch.float32)
    component_bits = torch.zeros(batch_size, component_count, dtype=torch.float32)
    sequences: list[str] = []
    components: list[list[str]] = []

    for batch_index in range(batch_size):
        for attempt in range(200):
            parts: list[str] = []
            bits: list[float] = []
            for _ in range(component_count):
                sequence, _kind, sequence_bits, _entropy = base.generate_single_token_curriculum_sequence(
                    rng=rng,
                    **generation_kwargs,
                )
                parts.append(sequence)
                bits.append(sequence_bits)
            joined = "".join(parts)
            if len(joined) <= seq_len:
                break
            if attempt == 199:
                raise RuntimeError(
                    "could not sample a multi-token sequence within --seq-len; "
                    "increase --seq-len or reduce --component-max-len"
                )

        cursor = 0
        for component_index, part in enumerate(parts):
            part = part[:component_max_len]
            part_tensor = base.sequence_to_tensor(part)
            part_len = len(part)
            component_target[batch_index, component_index, :part_len] = part_tensor
            component_mask[batch_index, component_index, :part_len] = 1.0
            component_lengths[batch_index, component_index] = part_len
            component_bits[batch_index, component_index] = bits[component_index]
            target[batch_index, cursor : cursor + part_len] = part_tensor
            mask[batch_index, cursor : cursor + part_len] = 1.0
            cursor += part_len
        lengths[batch_index] = cursor
        sequences.append(joined)
        components.append(parts)

    return (
        target.to(device),
        mask.to(device),
        lengths.to(device),
        component_target.to(device),
        component_mask.to(device),
        component_lengths.to(device),
        component_bits.to(device),
        sequences,
        components,
    )


@torch.no_grad()
def teacher_primitives(
    teacher: base.AdaptiveTokenizerAutoencoder,
    component_target: torch.Tensor,
    component_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size, component_count, component_len = component_target.shape
    flat_target = component_target.reshape(batch_size * component_count, component_len)
    flat_mask = component_mask.reshape(batch_size * component_count, component_len)
    rendered = teacher(flat_target, flat_mask)
    latents = rendered["latents"][:, 0, :].reshape(batch_size, component_count, -1)
    lengths = rendered["total_len"].reshape(batch_size, component_count)
    return {"latents": latents.detach(), "lengths": lengths.detach()}


def primitive_supervision_loss(
    rendered: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    component_lengths: torch.Tensor,
    component_count: int,
    *,
    latent_weight: float,
    length_weight: float,
    usage_weight: float,
    token_count_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model_latents = rendered["latents"][:, :component_count, :]
    teacher_latents = teacher["latents"][:, :component_count, :]
    latent_cosine = F.cosine_similarity(model_latents, teacher_latents, dim=-1)
    latent_loss = (1.0 - latent_cosine).mean()

    model_lengths = rendered["lengths"][:, :component_count]
    primitive_length_loss = F.smooth_l1_loss(model_lengths, component_lengths[:, :component_count])

    target_usage = torch.zeros_like(rendered["token_usage"])
    target_usage[:, :component_count] = 1.0
    usage_loss = F.mse_loss(rendered["token_usage"], target_usage)
    token_count_target = torch.full_like(rendered["token_count"], float(component_count))
    token_count_loss = F.smooth_l1_loss(rendered["token_count"], token_count_target)

    loss = (
        latent_weight * latent_loss
        + length_weight * primitive_length_loss
        + usage_weight * usage_loss
        + token_count_weight * token_count_loss
    )
    return loss, {
        "primitive_loss": loss.detach(),
        "primitive_latent_loss": latent_loss.detach(),
        "primitive_length_loss": primitive_length_loss.detach(),
        "primitive_usage_loss": usage_loss.detach(),
        "primitive_token_count_loss": token_count_loss.detach(),
        "teacher_latent_cosine": latent_cosine.mean().detach(),
        "teacher_len_error": (model_lengths - component_lengths[:, :component_count]).abs().mean().detach(),
    }


def loss_for_multitoken_batch(
    *,
    model: base.AdaptiveTokenizerAutoencoder,
    teacher: base.AdaptiveTokenizerAutoencoder,
    target: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    component_target: torch.Tensor,
    component_mask: torch.Tensor,
    component_lengths: torch.Tensor,
    component_count: int,
    args: argparse.Namespace,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recon_loss, rendered = base.loss_for_batch(
        model=model,
        target=target,
        mask=mask,
        target_lengths=lengths,
        length_weight=args.length_weight,
        token_cost_weight=token_cost_weight,
        token_sharp_weight=token_sharp_weight,
        decoder_sharp_weight=decoder_sharp_weight,
        latent_l2_weight=args.latent_l2_weight,
        alignment_loss_weight=args.alignment_loss_weight,
        alignment_mode=args.alignment_mode,
        alignment_temperature=args.alignment_temperature,
        alignment_gap_cost=args.alignment_gap_cost,
        alignment_window=args.alignment_window,
        alignment_shift_cost=args.alignment_shift_cost,
        alignment_global_weight=args.alignment_global_weight,
        alignment_global_window=args.alignment_global_window,
    )
    teacher_rendered = teacher_primitives(teacher, component_target, component_mask)
    primitive_loss, primitive_metrics = primitive_supervision_loss(
        rendered,
        teacher_rendered,
        component_lengths,
        component_count,
        latent_weight=args.primitive_latent_weight,
        length_weight=args.primitive_length_weight,
        usage_weight=args.primitive_usage_weight,
        token_count_weight=args.primitive_token_count_weight,
    )
    loss = recon_loss + primitive_loss
    return loss, {**rendered, **primitive_metrics}


def freeze_primitive_decoder(model: base.AdaptiveTokenizerAutoencoder) -> None:
    for module in [model.slot_embedding, model.decoder, model.length_head]:
        for parameter in module.parameters():
            parameter.requires_grad = False


def optimizer_for_model(model: base.AdaptiveTokenizerAutoencoder, args: argparse.Namespace) -> torch.optim.Optimizer:
    groups = [
        {"params": model.encoder.parameters(), "lr": args.encoder_lr, "name": "encoder"},
        {"params": model.encoder_norm.parameters(), "lr": args.encoder_lr, "name": "encoder_norm"},
        {"params": model.token_head.parameters(), "lr": args.token_head_lr, "name": "token_head"},
        {"params": model.to_latent.parameters(), "lr": args.latent_lr, "name": "to_latent"},
    ]
    decoder_params = list(model.slot_embedding.parameters()) + list(model.decoder.parameters()) + list(model.length_head.parameters())
    if any(parameter.requires_grad for parameter in decoder_params):
        groups.append({"params": decoder_params, "lr": args.decoder_lr, "name": "primitive_decoder"})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def copy_checkpoint_weights(model: base.AdaptiveTokenizerAutoencoder, checkpoint: dict) -> None:
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    real_missing = [name for name in missing if "num_batches_tracked" not in name]
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={real_missing}, unexpected={unexpected}")


def build_multitoken_model(args: argparse.Namespace) -> base.AdaptiveTokenizerAutoencoder:
    return base.AdaptiveTokenizerAutoencoder(
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


def build_teacher(teacher_args: argparse.Namespace, checkpoint: dict, device: torch.device) -> base.AdaptiveTokenizerAutoencoder:
    teacher = base.build_model(teacher_args).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def checkpoint_compatible(args: argparse.Namespace, teacher_args: argparse.Namespace) -> None:
    required = ["latent_dim", "max_slots_per_token", "encoder_hidden_dim", "encoder_layers", "decoder_hidden_dim", "slot_dim"]
    mismatches = []
    for name in required:
        if getattr(args, name) != getattr(teacher_args, name):
            mismatches.append(f"{name}: run={getattr(args, name)} checkpoint={getattr(teacher_args, name)}")
    if mismatches:
        raise ValueError("architecture must match the single-token checkpoint:\n" + "\n".join(mismatches))


def short_status(
    *,
    step: int,
    loss: torch.Tensor,
    rendered: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    val_loss: float,
    val_rendered: dict[str, torch.Tensor],
    val_target: torch.Tensor,
    val_mask: torch.Tensor,
    val_lengths: torch.Tensor,
    component_count: int,
) -> list[str]:
    train_metrics = base.reconstruction_metrics(rendered["soft_dna"], target, mask)
    val_metrics = base.reconstruction_metrics(val_rendered["soft_dna"], val_target, val_mask)
    return [
        (
            f"\nstep {step:06d} | val loss {val_loss:.4f} acc {val_metrics['accuracy']:.3f} "
            f"exact {val_metrics['exact']:.3f} | teacher_cos {val_rendered['teacher_latent_cosine'].item():.3f} "
            f"token_count {val_rendered['token_count'].mean().item():.2f}/{component_count}"
        ),
        (
            f"train loss {loss.item():.4f} ce {rendered['recon_loss'].item():.4f} "
            f"align {rendered['alignment_loss'].item():.4f} acc {train_metrics['accuracy']:.3f} "
            f"exact {train_metrics['exact']:.3f} len {rendered['length_loss'].item():.3f} "
            f"prim {rendered['primitive_loss'].item():.4f} z {rendered['primitive_latent_loss'].item():.4f} "
            f"tok {rendered['token_count'].mean().item():.2f} out {rendered['total_len'].mean().item():.1f} "
            f"target {lengths.mean().item():.1f}"
        ),
        (
            f"val   ce {val_rendered['recon_loss'].item():.4f} align {val_rendered['alignment_loss'].item():.4f} "
            f"len {val_rendered['length_loss'].item():.3f} prim {val_rendered['primitive_loss'].item():.4f} "
            f"z {val_rendered['primitive_latent_loss'].item():.4f} token_use "
            f"{', '.join(f'{v:.2f}' for v in val_rendered['token_usage'].mean(dim=0).detach().cpu().tolist())} "
            f"emit_len {', '.join(f'{v:.1f}' for v in val_rendered['lengths'].mean(dim=0).detach().cpu().tolist())} "
            f"out {val_rendered['total_len'].mean().item():.1f}/{val_lengths.mean().item():.1f}"
        ),
    ]


def evaluate(
    *,
    model: base.AdaptiveTokenizerAutoencoder,
    teacher: base.AdaptiveTokenizerAutoencoder,
    rng: random.Random,
    device: torch.device,
    args: argparse.Namespace,
    generation_kwargs: dict,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
) -> tuple[float, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    total_loss = 0.0
    last_rendered = None
    last_target = None
    last_mask = None
    last_lengths = None
    for _ in range(args.val_batches):
        batch = make_multitoken_batch(
            batch_size=args.batch_size,
            component_count=args.component_count,
            seq_len=args.seq_len,
            component_max_len=args.component_max_len,
            rng=rng,
            device=device,
            generation_kwargs=generation_kwargs,
        )
        target, mask, lengths, component_target, component_mask, component_lengths, _bits, _seqs, _parts = batch
        loss, rendered = loss_for_multitoken_batch(
            model=model,
            teacher=teacher,
            target=target,
            mask=mask,
            lengths=lengths,
            component_target=component_target,
            component_mask=component_mask,
            component_lengths=component_lengths,
            component_count=args.component_count,
            args=args,
            token_cost_weight=token_cost_weight,
            token_sharp_weight=token_sharp_weight,
            decoder_sharp_weight=decoder_sharp_weight,
        )
        total_loss += loss.item()
        last_rendered = rendered
        last_target = target
        last_mask = mask
        last_lengths = lengths
    assert last_rendered is not None and last_target is not None and last_mask is not None and last_lengths is not None
    return total_loss / float(args.val_batches), last_rendered, last_target, last_mask, last_lengths


def run_diagnostics(
    model: base.AdaptiveTokenizerAutoencoder,
    teacher: base.AdaptiveTokenizerAutoencoder,
    rng: random.Random,
    device: torch.device,
    args: argparse.Namespace,
    generation_kwargs: dict,
) -> None:
    model.eval()
    with torch.no_grad():
        batch = make_multitoken_batch(
            batch_size=1,
            component_count=args.component_count,
            seq_len=args.seq_len,
            component_max_len=args.component_max_len,
            rng=rng,
            device=device,
            generation_kwargs=generation_kwargs,
        )
        target, mask, lengths, component_target, component_mask, component_lengths, _bits, _seqs, parts = batch
        loss, rendered = loss_for_multitoken_batch(
            model=model,
            teacher=teacher,
            target=target,
            mask=mask,
            lengths=lengths,
            component_target=component_target,
            component_mask=component_mask,
            component_lengths=component_lengths,
            component_count=args.component_count,
            args=args,
            token_cost_weight=args.token_cost_weight,
            token_sharp_weight=args.token_sharp_weight,
            decoder_sharp_weight=args.decoder_sharp_weight,
        )
        seq_len = int(lengths[0].item())
        decoded = rendered["soft_dna"].argmax(dim=-1)[0, :seq_len]
        print("\nDiagnostics:")
        print("components:", " + ".join(parts[0]))
        print("target:    ", base.tensor_to_sequence(target[0, :seq_len]))
        print("decoded:   ", base.tensor_to_sequence(decoded))
        print("loss:", f"{loss.item():.4f}", "teacher_cos:", f"{rendered['teacher_latent_cosine'].item():.3f}")
        print("token usage:", ", ".join(f"{value:.2f}" for value in rendered["token_usage"][0].detach().cpu().tolist()))
        print("emit lengths:", ", ".join(f"{value:.1f}" for value in rendered["lengths"][0].detach().cpu().tolist()))
    model.train()


def train(args: argparse.Namespace) -> None:
    if args.max_tokens < args.component_count:
        raise ValueError("--max-tokens must be >= --component-count")
    if args.max_tokens * args.max_slots_per_token < args.seq_len:
        raise ValueError("--max-tokens * --max-slots-per-token must be >= --seq-len")
    if args.component_max_len > args.max_slots_per_token:
        raise ValueError("--component-max-len must be <= --max-slots-per-token")
    if args.component_count <= 1:
        raise ValueError("This stage is for multi-token training; set --component-count >= 2")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = base.pick_device(args)
    train_rng = random.Random(args.seed)
    val_rng = random.Random(args.seed + 1_000_000)
    diag_rng = random.Random(args.seed + 2_000_000)

    checkpoint = load_checkpoint(args.init_checkpoint, device)
    teacher_args = args_from_checkpoint(checkpoint)
    checkpoint_compatible(args, teacher_args)
    teacher = build_teacher(teacher_args, checkpoint, device)
    model = build_multitoken_model(args).to(device)
    copy_checkpoint_weights(model, checkpoint)
    if args.freeze_primitive_decoder:
        freeze_primitive_decoder(model)
    optimizer = optimizer_for_model(model, args)
    generation_kwargs = single_generation_kwargs(args)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(
        f"multi-token curriculum; components={args.component_count}; seq_len={args.seq_len}; "
        f"max_tokens={args.max_tokens}; slots/token={args.max_slots_per_token}; params={total}; trainable={trainable}"
    )
    print(
        f"init_checkpoint={args.init_checkpoint}; freeze_decoder={args.freeze_primitive_decoder}; "
        f"primitive_weights z/len/use/count={args.primitive_latent_weight}/"
        f"{args.primitive_length_weight}/{args.primitive_usage_weight}/{args.primitive_token_count_weight}"
    )

    for step in range(args.steps):
        token_cost_weight = base.current_weight(step, args.steps, args.token_cost_weight, args.token_cost_warmup_frac)
        token_sharp_weight = base.current_weight(step, args.steps, args.token_sharp_weight, args.token_sharp_warmup_frac)
        decoder_sharp_weight = base.current_weight(step, args.steps, args.decoder_sharp_weight, args.decoder_sharp_warmup_frac)
        batch = make_multitoken_batch(
            batch_size=args.batch_size,
            component_count=args.component_count,
            seq_len=args.seq_len,
            component_max_len=args.component_max_len,
            rng=train_rng,
            device=device,
            generation_kwargs=generation_kwargs,
        )
        target, mask, lengths, component_target, component_mask, component_lengths, _bits, _seqs, _parts = batch
        loss, rendered = loss_for_multitoken_batch(
            model=model,
            teacher=teacher,
            target=target,
            mask=mask,
            lengths=lengths,
            component_target=component_target,
            component_mask=component_mask,
            component_lengths=component_lengths,
            component_count=args.component_count,
            args=args,
            token_cost_weight=token_cost_weight,
            token_sharp_weight=token_sharp_weight,
            decoder_sharp_weight=decoder_sharp_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss, val_rendered, val_target, val_mask, val_lengths = evaluate(
                    model=model,
                    teacher=teacher,
                    rng=val_rng,
                    device=device,
                    args=args,
                    generation_kwargs=generation_kwargs,
                    token_cost_weight=token_cost_weight,
                    token_sharp_weight=token_sharp_weight,
                    decoder_sharp_weight=decoder_sharp_weight,
                )
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                payload = {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "single_token_checkpoint": str(args.init_checkpoint),
                    "step": step,
                    "validation_loss": best_val_loss,
                }
                torch.save(payload, checkpoint_dir / "best.pt")
                if args.save_all_bests:
                    torch.save(payload, checkpoint_dir / f"best_step_{step:09d}.pt")
            for line in short_status(
                step=step,
                loss=loss,
                rendered=rendered,
                target=target,
                mask=mask,
                lengths=lengths,
                val_loss=val_loss,
                val_rendered=val_rendered,
                val_target=val_target,
                val_mask=val_mask,
                val_lengths=val_lengths,
                component_count=args.component_count,
            ):
                print(line)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "single_token_checkpoint": str(args.init_checkpoint),
            "step": args.steps - 1,
            "validation_loss": best_val_loss,
        },
        checkpoint_dir / "latest.pt",
    )
    run_diagnostics(model, teacher, diag_rng, device, args, generation_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 multi-token DNA SoftPack curriculum.")
    parser.add_argument("--init-checkpoint", default="checkpoints/dna_single_token_program_easy_len48_motif4/best.pt")
    parser.add_argument("--checkpoint-dir", default="checkpoints/dna_multitoken_from_single_motif4_two_tokens")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--component-count", type=int, default=2)
    parser.add_argument("--component-min-len", type=int, default=3)
    parser.add_argument("--component-max-len", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-slots-per-token", type=int, default=48)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--encoder-hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-hidden-dim", type=int, default=96)
    parser.add_argument("--slot-dim", type=int, default=16)

    parser.add_argument("--curriculum-mode", choices=("program", "entropy"), default="program")
    parser.add_argument("--single-token-bit-budget", type=float, default=32.0)
    parser.add_argument("--single-token-kmer-sizes", default="1,2,3,4,5")
    parser.add_argument("--single-token-entropy-percentile", type=float, default=0.1)
    parser.add_argument("--single-token-max-tries", type=int, default=200)
    parser.add_argument("--single-token-families", default="random,gc,at,homopolymer,dinucleotide,motif")
    parser.add_argument("--program-max-motifs", type=int, default=2)
    parser.add_argument("--program-max-motif-len", type=int, default=4)
    parser.add_argument("--program-repeat-lambda", type=float, default=0.6)
    parser.add_argument("--program-long-repeat-lambda", type=float, default=6.0)
    parser.add_argument("--program-long-repeat-prob", type=float, default=0.05)
    parser.add_argument("--program-single-repeat-prob", type=float, default=0.65)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--token-head-lr", type=float, default=3e-4)
    parser.add_argument("--latent-lr", type=float, default=5e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--freeze-primitive-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--token-cost-weight", type=float, default=0.0)
    parser.add_argument("--token-cost-warmup-frac", type=float, default=0.0)
    parser.add_argument("--token-sharp-weight", type=float, default=0.0005)
    parser.add_argument("--token-sharp-warmup-frac", type=float, default=0.0)
    parser.add_argument("--decoder-sharp-weight", type=float, default=0.0)
    parser.add_argument("--decoder-sharp-warmup-frac", type=float, default=0.0)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)
    parser.add_argument("--primitive-latent-weight", type=float, default=1.0)
    parser.add_argument("--primitive-length-weight", type=float, default=0.5)
    parser.add_argument("--primitive-usage-weight", type=float, default=1.0)
    parser.add_argument("--primitive-token-count-weight", type=float, default=0.5)

    parser.add_argument("--alignment-loss-weight", type=float, default=0.25)
    parser.add_argument("--alignment-mode", choices=("local_window", "dp", "none"), default="local_window")
    parser.add_argument("--alignment-temperature", type=float, default=0.2)
    parser.add_argument("--alignment-gap-cost", type=float, default=0.75)
    parser.add_argument("--alignment-window", type=int, default=2)
    parser.add_argument("--alignment-shift-cost", type=float, default=0.01)
    parser.add_argument("--alignment-global-weight", type=float, default=0.25)
    parser.add_argument("--alignment-global-window", type=int, default=8)

    parser.add_argument("--token-rank-temperature", type=float, default=0.35)
    parser.add_argument("--token-usage-temperature", type=float, default=0.5)
    parser.add_argument("--gate-temperature", type=float, default=0.2)
    parser.add_argument("--pack-temperature", type=float, default=0.1)
    parser.add_argument("--initial-token-stride", type=float, default=12.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--save-all-bests", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
