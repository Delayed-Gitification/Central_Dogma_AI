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
from central_dogma_ai.torch_model import fixed_translate_codons  # noqa: E402

try:
    from mamba_ssm import Mamba2
except ImportError as exc:
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
    except ImportError:
        raise RuntimeError(
            "Official Mamba2 is required. On the CUDA node, install it with something like:\n"
            "  python -m pip install causal-conv1d mamba-ssm\n"
            "Use a PyTorch/CUDA module or environment that matches the cluster CUDA version."
        ) from exc


NONSTOP_AA = [aa for aa in AMINO_ACIDS if aa != "*"]
MODEL_TRACK_NAMES = ("exon_prior",)
ANNOTATION_TRACK_NAMES = ("exon_prior", "donor", "acceptor", "true_transcript_rank")


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
    model_tracks = torch.tensor([[value] for value in exon_prior], dtype=torch.float32)
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
        "target": target,
        "cds_target": cds_target,
    }


def make_batch(
    batch_size: int,
    device: torch.device,
    min_protein_codons: int,
    max_protein_codons: int,
    min_exon_count: int = 1,
    max_exon_count: int = 6,
    min_exon_bases: int = 5,
    min_intron_length: int = 200,
    max_intron_length: int = 3_000,
    length_bucket_size: int = 1_024,
    seed: int | None = None,
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
    max_length = round_up_to_multiple(max(example["dna"].shape[0] for example in examples), length_bucket_size)
    max_target_length = max(example["target"].shape[0] for example in examples)
    max_transcript_bases = max_target_length * 3

    dna_rows = []
    track_rows = []
    target_rows = []
    target_mask_rows = []
    base_target_rows = []
    base_mask_rows = []
    pointer_target_rows = []
    for example in examples:
        dna = example["dna"]
        tracks = example["tracks"]
        target = example["target"]
        base_target = example["cds_target"]
        padding = max_length - dna.shape[0]
        target_padding = max_target_length - target.shape[0]
        base_padding = max_transcript_bases - base_target.shape[0]
        dna_rows.append(torch.cat([dna, torch.zeros(padding, 4)], dim=0))
        track_rows.append(torch.cat([tracks, torch.zeros(padding, tracks.shape[1])], dim=0))
        target_rows.append(torch.cat([target, torch.zeros(target_padding, dtype=target.dtype)], dim=0))
        target_mask_rows.append(
            torch.cat(
                [
                    torch.ones(target.shape[0], dtype=torch.bool),
                    torch.zeros(target_padding, dtype=torch.bool),
                ],
                dim=0,
            )
        )
        base_target_rows.append(torch.cat([base_target, torch.zeros(base_padding, dtype=base_target.dtype)], dim=0))
        base_mask_rows.append(
            torch.cat(
                [
                    torch.ones(base_target.shape[0], dtype=torch.bool),
                    torch.zeros(base_padding, dtype=torch.bool),
                ],
                dim=0,
            )
        )
        true_transcript_rank = example["annotations"][:, 3]
        exonic_positions = true_transcript_rank >= 0
        transcript_ranks = true_transcript_rank[exonic_positions].long()
        genomic_indices = torch.nonzero(exonic_positions, as_tuple=False).flatten().long()
        pointer_target = torch.zeros(max_transcript_bases, dtype=torch.long)
        pointer_target[transcript_ranks] = genomic_indices
        pointer_target_rows.append(pointer_target)

    return (
        torch.stack(dna_rows).to(device),
        torch.stack(track_rows).to(device),
        torch.stack(target_rows).to(device),
        torch.stack(target_mask_rows).to(device),
        torch.stack(base_target_rows).to(device),
        torch.stack(base_mask_rows).to(device),
        torch.stack(pointer_target_rows).to(device),
        max_transcript_bases,
        examples,
    )


class OfficialMamba2Block(nn.Module):
    def __init__(self, hidden_dim: int, chunk_size: int = 16, headdim: int = 8):
        super().__init__()
        self.chunk_size = chunk_size
        self.norm = nn.LayerNorm(hidden_dim)
        d_inner = 2 * hidden_dim
        if d_inner % headdim != 0:
            raise ValueError(f"2 * hidden_dim must be divisible by headdim, got {d_inner=} and {headdim=}")
        nheads = d_inner // headdim
        fused_projection_width = 2 * d_inner + 2 * 32 + nheads
        if fused_projection_width % 8 != 0:
            raise ValueError(
                "Official Mamba2's fused CUDA causal-conv path needs an internal projection width "
                f"divisible by 8, got {fused_projection_width}. Try --headdim 8 for hidden_dim=32."
            )
        self.mamba = Mamba2(
            d_model=hidden_dim,
            d_state=32,
            d_conv=4,
            expand=2,
            headdim=headdim,
            chunk_size=chunk_size,
        )
        self._reset_mamba_parameters()

    def _reset_mamba_parameters(self) -> None:
        with torch.no_grad():
            if hasattr(self.mamba, "dt_bias"):
                self.mamba.dt_bias.fill_(-2.0)
            if hasattr(self.mamba, "A_log"):
                self.mamba.A_log.zero_()
            if hasattr(self.mamba, "D"):
                self.mamba.D.fill_(1.0)

    def _pad_to_chunk(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        remainder = x.shape[1] % self.chunk_size
        if remainder == 0:
            return x, 0
        pad_length = self.chunk_size - remainder
        padding = torch.zeros(x.shape[0], pad_length, x.shape[2], dtype=x.dtype, device=x.device)
        return torch.cat([x, padding], dim=1), pad_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        normalised = self.norm(x)
        padded, pad_length = self._pad_to_chunk(normalised)
        y = self.mamba(padded)
        if isinstance(y, tuple):
            y = y[0]
        if pad_length:
            y = y[:, :-pad_length]
        return residual + y


class OfficialMamba2Encoder(nn.Module):
    def __init__(self, hidden_dim: int, layers: int = 2, chunk_size: int = 16, headdim: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [OfficialMamba2Block(hidden_dim, chunk_size=chunk_size, headdim=headdim) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MambaSplicePointerTranslator(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 32,
        layers: int = 3,
        chunk_size: int = 16,
        headdim: int = 8,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scan_blocks = OfficialMamba2Encoder(
            hidden_dim=hidden_dim,
            layers=layers,
            chunk_size=chunk_size,
            headdim=headdim,
        )
        self.query_projection = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coordinate_step_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.content_scale = nn.Parameter(torch.tensor(0.2))
        self.log_coordinate_sharpness = nn.Parameter(torch.tensor(-1.0))
        self.exon_prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, dna_one_hot: torch.Tensor, splice_tracks: torch.Tensor, transcript_bases: int):
        if transcript_bases % 3 != 0:
            raise ValueError(f"transcript_bases must be divisible by 3, got {transcript_bases}")
        batch_size, genome_length, _ = dna_one_hot.shape
        exon_prior = splice_tracks[..., 0:1]
        features = torch.cat([dna_one_hot, splice_tracks], dim=-1)

        genome_position = torch.linspace(0, 1, genome_length, device=dna_one_hot.device)
        genome_position = genome_position[None, :, None].expand(batch_size, -1, -1)
        encoded = (
            self.input_projection(features)
            + self.position_projection(torch.cat([genome_position, exon_prior], dim=-1))
        )
        encoded = self.scan_blocks(encoded)

        coordinate_step = F.softplus(self.coordinate_step_head(encoded).squeeze(-1))
        coordinate_step = coordinate_step * exon_prior.squeeze(-1)
        latent_coordinate = torch.cumsum(coordinate_step, dim=1)
        latent_coordinate = latent_coordinate / latent_coordinate[:, -1:].clamp_min(1e-6)

        target_index = torch.arange(transcript_bases, device=dna_one_hot.device)
        target_coordinate = torch.linspace(0, 1, transcript_bases, device=dna_one_hot.device)
        target_frame = F.one_hot(target_index % 3, num_classes=3).to(dtype=encoded.dtype)
        query_features = torch.cat([target_coordinate[:, None], target_frame], dim=-1)
        query = self.query_projection(query_features)
        content_logits = torch.einsum("bld,td->btl", encoded, query) / math.sqrt(encoded.shape[-1])

        coordinate_sharpness = F.softplus(self.log_coordinate_sharpness)
        coordinate_bias = -coordinate_sharpness * (
            latent_coordinate[:, None, :] - target_coordinate[None, :, None]
        ).pow(2)
        exon_bias = self.exon_prior_scale.abs() * torch.log(
            exon_prior.squeeze(-1).clamp_min(1e-4)
        )[:, None, :]

        pointer_logits = self.content_scale * content_logits + coordinate_bias + exon_bias
        attention = pointer_logits.softmax(dim=-1)

        transcript_base_probs = torch.einsum("btl,blc->btc", attention, dna_one_hot)
        codon_bases = transcript_base_probs.reshape(batch_size, -1, 3, 4)
        amino_acid_probs = fixed_translate_codons(codon_bases).clamp_min(1e-8)

        attention_entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        mean_exon_attention = torch.einsum(
            "btl,bl->bt", attention, exon_prior.squeeze(-1)
        ).mean()
        coordinate_span = (
            latent_coordinate.max(dim=1).values - latent_coordinate.min(dim=1).values
        ).mean()

        return amino_acid_probs, transcript_base_probs, attention, pointer_logits, {
            "attention_entropy": attention_entropy.detach(),
            "coordinate_sharpness": coordinate_sharpness.detach(),
            "coordinate_span": coordinate_span.detach(),
            "exon_prior_scale": self.exon_prior_scale.detach().abs(),
            "mean_exon_attention": mean_exon_attention.detach(),
        }


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA is not available. Submit this script to a GPU node with a CUDA PyTorch build.")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    return device


def get_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    decay_step = min(max(0, step - warmup_steps), decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
    return min_lr + (base_lr - min_lr) * cosine


def get_linear_annealed_weight(step: int, start_weight: float, end_step: int) -> float:
    if start_weight <= 0 or end_step <= 0:
        return 0.0
    if step >= end_step:
        return 0.0
    return start_weight * (1.0 - float(step) / float(end_step))


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def estimate_max_genome_length(args: argparse.Namespace) -> int:
    max_cds_bases = args.max_protein_codons * 3
    max_intronic_bases = max(0, args.max_exon_count - 1) * args.max_intron_length
    return max_cds_bases + max_intronic_bases


def resolve_micro_batch_size(args: argparse.Namespace) -> int:
    if args.micro_batch_size > 0:
        return min(args.batch_size, args.micro_batch_size)
    estimated_length = max(1, estimate_max_genome_length(args))
    token_limited_size = max(1, args.max_micro_batch_tokens // estimated_length)
    return min(args.batch_size, token_limited_size)


def resolve_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    return checkpoint_path


def checkpoint_payload(
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    final_metrics: dict[str, float | int] | None,
    best_loss: float,
) -> dict[str, object]:
    return {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "final_metrics": final_metrics,
        "best_loss": best_loss,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_random_state": random.getstate(),
    }


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, dict[str, float | int] | None]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if checkpoint.get("cuda_rng_state_all") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    start_step = int(checkpoint.get("step", -1)) + 1
    best_loss = float(checkpoint.get("best_loss", float("inf")))
    final_metrics = checkpoint.get("final_metrics")
    if final_metrics is not None and not isinstance(final_metrics, dict):
        final_metrics = None
    return start_step, best_loss, final_metrics


def maybe_save_checkpoint(
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    final_metrics: dict[str, float | int] | None,
    best_loss: float,
    checkpoint_dir: Path,
    is_final_step: bool,
) -> None:
    should_save_latest = args.checkpoint_every > 0 and ((step + 1) % args.checkpoint_every == 0 or is_final_step)
    should_save_numbered = args.checkpoint_keep_every > 0 and (
        (step + 1) % args.checkpoint_keep_every == 0 or is_final_step
    )
    if not should_save_latest and not should_save_numbered:
        return

    payload = checkpoint_payload(
        step=step,
        model=model,
        optimizer=optimizer,
        args=args,
        final_metrics=final_metrics,
        best_loss=best_loss,
    )
    if should_save_latest:
        latest_path = checkpoint_dir / "latest.pt"
        save_checkpoint(latest_path, payload)
        print(f"saved checkpoint: {latest_path}", flush=True)
    if should_save_numbered:
        numbered_path = checkpoint_dir / f"step_{step + 1:09d}.pt"
        save_checkpoint(numbered_path, payload)
        print(f"saved checkpoint: {numbered_path}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.min_protein_codons < 1:
        raise ValueError("--min-protein-codons must be at least 1")
    if args.max_protein_codons < args.min_protein_codons:
        raise ValueError("--max-protein-codons must be >= --min-protein-codons")
    if args.min_exon_count < 1:
        raise ValueError("--min-exon-count must be at least 1")
    if args.max_exon_count < args.min_exon_count:
        raise ValueError("--max-exon-count must be >= --min-exon-count")
    if args.min_exon_bases < 1:
        raise ValueError("--min-exon-bases must be at least 1")
    if args.min_intron_length < 4:
        raise ValueError("--min-intron-length must be at least 4")
    if args.max_intron_length < args.min_intron_length:
        raise ValueError("--max-intron-length must be >= --min-intron-length")
    if args.eval_protein_codons < 1:
        raise ValueError("--eval-protein-codons must be at least 1")
    if args.eval_exon_count < 1:
        raise ValueError("--eval-exon-count must be at least 1")
    if args.eval_exon_count * args.min_exon_bases > args.eval_protein_codons * 3:
        raise ValueError("--eval-exon-count is too large for --eval-protein-codons and --min-exon-bases")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.micro_batch_size < 0:
        raise ValueError("--micro-batch-size must be non-negative; use 0 for auto")
    if args.max_micro_batch_tokens < 1:
        raise ValueError("--max-micro-batch-tokens must be at least 1")
    if args.length_bucket_size < 0:
        raise ValueError("--length-bucket-size must be non-negative")
    if args.pointer_loss_weight < 0:
        raise ValueError("--pointer-loss-weight must be non-negative")
    if args.pointer_loss_anneal_steps < 0:
        raise ValueError("--pointer-loss-anneal-steps must be non-negative")
    if args.print_every < 1:
        raise ValueError("--print-every must be at least 1")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.checkpoint_keep_every < 0:
        raise ValueError("--checkpoint-keep-every must be non-negative")


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"model tracks: {MODEL_TRACK_NAMES}")
    print(f"annotation tracks generated: {ANNOTATION_TRACK_NAMES}")
    print("true_transcript_rank is used only for annealed pointer supervision")
    print(
        "synthetic length regime: "
        f"protein_codons={args.min_protein_codons}-{args.max_protein_codons}, "
        f"exons={args.min_exon_count}-{args.max_exon_count}, "
        f"introns={args.min_intron_length}-{args.max_intron_length} bp"
    )
    effective_micro_batch_size = resolve_micro_batch_size(args)
    micro_batch_mode = "auto" if args.micro_batch_size == 0 else "manual"
    print(
        f"effective batch size: {args.batch_size}; micro batch size: {effective_micro_batch_size}; "
        f"micro batch mode: {micro_batch_mode}; length bucket: {args.length_bucket_size} bp"
    )

    model = MambaSplicePointerTranslator(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        chunk_size=args.chunk_size,
        headdim=args.headdim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir)
    checkpointing_enabled = (
        args.checkpoint_every > 0 or args.checkpoint_keep_every > 0 or args.save_best_checkpoint
    )
    if checkpointing_enabled:
        print(f"checkpoint directory: {checkpoint_dir}")

    start_step = 0
    best_loss = float("inf")
    final_metrics = None
    resume_path = None
    if args.resume_from:
        resume_path = resolve_checkpoint_path(args.resume_from)
    elif args.auto_resume:
        latest_path = checkpoint_dir / "latest.pt"
        if latest_path.exists():
            resume_path = latest_path

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {resume_path}")
        start_step, best_loss, final_metrics = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        print(f"resumed checkpoint: {resume_path}")
        print(f"resuming from step={start_step} with best_loss={best_loss:.6f}")

    if start_step >= args.steps:
        print(f"checkpoint has already reached requested --steps={args.steps}; skipping training loop")

    for step in range(start_step, args.steps):
        lr = get_lr(
            step=step,
            total_steps=args.steps,
            base_lr=args.learning_rate,
            min_lr=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        pointer_loss_weight = get_linear_annealed_weight(
            step=step,
            start_weight=args.pointer_loss_weight,
            end_step=args.pointer_loss_anneal_steps,
        )

        optimizer.zero_grad(set_to_none=True)
        micro_batch_count = math.ceil(args.batch_size / effective_micro_batch_size)
        total_examples = 0
        loss_sum = 0.0
        aa_loss_sum = 0.0
        nt_loss_sum = 0.0
        pointer_loss_sum = 0.0
        token_correct = 0
        token_total = 0
        exact_sum = 0.0
        nucleotide_correct = 0
        nucleotide_total = 0
        nucleotide_exact_sum = 0.0
        pointer_correct_total = 0
        pointer_total = 0
        pointer_exact_sum = 0.0
        target_lengths_all = []
        genome_lengths = []
        intron_lengths = []
        attention_entropy_sum = 0.0
        coordinate_sharpness_sum = 0.0
        coordinate_span_sum = 0.0
        mean_exon_attention_sum = 0.0

        for micro_batch_index in range(micro_batch_count):
            micro_batch_size = min(
                effective_micro_batch_size,
                args.batch_size - micro_batch_index * effective_micro_batch_size,
            )
            dna, splice_tracks, target, target_mask, base_target, base_mask, pointer_target, transcript_bases, examples = make_batch(
                batch_size=micro_batch_size,
                device=device,
                min_protein_codons=args.min_protein_codons,
                max_protein_codons=args.max_protein_codons,
                min_exon_count=args.min_exon_count,
                max_exon_count=args.max_exon_count,
                min_exon_bases=args.min_exon_bases,
                min_intron_length=args.min_intron_length,
                max_intron_length=args.max_intron_length,
                length_bucket_size=args.length_bucket_size,
                seed=args.batch_seed_offset + step * micro_batch_count + micro_batch_index,
            )
            amino_acid_probs, transcript_base_probs, _attention, pointer_logits, diagnostics = model(
                dna,
                splice_tracks,
                transcript_bases=transcript_bases,
            )
            per_token_loss = F.nll_loss(
                torch.log(amino_acid_probs).reshape(-1, len(AMINO_ACIDS)),
                target.reshape(-1),
                reduction="none",
            ).reshape_as(target)
            aa_loss = (per_token_loss * target_mask.to(per_token_loss.dtype)).sum() / target_mask.sum().clamp_min(1)
            per_base_loss = F.nll_loss(
                torch.log(transcript_base_probs.clamp_min(1e-8)).reshape(-1, len(DNA_BASES)),
                base_target.reshape(-1),
                reduction="none",
            ).reshape_as(base_target)
            nt_loss = (per_base_loss * base_mask.to(per_base_loss.dtype)).sum() / base_mask.sum().clamp_min(1)
            per_pointer_loss = F.cross_entropy(
                pointer_logits.reshape(-1, pointer_logits.shape[-1]),
                pointer_target.reshape(-1),
                reduction="none",
            ).reshape_as(pointer_target)
            pointer_loss = (per_pointer_loss * base_mask.to(per_pointer_loss.dtype)).sum() / base_mask.sum().clamp_min(1)
            loss = aa_loss + args.nucleotide_loss_weight * nt_loss + pointer_loss_weight * pointer_loss
            loss_scale = float(micro_batch_size) / float(args.batch_size)
            (loss * loss_scale).backward()

            with torch.no_grad():
                predicted = amino_acid_probs.argmax(dim=-1)
                correct = (predicted == target) & target_mask
                predicted_bases = transcript_base_probs.argmax(dim=-1)
                base_correct = (predicted_bases == base_target) & base_mask
                predicted_pointer = pointer_logits.argmax(dim=-1)
                pointer_correct = (predicted_pointer == pointer_target) & base_mask
                target_lengths = target_mask.sum(dim=1)

                total_examples += micro_batch_size
                loss_sum += loss.item() * micro_batch_size
                aa_loss_sum += aa_loss.item() * micro_batch_size
                nt_loss_sum += nt_loss.item() * micro_batch_size
                pointer_loss_sum += pointer_loss.item() * micro_batch_size
                token_correct += int(correct.sum().item())
                token_total += int(target_mask.sum().item())
                exact_sum += float(((predicted == target) | ~target_mask).all(dim=1).float().sum().item())
                nucleotide_correct += int(base_correct.sum().item())
                nucleotide_total += int(base_mask.sum().item())
                nucleotide_exact_sum += float(((predicted_bases == base_target) | ~base_mask).all(dim=1).float().sum().item())
                pointer_correct_total += int(pointer_correct.sum().item())
                pointer_total += int(base_mask.sum().item())
                pointer_exact_sum += float(((predicted_pointer == pointer_target) | ~base_mask).all(dim=1).float().sum().item())
                target_lengths_all.extend(int(length) for length in target_lengths.tolist())
                genome_lengths.extend(len(str(example["genome"])) for example in examples)
                intron_lengths.extend(
                    int(intron_length)
                    for example in examples
                    for intron_length in example["intron_lengths"]
                )
                attention_entropy_sum += diagnostics["attention_entropy"].item() * micro_batch_size
                coordinate_sharpness_sum += diagnostics["coordinate_sharpness"].item() * micro_batch_size
                coordinate_span_sum += diagnostics["coordinate_span"].item() * micro_batch_size
                mean_exon_attention_sum += diagnostics["mean_exon_attention"].item() * micro_batch_size

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = loss_sum / max(1, total_examples)
        aa_loss_value = aa_loss_sum / max(1, total_examples)
        nt_loss_value = nt_loss_sum / max(1, total_examples)
        pointer_loss_value = pointer_loss_sum / max(1, total_examples)
        token_accuracy = token_correct / max(1, token_total)
        exact = exact_sum / max(1, total_examples)
        nucleotide_accuracy = nucleotide_correct / max(1, nucleotide_total)
        nucleotide_exact = nucleotide_exact_sum / max(1, total_examples)
        pointer_accuracy = pointer_correct_total / max(1, pointer_total)
        pointer_exact = pointer_exact_sum / max(1, total_examples)
        mean_target_length = sum(target_lengths_all) / max(1, len(target_lengths_all))
        max_target_length = max(target_lengths_all) if target_lengths_all else 0
        mean_genome_length = sum(genome_lengths) / max(1, len(genome_lengths))
        max_genome_length = max(genome_lengths) if genome_lengths else 0
        mean_intron_length = sum(intron_lengths) / len(intron_lengths) if intron_lengths else 0.0
        max_intron_length = max(intron_lengths) if intron_lengths else 0
        attention_entropy = attention_entropy_sum / max(1, total_examples)
        coordinate_sharpness = coordinate_sharpness_sum / max(1, total_examples)
        coordinate_span = coordinate_span_sum / max(1, total_examples)
        mean_exon_attention = mean_exon_attention_sum / max(1, total_examples)

        final_metrics = {
            "step": step,
            "lr": lr,
            "loss": loss_value,
            "aa_loss": aa_loss_value,
            "nt_loss": nt_loss_value,
            "pointer_loss": pointer_loss_value,
            "pointer_loss_weight": pointer_loss_weight,
            "micro_batch_size": effective_micro_batch_size,
            "length_bucket_size": args.length_bucket_size,
            "token_accuracy": token_accuracy,
            "exact_match": exact,
            "nucleotide_accuracy": nucleotide_accuracy,
            "nucleotide_exact_match": nucleotide_exact,
            "pointer_accuracy": pointer_accuracy,
            "pointer_exact_match": pointer_exact,
            "mean_target_length": mean_target_length,
            "max_target_length": max_target_length,
            "mean_genome_length": mean_genome_length,
            "max_genome_length": max_genome_length,
            "mean_intron_length": mean_intron_length,
            "max_intron_length": max_intron_length,
            "attention_entropy": attention_entropy,
            "coordinate_sharpness": coordinate_sharpness,
            "coordinate_span": coordinate_span,
            "mean_exon_attention": mean_exon_attention,
        }

        is_final_step = step == args.steps - 1
        is_report_step = step % args.print_every == 0 or is_final_step
        if args.save_best_checkpoint and is_report_step and loss_value < best_loss:
            best_loss = loss_value
            best_path = checkpoint_dir / "best.pt"
            save_checkpoint(
                best_path,
                checkpoint_payload(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    final_metrics=final_metrics,
                    best_loss=best_loss,
                ),
            )
            print(f"saved best checkpoint: {best_path}", flush=True)

        if is_report_step:
            print(
                f"step={step:03d} lr={lr:.2e} loss={loss_value:.3f} "
                f"aa_loss={aa_loss_value:.3f} nt_loss={nt_loss_value:.3f} "
                f"ptr_loss={pointer_loss_value:.3f} ptr_w={pointer_loss_weight:.3f} "
                f"token_acc={token_accuracy:.3f} exact={exact:.3f} "
                f"nt_acc={nucleotide_accuracy:.3f} nt_exact={nucleotide_exact:.3f} "
                f"ptr_acc={pointer_accuracy:.3f} ptr_exact={pointer_exact:.3f} "
                f"aa_len_mean={mean_target_length:.1f} aa_len_max={max_target_length} "
                f"genome_len_mean={mean_genome_length:.0f} genome_len_max={max_genome_length} "
                f"intron_len_mean={mean_intron_length:.0f} intron_len_max={max_intron_length} "
                f"micro_batch={effective_micro_batch_size}/{args.batch_size} "
                f"entropy={attention_entropy:.3f} "
                f"coord_span={coordinate_span:.2f} "
                f"coord_sharp={coordinate_sharpness:.3f} "
                f"exon_attention={mean_exon_attention:.3f}",
                flush=True,
            )

        maybe_save_checkpoint(
            step=step,
            model=model,
            optimizer=optimizer,
            args=args,
            final_metrics=final_metrics,
            best_loss=best_loss,
            checkpoint_dir=checkpoint_dir,
            is_final_step=is_final_step,
        )

    dna, splice_tracks, _target, target_mask, base_target, base_mask, _pointer_target, transcript_bases, examples = make_batch(
        batch_size=1,
        device=device,
        min_protein_codons=args.eval_protein_codons,
        max_protein_codons=args.eval_protein_codons,
        min_exon_count=args.eval_exon_count,
        max_exon_count=args.eval_exon_count,
        min_exon_bases=args.min_exon_bases,
        min_intron_length=args.min_intron_length,
        max_intron_length=args.max_intron_length,
        length_bucket_size=args.length_bucket_size,
        seed=args.eval_seed,
    )
    model.eval()
    with torch.no_grad():
        amino_acid_probs, transcript_base_probs, _attention, _pointer_logits, diagnostics = model(
            dna,
            splice_tracks,
            transcript_bases=transcript_bases,
        )
    eval_length = int(target_mask[0].sum().item())
    eval_base_length = int(base_mask[0].sum().item())
    prediction = "".join(AMINO_ACIDS[index] for index in amino_acid_probs.argmax(dim=-1)[0, :eval_length].tolist())
    predicted_cds = "".join(DNA_BASES[index] for index in transcript_base_probs.argmax(dim=-1)[0, :eval_base_length].tolist())

    print("\nFinal training metrics:")
    if final_metrics is not None:
        for key, value in final_metrics.items():
            print(f"{key}: {value}")

    print("\nHeld-out synthetic example:")
    print("target:    ", examples[0]["protein"])
    print("predicted: ", prediction)
    print("target cds:   ", examples[0]["cds"])
    print("predicted cds:", predicted_cds)
    print("protein codons:", examples[0]["protein_codons"])
    print("exon count:", examples[0]["exon_count"])
    print("exon lengths:", examples[0]["exon_lengths"])
    print("intron lengths:", examples[0]["intron_lengths"])
    print("genome length:", len(examples[0]["genome"]))
    print("attention entropy:", diagnostics["attention_entropy"].item())
    print("coordinate span:", diagnostics["coordinate_span"].item())
    print("coordinate sharpness:", diagnostics["coordinate_sharpness"].item())
    print("mean exon attention:", diagnostics["mean_exon_attention"].item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the synthetic splice-transducer with official mamba-ssm Mamba2 on CUDA."
    )
    parser.add_argument("--device", default="auto", help="Device to use. Default requires CUDA via auto.")
    parser.add_argument("--steps", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=0, help="Per-forward batch size. Use 0 for auto.")
    parser.add_argument("--max-micro-batch-tokens", type=int, default=500_000)
    parser.add_argument("--min-protein-codons", type=int, default=96)
    parser.add_argument("--max-protein-codons", type=int, default=256)
    parser.add_argument("--min-exon-count", type=int, default=3)
    parser.add_argument("--max-exon-count", type=int, default=10)
    parser.add_argument("--min-exon-bases", type=int, default=15)
    parser.add_argument("--eval-protein-codons", type=int, default=160)
    parser.add_argument("--eval-exon-count", type=int, default=6)
    parser.add_argument("--min-intron-length", type=int, default=200)
    parser.add_argument("--max-intron-length", type=int, default=3_000)
    parser.add_argument("--length-bucket-size", type=int, default=1_024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--nucleotide-loss-weight", type=float, default=0.5)
    parser.add_argument("--pointer-loss-weight", type=float, default=1.0)
    parser.add_argument("--pointer-loss-anneal-steps", type=int, default=10_000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-seed-offset", type=int, default=10_000)
    parser.add_argument("--eval-seed", type=int, default=123_456)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints/synthetic_splice_official_mamba2_variable_lengths")
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-keep-every", type=int, default=10_000)
    parser.add_argument("--save-best-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from", default="")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()