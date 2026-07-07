from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def configure_mamba_cache(cache_root: Path | None = None) -> Path:
    """Keep Triton/Mamba JIT artifacts out of quota-limited default caches."""

    configured_root = cache_root or os.environ.get("MAMBA_TRITON_CACHE_ROOT") or os.environ.get("CACHE_ROOT")
    root = Path(configured_root) if configured_root else ROOT / ".cache" / "mamba_triton_cache"
    root = root.expanduser()
    os.environ["MAMBA_TRITON_CACHE_ROOT"] = str(root)
    cache_dirs = {
        "TRITON_CACHE_DIR": root / "triton",
        "TORCH_EXTENSIONS_DIR": root / "torch_extensions",
        "XDG_CACHE_HOME": root / "xdg",
        "TMPDIR": root / "tmp",
    }
    for env_name, path in cache_dirs.items():
        os.environ[env_name] = str(path)
        path.mkdir(parents=True, exist_ok=True)
    return root

from central_dogma_ai.structured_phase import (  # noqa: E402
    PHASE_STATES,
    PathCodonLogits,
    SplicePath,
    SyntheticPhaseGene,
    StructuredPhaseLayer,
    StructuredTranslationPhaseModel,
    generate_multiexon_phase_gene,
    generate_single_exon_phase_gene,
    phase_nll_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Mamba + structured phase helper using weighted candidate splice paths."
    )
    parser.add_argument("--device", default="auto", help="cuda, cpu, mps, or auto")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "structured_phase_helper_weighted_paths",
    )
    parser.add_argument("--init-from", type=Path, default=None, help="Load model weights from a previous checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Also restore optimizer state and continue step numbering.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--examples-per-step", type=int, default=4)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--min-orf-codons", type=int, default=2)
    parser.add_argument("--start-loss-weight", type=float, default=0.25)
    parser.add_argument("--stop-loss-weight", type=float, default=0.25)
    parser.add_argument("--mode", choices=("single_exon", "multiexon"), default="single_exon")
    parser.add_argument("--evidence-model", choices=("motif", "mamba"), default="motif")
    parser.add_argument("--mamba-hidden-dim", type=int, default=64)
    parser.add_argument("--mamba-layers", type=int, default=4)
    parser.add_argument("--mamba-chunk-size", type=int, default=16)
    parser.add_argument("--mamba-headdim", type=int, default=8)
    parser.add_argument(
        "--mamba-cache-root",
        type=Path,
        default=None,
        help="Scratch/cache root for Mamba/Triton compilation artifacts.",
    )
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--min-utr5-length", type=int, default=6)
    parser.add_argument("--max-utr5-length", type=int, default=24)
    parser.add_argument("--min-coding-codons", type=int, default=3)
    parser.add_argument("--max-coding-codons", type=int, default=12)
    parser.add_argument("--min-utr3-length", type=int, default=6)
    parser.add_argument("--max-utr3-length", type=int, default=24)

    parser.add_argument("--min-exons", type=int, default=2)
    parser.add_argument("--max-exons", type=int, default=4)
    parser.add_argument("--min-exon-length", type=int, default=6)
    parser.add_argument("--min-intron-length", type=int, default=8)
    parser.add_argument("--max-intron-length", type=int, default=24)
    parser.add_argument(
        "--allow-split-start-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow ATG/stop codons to be split by introns. Enabled by default for natural random CDS intron insertion."
        ),
    )
    parser.add_argument(
        "--require-split-codon",
        choices=("none", "any", "start", "stop"),
        default="none",
        help="Force sampled multiexon genes to have split start and/or stop codons across exon junctions.",
    )
    parser.add_argument(
        "--unsplit-codon-fraction",
        type=float,
        default=0.0,
        help="When requiring split codons, keep this fraction as clean examples with neither start nor stop split.",
    )
    parser.add_argument("--init-textbook", action="store_true", help="Initialize motif detector as ATG/stop oracle.")
    parser.add_argument(
        "--splice-track-mode",
        choices=("first_path", "union", "weighted"),
        default="weighted",
        help="How to encode candidate splice graph evidence for Mamba.",
    )
    parser.add_argument(
        "--soft-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow phase targets to be probability distributions for mixed isoform labels.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def randint(rng: random.Random, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"Invalid range: {low}..{high}")
    return rng.randint(low, high)


def codon_is_contiguous_in_genome(gene: SyntheticPhaseGene, codon_start: int, codon: str | None = None) -> bool:
    observed = gene.dna[codon_start : codon_start + 3]
    if codon is not None and observed == codon:
        return True
    if codon is None and len(observed) == 3:
        path_set = set(gene.paths[0].genomic_indices.tolist())
        return all(index in path_set for index in range(codon_start, codon_start + 3))
    return False


def split_codon_flags(gene: SyntheticPhaseGene) -> tuple[bool, bool]:
    start_split = not codon_is_contiguous_in_genome(gene, gene.start_codon_start, "ATG")
    stop_split = gene.dna[gene.stop_codon_start : gene.stop_codon_start + 3] not in {"TAA", "TAG", "TGA"}
    return start_split, stop_split


def split_requirement_is_met(gene: SyntheticPhaseGene, requirement: str) -> bool:
    start_split, stop_split = split_codon_flags(gene)
    if requirement == "none":
        return True
    if requirement == "any":
        return start_split or stop_split
    if requirement == "start":
        return start_split
    if requirement == "stop":
        return stop_split
    raise ValueError(f"Unknown split requirement: {requirement}")


def split_counts_for_gene(gene: SyntheticPhaseGene) -> dict[str, int]:
    start_split, stop_split = split_codon_flags(gene)
    return {
        "split_start": int(start_split),
        "split_stop": int(stop_split),
        "split_any": int(start_split or stop_split),
        "split_none": int(not start_split and not stop_split),
    }


def make_gene(args: argparse.Namespace, rng: random.Random) -> SyntheticPhaseGene:
    if args.require_split_codon != "none" and not args.allow_split_start_stop:
        raise ValueError("--require-split-codon needs --allow-split-start-stop")
    if args.unsplit_codon_fraction < 0.0 or args.unsplit_codon_fraction > 1.0:
        raise ValueError("--unsplit-codon-fraction must be between 0 and 1")

    if args.mode == "single_exon":
        utr5_length = randint(rng, args.min_utr5_length, args.max_utr5_length)
        coding_codons = randint(rng, args.min_coding_codons, args.max_coding_codons)
        utr3_length = randint(rng, args.min_utr3_length, args.max_utr3_length)
        return generate_single_exon_phase_gene(
            utr5_length=utr5_length,
            coding_codons=coding_codons,
            utr3_length=utr3_length,
            seed=rng.randrange(2**31),
        )

    use_unsplit_anchor = (
        args.require_split_codon != "none"
        and args.unsplit_codon_fraction > 0.0
        and rng.random() < args.unsplit_codon_fraction
    )
    for _attempt in range(500):
        utr5_length = randint(rng, args.min_utr5_length, args.max_utr5_length)
        coding_codons = randint(rng, args.min_coding_codons, args.max_coding_codons)
        utr3_length = randint(rng, args.min_utr3_length, args.max_utr3_length)
        exon_count = randint(rng, args.min_exons, args.max_exons)
        try:
            gene = generate_multiexon_phase_gene(
                utr5_length=utr5_length,
                coding_codons=coding_codons,
                utr3_length=utr3_length,
                exon_count=exon_count,
                min_exon_length=args.min_exon_length,
                min_intron_length=args.min_intron_length,
                max_intron_length=args.max_intron_length,
                seed=rng.randrange(2**31),
            )
        except ValueError:
            continue
        start_split, stop_split = split_codon_flags(gene)
        if args.require_split_codon != "none":
            if use_unsplit_anchor and not start_split and not stop_split:
                return gene
            if not use_unsplit_anchor and split_requirement_is_met(gene, args.require_split_codon):
                return gene
            continue
        if args.allow_split_start_stop or (not start_split and not stop_split):
            return gene
    raise RuntimeError(
        "Could not sample a valid multiexon gene. Try lowering --min-exon-length/--max-exons "
        "or increasing UTR/coding length ranges."
    )


def tensor_to_device(gene: SyntheticPhaseGene, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return gene.dna_one_hot.to(device), gene.target_states.to(device)


def target_labels(target_states: torch.Tensor) -> torch.Tensor:
    if target_states.dtype.is_floating_point:
        if target_states.ndim != 2 or target_states.shape[-1] != len(PHASE_STATES):
            raise ValueError("soft target states must have shape L x states")
        return target_states.argmax(dim=-1)
    return target_states


def phase_loss_for_target(output, target_states: torch.Tensor, *, allow_soft_targets: bool) -> torch.Tensor:
    if target_states.dtype.is_floating_point:
        if not allow_soft_targets:
            raise ValueError("Received soft target states but --no-soft-targets is set")
        target = target_states.to(device=output.state_log_probs.device, dtype=output.state_log_probs.dtype)
        if target.shape != output.state_log_probs.shape:
            raise ValueError("soft target states must match output state log-prob shape")
        safe_log_probs = output.state_log_probs.masked_fill(target <= 0, 0.0)
        return -(target * safe_log_probs).sum(dim=-1).mean()
    return phase_nll_loss(output, target_states)


def splice_tracks_for_gene(gene: SyntheticPhaseGene, device: torch.device, mode: str) -> torch.Tensor:
    return splice_tracks_for_paths(len(gene.dna), gene.paths, device, mode=mode)


def splice_tracks_for_paths(length: int, paths: tuple, device: torch.device, *, mode: str = "weighted") -> torch.Tensor:
    tracks = torch.zeros(length, 2, dtype=torch.float32, device=device)
    if not paths:
        return tracks
    if mode == "first_path":
        paths = paths[:1]
    path_log_weights = []
    for path in paths:
        if path.log_weight is None:
            path_log_weights.append(torch.zeros((), dtype=torch.float32, device=device))
        else:
            path_log_weights.append(path.log_weight.to(device=device, dtype=torch.float32))
    if mode == "union":
        path_weights = torch.ones(len(paths), dtype=torch.float32, device=device)
    elif mode in {"first_path", "weighted"}:
        path_weights = torch.softmax(torch.stack(path_log_weights), dim=0)
    else:
        raise ValueError(f"Unknown splice track mode: {mode}")
    for path, path_weight in zip(paths, path_weights):
        path_indices = path.genomic_indices.to(device=device, dtype=torch.long)
        for offset in range(int(path_indices.numel()) - 1):
            left = int(path_indices[offset].item())
            right = int(path_indices[offset + 1].item())
            if right != left + 1:
                tracks[left, 0] += path_weight
                tracks[right, 1] += path_weight
    return tracks.clamp(max=1.0)
    return tracks


def import_mamba2():
    if "TRITON_CACHE_DIR" not in os.environ:
        configure_mamba_cache()
    try:
        from mamba_ssm import Mamba2
    except ImportError as exc:
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
        except ImportError:
            raise RuntimeError(
                "Mamba evidence model needs mamba-ssm. Use the bipangolin/cluster env where Mamba2 is installed."
            ) from exc
    return Mamba2


class OfficialMamba2Block(nn.Module):
    def __init__(self, hidden_dim: int, *, chunk_size: int = 16, headdim: int = 8):
        super().__init__()
        Mamba2 = import_mamba2()
        d_inner = 2 * hidden_dim
        if d_inner % headdim != 0:
            raise ValueError("2 * hidden_dim must be divisible by headdim")
        nheads = d_inner // headdim
        fused_projection_width = 2 * d_inner + 2 * 32 + nheads
        if fused_projection_width % 8 != 0:
            raise ValueError("Mamba2 fused projection width must be divisible by 8; try --mamba-headdim 8")
        self.chunk_size = chunk_size
        self.norm = nn.LayerNorm(hidden_dim)
        self.mamba = Mamba2(
            d_model=hidden_dim,
            d_state=32,
            d_conv=4,
            expand=2,
            headdim=headdim,
            chunk_size=chunk_size,
        )
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
        padding = x.new_zeros(x.shape[0], pad_length, x.shape[2])
        return torch.cat([x, padding], dim=1), pad_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        padded, pad_length = self._pad_to_chunk(self.norm(x))
        y = self.mamba(padded)
        if isinstance(y, tuple):
            y = y[0]
        if pad_length:
            y = y[:, :-pad_length]
        return residual + y


class OfficialMamba2Encoder(nn.Module):
    def __init__(self, hidden_dim: int, *, layers: int, chunk_size: int, headdim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [OfficialMamba2Block(hidden_dim, chunk_size=chunk_size, headdim=headdim) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MambaStructuredPhaseModel(nn.Module):
    def __init__(
        self,
        *,
        min_orf_codons: int,
        hidden_dim: int,
        layers: int,
        chunk_size: int,
        headdim: int,
        bidirectional: bool,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_projection = nn.Linear(6, hidden_dim)
        self.local_stem = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.forward_encoder = OfficialMamba2Encoder(
            hidden_dim,
            layers=layers,
            chunk_size=chunk_size,
            headdim=headdim,
        )
        if bidirectional:
            self.reverse_encoder = OfficialMamba2Encoder(
                hidden_dim,
                layers=layers,
                chunk_size=chunk_size,
                headdim=headdim,
            )
            self.bidirectional_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.codon_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.phase_layer = StructuredPhaseLayer(min_orf_codons=min_orf_codons)

    def encode_genome(self, dna_one_hot: torch.Tensor, splice_tracks: torch.Tensor) -> torch.Tensor:
        features = torch.cat([dna_one_hot, splice_tracks], dim=-1).unsqueeze(0)
        hidden = self.input_projection(features)
        hidden = hidden + self.local_stem(hidden.transpose(1, 2)).transpose(1, 2)
        forward_hidden = self.forward_encoder(hidden)
        if not self.bidirectional:
            return forward_hidden.squeeze(0)
        reverse_hidden = torch.flip(self.reverse_encoder(torch.flip(hidden, dims=(1,))), dims=(1,))
        return self.bidirectional_fusion(torch.cat([forward_hidden, reverse_hidden], dim=-1)).squeeze(0)

    def path_logits(self, hidden: torch.Tensor, paths: tuple) -> tuple[PathCodonLogits, ...]:
        scored_paths = []
        for path in paths:
            indices = path.genomic_indices.to(device=hidden.device, dtype=torch.long)
            path_hidden = hidden[indices]
            length = int(path_hidden.shape[0])
            if length < 3:
                low = hidden.new_full((length,), -1.0e4)
                scored_paths.append(PathCodonLogits(start_logits=low, stop_logits=low))
                continue
            windows = torch.cat([path_hidden[:-2], path_hidden[1:-1], path_hidden[2:]], dim=-1)
            logits = self.codon_head(windows)
            pad = hidden.new_full((2,), -1.0e4)
            scored_paths.append(
                PathCodonLogits(
                    start_logits=torch.cat([logits[:, 0], pad], dim=0),
                    stop_logits=torch.cat([logits[:, 1], pad], dim=0),
                )
            )
        return tuple(scored_paths)

    def forward(self, dna_one_hot: torch.Tensor, splice_tracks: torch.Tensor, paths: tuple) -> object:
        """Use candidate splice graph paths instead of reconstructing one path from tracks."""

        hidden = self.encode_genome(dna_one_hot, splice_tracks)
        path_codon_logits = self.path_logits(hidden, paths)
        zeros = hidden.new_zeros(hidden.shape[0])
        return self.phase_layer(
            start_logits=zeros,
            stop_logits=zeros,
            paths=paths,
            path_codon_logits=path_codon_logits,
        )


def run_model_on_gene(model: nn.Module, args: argparse.Namespace, gene: SyntheticPhaseGene, dna_one_hot: torch.Tensor):
    if args.evidence_model == "mamba":
        splice_tracks = splice_tracks_for_gene(gene, dna_one_hot.device, mode=args.splice_track_mode)
        return model(dna_one_hot, splice_tracks, gene.paths)
    return model(dna_one_hot, gene.paths)


def state_counts(target: torch.Tensor) -> dict[str, int]:
    labels = target_labels(target)
    return {name: int((labels == index).sum().item()) for index, name in enumerate(PHASE_STATES)}


def compute_loss(
    output,
    target_states: torch.Tensor,
    gene: SyntheticPhaseGene,
    *,
    start_loss_weight: float,
    stop_loss_weight: float,
    allow_soft_targets: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    phase_loss = phase_loss_for_target(output, target_states, allow_soft_targets=allow_soft_targets)
    start_loss = -output.initiation_log_probs[gene.start_codon_start]
    stop_loss = -output.termination_log_probs[gene.stop_codon_start]
    total = phase_loss + start_loss_weight * start_loss + stop_loss_weight * stop_loss
    parts = {
        "phase": float(phase_loss.detach().item()),
        "start": float(start_loss.detach().item()),
        "stop": float(stop_loss.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, parts


@torch.no_grad()
def evaluate(
    model: StructuredTranslationPhaseModel,
    args: argparse.Namespace,
    *,
    device: torch.device,
    seed: int,
    examples: int,
) -> dict[str, object]:
    rng = random.Random(seed)
    model.eval()
    total_loss = 0.0
    total_phase_loss = 0.0
    total_start_loss = 0.0
    total_stop_loss = 0.0
    total_bases = 0
    correct_bases = 0
    exact_genes = 0
    start_exact = 0
    stop_exact = 0
    state_total = [0 for _ in PHASE_STATES]
    state_correct = [0 for _ in PHASE_STATES]
    length_sum = 0
    exon_sum = 0
    intron_sum = 0
    split_counts = {"split_start": 0, "split_stop": 0, "split_any": 0, "split_none": 0}

    for _ in range(examples):
        gene = make_gene(args, rng)
        dna_one_hot, target = tensor_to_device(gene, device)
        output = run_model_on_gene(model, args, gene, dna_one_hot)
        loss, parts = compute_loss(
            output,
            target,
            gene,
            start_loss_weight=args.start_loss_weight,
            stop_loss_weight=args.stop_loss_weight,
            allow_soft_targets=args.soft_targets,
        )
        predicted = output.state_log_probs.argmax(dim=-1)
        labels = target_labels(target)
        matches = predicted == labels

        total_loss += parts["total"]
        total_phase_loss += parts["phase"]
        total_start_loss += parts["start"]
        total_stop_loss += parts["stop"]
        total_bases += int(labels.numel())
        correct_bases += int(matches.sum().item())
        exact_genes += int(bool(matches.all().item()))
        start_exact += int(output.initiation_log_probs.argmax().item() == gene.start_codon_start)
        stop_exact += int(output.termination_log_probs.argmax().item() == gene.stop_codon_start)
        length_sum += len(gene.dna)
        exon_sum += len(gene.exon_lengths)
        intron_sum += len(gene.intron_lengths)
        for key, value in split_counts_for_gene(gene).items():
            split_counts[key] += value

        for state_index in range(len(PHASE_STATES)):
            mask = labels == state_index
            state_total[state_index] += int(mask.sum().item())
            state_correct[state_index] += int((matches & mask).sum().item())

    per_state = {
        name: (state_correct[index] / state_total[index] if state_total[index] else float("nan"))
        for index, name in enumerate(PHASE_STATES)
    }
    return {
        "loss": total_loss / examples,
        "phase_loss": total_phase_loss / examples,
        "start_loss": total_start_loss / examples,
        "stop_loss": total_stop_loss / examples,
        "base_accuracy": correct_bases / total_bases,
        "gene_exact": exact_genes / examples,
        "start_exact": start_exact / examples,
        "stop_exact": stop_exact / examples,
        "per_state": per_state,
        "mean_length": length_sum / examples,
        "mean_exons": exon_sum / examples,
        "mean_introns": intron_sum / examples,
        "split_start": split_counts["split_start"] / examples,
        "split_stop": split_counts["split_stop"] / examples,
        "split_any": split_counts["split_any"] / examples,
        "split_none": split_counts["split_none"] / examples,
    }


def format_metrics(prefix: str, metrics: dict[str, object]) -> str:
    per_state = metrics["per_state"]
    state_bits = ", ".join(f"{name}={per_state[name]:.3f}" for name in PHASE_STATES)
    return (
        f"{prefix} loss={metrics['loss']:.4f} "
        f"(phase={metrics['phase_loss']:.4f}, start={metrics['start_loss']:.4f}, stop={metrics['stop_loss']:.4f})\n"
        f"           phase base={metrics['base_accuracy']:.3f}, gene exact={metrics['gene_exact']:.3f}, "
        f"start peak={metrics['start_exact']:.3f}, stop peak={metrics['stop_exact']:.3f}\n"
        f"           split codons any={metrics['split_any']:.3f}, none={metrics['split_none']:.3f}, "
        f"start={metrics['split_start']:.3f}, stop={metrics['split_stop']:.3f}\n"
        f"           per-state {state_bits}"
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    validation: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "validation": validation,
            "phase_states": PHASE_STATES,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    resume: bool,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    strict = bool(resume)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if not strict:
        missing = load_result.missing_keys
        unexpected = load_result.unexpected_keys
        if missing or unexpected:
            print(f"checkpoint partially loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    start_step = 0
    best_loss = float("inf")
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        validation = checkpoint.get("validation") or {}
        if "loss" in validation:
            best_loss = float(validation["loss"])
    return start_step, best_loss


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.evidence_model == "motif":
        return StructuredTranslationPhaseModel(min_orf_codons=args.min_orf_codons)
    return MambaStructuredPhaseModel(
        min_orf_codons=args.min_orf_codons,
        hidden_dim=args.mamba_hidden_dim,
        layers=args.mamba_layers,
        chunk_size=args.mamba_chunk_size,
        headdim=args.mamba_headdim,
        bidirectional=args.bidirectional,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mamba_cache_root = configure_mamba_cache(args.mamba_cache_root)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(args).to(device)
    if args.init_textbook and hasattr(model, "feature_extractor"):
        model.feature_extractor.initialize_textbook_motifs(strength=4.0, bias=-8.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    best_loss = float("inf")
    if args.init_from is not None:
        start_step, best_loss = load_checkpoint(
            args.init_from,
            model=model,
            optimizer=optimizer,
            device=device,
            resume=args.resume,
        )
    train_rng = random.Random(args.seed + 101)
    start_time = time.time()

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Mamba/Triton cache root: {mamba_cache_root}")
    print("task: structured translated phase helper with weighted candidate splice paths")
    print(f"states: {PHASE_STATES}")
    print(f"evidence model: {args.evidence_model}")
    if args.evidence_model == "mamba":
        print(
            "Mamba evidence: "
            f"hidden={args.mamba_hidden_dim}, layers={args.mamba_layers}, "
            f"chunk={args.mamba_chunk_size}, headdim={args.mamba_headdim}, "
            f"{'bidirectional' if args.bidirectional else 'forward-only'}"
        )
        print("Mamba path source: weighted candidate splice paths supplied to structured layer")
        print(f"Mamba splice tracks: {args.splice_track_mode} donor/acceptor evidence; no path reconstruction")
        print(f"phase target mode: {'soft labels allowed' if args.soft_targets else 'hard labels only'}")
    print(
        f"mode: {args.mode}; examples_per_step={args.examples_per_step}; "
        f"validation_examples={args.validation_examples}; validate_every={args.validate_every}"
    )
    print(
        "synthetic data: "
        f"UTR5={args.min_utr5_length}-{args.max_utr5_length} bp, "
        f"coding={args.min_coding_codons}-{args.max_coding_codons} codons, "
        f"UTR3={args.min_utr3_length}-{args.max_utr3_length} bp"
    )
    if args.mode == "multiexon":
        print(
            "splice structure: "
            f"exons={args.min_exons}-{args.max_exons}, "
            f"introns={args.min_intron_length}-{args.max_intron_length} bp, "
            f"split_start_stop={'allowed' if args.allow_split_start_stop else 'disabled'}, "
            f"require_split={args.require_split_codon}, "
            f"unsplit_fraction={args.unsplit_codon_fraction:.2f}"
        )
    print(f"checkpoint directory: {args.checkpoint_dir}")
    if args.init_from is not None:
        print(f"loaded checkpoint: {args.init_from}")
        if args.resume:
            print(f"resuming from step: {start_step}")
    print("starting training")

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss = None
        loss_parts = {"phase": 0.0, "start": 0.0, "stop": 0.0, "total": 0.0}
        train_counts = {"bases": 0, "correct": 0, "exact": 0, "start": 0, "stop": 0}
        state_total = [0 for _ in PHASE_STATES]
        state_correct = [0 for _ in PHASE_STATES]
        length_sum = 0
        exon_sum = 0
        intron_sum = 0
        split_counts = {"split_start": 0, "split_stop": 0, "split_any": 0, "split_none": 0}

        for _example_index in range(args.examples_per_step):
            gene = make_gene(args, train_rng)
            dna_one_hot, target = tensor_to_device(gene, device)
            output = run_model_on_gene(model, args, gene, dna_one_hot)
            loss, parts = compute_loss(
                output,
                target,
                gene,
                start_loss_weight=args.start_loss_weight,
                stop_loss_weight=args.stop_loss_weight,
                allow_soft_targets=args.soft_targets,
            )
            scaled_loss = loss / args.examples_per_step
            step_loss = scaled_loss if step_loss is None else step_loss + scaled_loss

            with torch.no_grad():
                predicted = output.state_log_probs.argmax(dim=-1)
                labels = target_labels(target)
                matches = predicted == labels
                train_counts["bases"] += int(labels.numel())
                train_counts["correct"] += int(matches.sum().item())
                train_counts["exact"] += int(bool(matches.all().item()))
                train_counts["start"] += int(output.initiation_log_probs.argmax().item() == gene.start_codon_start)
                train_counts["stop"] += int(output.termination_log_probs.argmax().item() == gene.stop_codon_start)
                length_sum += len(gene.dna)
                exon_sum += len(gene.exon_lengths)
                intron_sum += len(gene.intron_lengths)
                for key, value in split_counts_for_gene(gene).items():
                    split_counts[key] += value
                for state_index in range(len(PHASE_STATES)):
                    mask = labels == state_index
                    state_total[state_index] += int(mask.sum().item())
                    state_correct[state_index] += int((matches & mask).sum().item())
            for key in loss_parts:
                loss_parts[key] += parts[key] / args.examples_per_step

        assert step_loss is not None
        step_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        should_print = step == 1 or step % args.print_every == 0 or step == args.steps
        should_save = step % args.checkpoint_every == 0 or step == args.steps
        should_validate = (
            step == 1
            or step == args.steps
            or should_save
            or (args.validate_every > 0 and step % args.validate_every == 0)
        )
        validation = None
        if should_validate:
            validation = evaluate(
                model,
                args,
                device=device,
                seed=args.seed + 100000 + step,
                examples=args.validation_examples,
            )
            if validation["loss"] < best_loss:
                best_loss = float(validation["loss"])
                save_checkpoint(
                    args.checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    step=step,
                    validation=validation,
                )
                print(f"saved best checkpoint: {args.checkpoint_dir / 'best.pt'}")
        if should_save:
            if validation is None:
                validation = evaluate(
                    model,
                    args,
                    device=device,
                    seed=args.seed + 100000 + step,
                    examples=args.validation_examples,
                )
            save_checkpoint(
                args.checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                step=step,
                validation=validation,
            )
            with (args.checkpoint_dir / "latest_metrics.json").open("w") as handle:
                json.dump({"step": step, "validation": validation}, handle, indent=2)
            print(f"saved checkpoint: {args.checkpoint_dir / 'latest.pt'}")

        if should_print or should_validate or should_save:
            per_state = {
                name: (state_correct[index] / state_total[index] if state_total[index] else float("nan"))
                for index, name in enumerate(PHASE_STATES)
            }
            elapsed = time.time() - start_time
            print(
                f"\nStep {step:06d} | learning_rate={optimizer.param_groups[0]['lr']:.2e} | "
                f"elapsed={elapsed:.1f}s"
            )
            print(
                "Batch shape | "
                f"genes={args.examples_per_step}, genome mean={length_sum / args.examples_per_step:.1f} bp, "
                f"exons mean={exon_sum / args.examples_per_step:.2f}, "
                f"introns mean={intron_sum / args.examples_per_step:.2f}"
            )
            print(
                f"train      loss={loss_parts['total']:.4f} "
                f"(phase={loss_parts['phase']:.4f}, start={loss_parts['start']:.4f}, stop={loss_parts['stop']:.4f})"
            )
            print(
                f"           phase base={train_counts['correct'] / train_counts['bases']:.3f}, "
                f"gene exact={train_counts['exact'] / args.examples_per_step:.3f}, "
                f"start peak={train_counts['start'] / args.examples_per_step:.3f}, "
                f"stop peak={train_counts['stop'] / args.examples_per_step:.3f}"
            )
            print(
                f"           split codons any={split_counts['split_any'] / args.examples_per_step:.3f}, "
                f"none={split_counts['split_none'] / args.examples_per_step:.3f}, "
                f"start={split_counts['split_start'] / args.examples_per_step:.3f}, "
                f"stop={split_counts['split_stop'] / args.examples_per_step:.3f}"
            )
            print("           per-state " + ", ".join(f"{name}={per_state[name]:.3f}" for name in PHASE_STATES))
            if validation is not None:
                print(format_metrics("validation", validation))


if __name__ == "__main__":
    main()
