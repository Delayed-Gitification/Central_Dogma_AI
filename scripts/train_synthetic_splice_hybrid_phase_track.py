from __future__ import annotations

import argparse
import math
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


def configure_mamba_cache() -> Path:
    """Keep Triton/Mamba JIT artifacts out of quota-limited default caches."""

    cache_root = Path(
        os.environ.get("MAMBA_TRITON_CACHE_ROOT")
        or os.environ.get("CACHE_ROOT")
        or ROOT / ".cache" / "synthetic_splice_official_mamba2_phase_track"
    ).expanduser()
    cache_dirs = {
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TMPDIR": cache_root / "tmp",
    }
    for env_name, path in cache_dirs.items():
        os.environ.setdefault(env_name, str(path))
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)
    return cache_root


MAMBA_CACHE_ROOT = configure_mamba_cache()

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
MODEL_TRACK_NAMES = ("donor", "acceptor", "start_codon", "stop_codon")
ANNOTATION_TRACK_NAMES = (
    "exon_prior",
    "donor",
    "acceptor",
    "true_cds_rank",
    "region_code",
    "start_codon",
    "stop_codon",
)
REGION_INTRON = 0.0
REGION_CDS = 1.0
REGION_UTR5 = 2.0
REGION_UTR3 = 3.0


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


def random_coding_protein(codons: int, rng: random.Random) -> str:
    if codons < 2:
        raise ValueError("coding sequences need at least a start and stop codon")
    return "M" + "".join(rng.choice(NONSTOP_AA) for _ in range(codons - 2)) + "*"


def reverse_translate(protein: str, rng: random.Random) -> str:
    return "".join(rng.choice(CODONS_BY_AA[amino_acid]) for amino_acid in protein)


def reverse_translate_coding_protein(protein: str, rng: random.Random) -> str:
    if not protein or protein[0] != "M" or protein[-1] != "*":
        raise ValueError("coding proteins must start with M and end with *")
    codons = ["ATG"]
    codons.extend(rng.choice(CODONS_BY_AA[amino_acid]) for amino_acid in protein[1:-1])
    codons.append(rng.choice(CODONS_BY_AA["*"]))
    return "".join(codons)


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


def random_sampled_exon_lengths(
    exon_count: int,
    rng: random.Random,
    min_exon_bases: int,
    max_exon_bases: int,
    median_exon_bases: int,
    rare_short_exon_prob: float,
    force_total_multiple_of_3: bool = True,
) -> list[int]:
    """Sample exon lengths with a long tail and occasional very short exons."""

    if exon_count < 1:
        raise ValueError("exon_count must be positive")
    short_max = min(max_exon_bases, max(min_exon_bases, 12))
    lengths = []
    for _ in range(exon_count):
        if rng.random() < rare_short_exon_prob:
            length = rng.randint(min_exon_bases, short_max)
        else:
            length = round(rng.lognormvariate(math.log(float(median_exon_bases)), 0.75))
            length = max(min_exon_bases, min(max_exon_bases, length))
        lengths.append(length)

    remainder = sum(lengths) % 3
    if force_total_multiple_of_3 and remainder:
        add_bases = 3 - remainder
        for index in range(len(lengths) - 1, -1, -1):
            if lengths[index] + add_bases <= max_exon_bases:
                lengths[index] += add_bases
                break
        else:
            for index in range(len(lengths) - 1, -1, -1):
                if lengths[index] - remainder >= min_exon_bases:
                    lengths[index] -= remainder
                    break
            else:
                lengths[-1] += add_bases
    return lengths


def make_synthetic_example(
    protein_codons: int,
    exon_count: int,
    min_intron_length: int,
    max_intron_length: int,
    rng: random.Random,
    min_exon_bases: int = 5,
    max_exon_bases: int = 300,
    median_exon_bases: int = 50,
    rare_short_exon_prob: float = 0.05,
    exon_length_mode: str = "split",
    min_utr5_length: int = 0,
    max_utr5_length: int = 0,
    min_utr3_length: int = 0,
    max_utr3_length: int = 0,
) -> dict[str, object]:
    if exon_length_mode == "sampled":
        for _attempt in range(100):
            exon_lengths = random_sampled_exon_lengths(
                exon_count=exon_count,
                rng=rng,
                min_exon_bases=min_exon_bases,
                max_exon_bases=max_exon_bases,
                median_exon_bases=median_exon_bases,
                rare_short_exon_prob=rare_short_exon_prob,
                force_total_multiple_of_3=False,
            )
            total_exonic_bases = sum(exon_lengths)
            if total_exonic_bases < min_utr5_length + min_utr3_length + 6:
                continue
            effective_max_utr5 = min(max_utr5_length, total_exonic_bases - min_utr3_length - 6)
            if effective_max_utr5 < min_utr5_length:
                continue
            utr5_length = rng.randint(min_utr5_length, effective_max_utr5)
            effective_max_utr3 = min(max_utr3_length, total_exonic_bases - utr5_length - 6)
            if effective_max_utr3 < min_utr3_length:
                continue
            utr3_length = rng.randint(min_utr3_length, effective_max_utr3)
            coding_bases = total_exonic_bases - utr5_length - utr3_length
            if coding_bases < 6:
                continue
            remainder = coding_bases % 3
            if remainder:
                if utr3_length + remainder <= max_utr3_length:
                    utr3_length += remainder
                elif utr5_length + remainder <= max_utr5_length:
                    utr5_length += remainder
                else:
                    continue
            coding_bases = sum(exon_lengths) - utr5_length - utr3_length
            if coding_bases >= 6 and coding_bases % 3 == 0:
                break
        else:
            raise ValueError("could not sample exon/UTR lengths with an in-frame CDS")
        protein_codons = coding_bases // 3
    else:
        utr5_length = rng.randint(min_utr5_length, max_utr5_length)
        utr3_length = rng.randint(min_utr3_length, max_utr3_length)
        exon_lengths = random_split_lengths(
            utr5_length + protein_codons * 3 + utr3_length,
            exon_count,
            rng,
            min_part=min_exon_bases,
        )

    protein = random_coding_protein(protein_codons, rng)
    cds = reverse_translate_coding_protein(protein, rng)
    utr5 = random_dna(utr5_length, rng)
    utr3 = random_dna(utr3_length, rng)
    transcript = utr5 + cds + utr3

    genome_parts = []
    exon_prior = []
    donor_track = []
    acceptor_track = []
    start_codon_track = []
    stop_codon_track = []
    true_cds_rank = []
    region_code = []
    intron_lengths = []

    transcript_length = len(transcript)
    transcript_cursor = 0
    for exon_index, exon_length in enumerate(exon_lengths):
        exon = transcript[transcript_cursor : transcript_cursor + exon_length]
        for base_index, base in enumerate(exon):
            transcript_index = transcript_cursor + base_index
            cds_rank = transcript_index - utr5_length
            is_cds = 0 <= cds_rank < len(cds)
            is_utr5 = transcript_index < utr5_length
            is_utr3 = transcript_index >= utr5_length + len(cds)
            genome_parts.append(base)
            exon_prior.append(1.0)
            donor_track.append(1.0 if exon_index < exon_count - 1 and base_index == exon_length - 1 else 0.0)
            acceptor_track.append(1.0 if exon_index > 0 and base_index == 0 else 0.0)
            start_codon_track.append(1.0 if 0 <= cds_rank < 3 else 0.0)
            stop_codon_track.append(1.0 if len(cds) - 3 <= cds_rank < len(cds) else 0.0)
            true_cds_rank.append(float(cds_rank) if is_cds else -1.0)
            if is_cds:
                region_code.append(REGION_CDS)
            elif is_utr5:
                region_code.append(REGION_UTR5)
            elif is_utr3:
                region_code.append(REGION_UTR3)
            else:
                raise RuntimeError(f"transcript index outside transcript: {transcript_index=} {transcript_length=}")
        transcript_cursor += exon_length

        if exon_index < exon_count - 1:
            intron_length = rng.randint(min_intron_length, max_intron_length)
            intron_lengths.append(intron_length)
            intron = "GT" + random_dna(intron_length - 4, rng) + "AG"
            for base in intron:
                genome_parts.append(base)
                exon_prior.append(0.0)
                donor_track.append(0.0)
                acceptor_track.append(0.0)
                start_codon_track.append(0.0)
                stop_codon_track.append(0.0)
                true_cds_rank.append(-1.0)
                region_code.append(REGION_INTRON)

    genome = "".join(genome_parts)
    target = torch.tensor([AA_TO_INDEX[amino_acid] for amino_acid in protein], dtype=torch.long)
    cds_target = torch.tensor([DNA_TO_INDEX[base] for base in cds], dtype=torch.long)
    model_tracks = torch.tensor(list(zip(donor_track, acceptor_track, start_codon_track, stop_codon_track)), dtype=torch.float32)
    emit_target = torch.tensor(exon_prior, dtype=torch.float32)
    annotations = torch.tensor(
        list(zip(exon_prior, donor_track, acceptor_track, true_cds_rank, region_code, start_codon_track, stop_codon_track)),
        dtype=torch.float32,
    )
    return {
        "genome": genome,
        "protein": protein,
        "cds": cds,
        "utr5": utr5,
        "utr3": utr3,
        "transcript": transcript,
        "protein_codons": protein_codons,
        "exon_count": exon_count,
        "exon_lengths": exon_lengths,
        "intron_lengths": intron_lengths,
        "utr5_length": utr5_length,
        "utr3_length": utr3_length,
        "dna": one_hot_dna(genome),
        "tracks": model_tracks,
        "annotations": annotations,
        "emit_target": emit_target,
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
    max_exon_bases: int = 300,
    median_exon_bases: int = 50,
    rare_short_exon_prob: float = 0.05,
    exon_length_mode: str = "split",
    min_utr5_length: int = 0,
    max_utr5_length: int = 0,
    min_utr3_length: int = 0,
    max_utr3_length: int = 0,
    min_intron_length: int = 30,
    max_intron_length: int = 300,
    length_bucket_size: int = 256,
    seed: int | None = None,
) -> tuple[
    torch.Tensor,
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
        if exon_length_mode == "sampled":
            max_allowed_exons = max_exon_count
            min_allowed_exons = min_exon_count
        else:
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
                max_exon_bases=max_exon_bases,
                median_exon_bases=median_exon_bases,
                rare_short_exon_prob=rare_short_exon_prob,
                exon_length_mode=exon_length_mode,
                min_utr5_length=min_utr5_length,
                max_utr5_length=max_utr5_length,
                min_utr3_length=min_utr3_length,
                max_utr3_length=max_utr3_length,
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
    emit_target_rows = []
    pointer_target_rows = []
    for example in examples:
        dna = example["dna"]
        tracks = example["tracks"]
        emit_target = example["emit_target"]
        target = example["target"]
        base_target = example["cds_target"]
        padding = max_length - dna.shape[0]
        target_padding = max_target_length - target.shape[0]
        base_padding = max_transcript_bases - base_target.shape[0]
        dna_rows.append(torch.cat([dna, torch.zeros(padding, 4)], dim=0))
        track_rows.append(torch.cat([tracks, torch.zeros(padding, tracks.shape[1])], dim=0))
        emit_target_rows.append(torch.cat([emit_target, torch.zeros(padding, dtype=emit_target.dtype)], dim=0))
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
        torch.stack(emit_target_rows).to(device),
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


class LocalWindowAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        window_size: int = 129,
        feedforward_mult: int = 2,
    ):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim must be divisible by heads, got {hidden_dim=} and {heads=}")
        self.heads = heads
        self.window_size = window_size
        self.radius = window_size // 2
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * feedforward_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * feedforward_mult, hidden_dim),
        )

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (0, 0, self.radius, self.radius))
        return padded.unfold(dimension=2, size=self.window_size, step=1).permute(0, 2, 1, 4, 3)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = x.shape
        residual = x
        normalised = self.attn_norm(x)
        q, k, v = self.qkv(normalised).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.heads, self.head_dim)
        k = k.view(batch_size, sequence_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, sequence_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k_windows = self._windows(k)
        v_windows = self._windows(v)
        scores = (q.unsqueeze(-2) * k_windows).sum(dim=-1) * self.scale
        if valid_mask is not None:
            valid_windows = F.pad(valid_mask, (self.radius, self.radius), value=False).unfold(
                dimension=1,
                size=self.window_size,
                step=1,
            )
            query_valid = valid_mask[:, :, None, None]
            key_valid = valid_windows[:, :, None, :]
            scores = scores.masked_fill(~(key_valid | ~query_valid), torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        context = (attention.unsqueeze(-1) * v_windows).sum(dim=-2).reshape(batch_size, sequence_length, hidden_dim)
        x = residual + self.out_projection(context)
        return x + self.ffn(self.ffn_norm(x))


class MambaEmitSkipTranslator(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        layers: int = 3,
        chunk_size: int = 16,
        headdim: int = 8,
        use_prior_emit_mask: bool = False,
        max_assignment_sharpness: float = 10.0,
    ):
        super().__init__()
        self.use_prior_emit_mask = use_prior_emit_mask
        self.max_assignment_sharpness = max_assignment_sharpness
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
        self.emit_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_assignment_sharpness = nn.Parameter(torch.tensor(2.0))
        self._reset_emit_parameters()

    def _reset_emit_parameters(self) -> None:
        final_layer = self.emit_head[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, 4.0)

    def forward(self, dna_one_hot: torch.Tensor, splice_tracks: torch.Tensor, transcript_bases: int):
        if transcript_bases % 3 != 0:
            raise ValueError(f"transcript_bases must be divisible by 3, got {transcript_bases}")
        batch_size, genome_length, _ = dna_one_hot.shape
        splice_site_signal = splice_tracks.max(dim=-1, keepdim=True).values.clamp(0, 1)
        features = torch.cat([dna_one_hot, splice_tracks], dim=-1)

        genome_position = torch.linspace(0, 1, genome_length, device=dna_one_hot.device)
        genome_position = genome_position[None, :, None].expand(batch_size, -1, -1)
        encoded = (
            self.input_projection(features)
            + self.position_projection(torch.cat([genome_position, splice_site_signal], dim=-1))
        )
        encoded = self.scan_blocks(encoded)

        emit_logits = self.emit_head(encoded).squeeze(-1)
        emit_gate = splice_site_signal.squeeze(-1).clamp(0, 1) if self.use_prior_emit_mask else torch.ones_like(emit_logits)
        genome_mask = (dna_one_hot.sum(dim=-1) > 0).to(emit_logits.dtype)
        emit_prob = torch.sigmoid(emit_logits) * emit_gate * genome_mask
        soft_rank = torch.cumsum(emit_prob, dim=1)

        target_index = torch.arange(transcript_bases, device=dna_one_hot.device)
        target_coordinate = target_index.to(dtype=encoded.dtype) + 1.0
        assignment_sharpness = F.softplus(self.log_assignment_sharpness).clamp(max=self.max_assignment_sharpness)
        assignment_logits = -assignment_sharpness * (
            soft_rank[:, None, :] - target_coordinate[None, :, None]
        ).pow(2)
        assignment_logits = assignment_logits + torch.log(emit_prob.clamp_min(1e-6))[:, None, :]
        assignment = assignment_logits.softmax(dim=-1)

        transcript_base_probs = torch.einsum("btl,blc->btc", assignment, dna_one_hot)
        codon_bases = transcript_base_probs.reshape(batch_size, -1, 3, 4)
        amino_acid_probs = fixed_translate_codons(codon_bases).clamp_min(1e-8)

        assignment_entropy = -(
            assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        mean_splice_site_assignment = torch.einsum(
            "btl,bl->bt", assignment, splice_site_signal.squeeze(-1)
        ).mean()
        emit_count = emit_prob.sum(dim=1)

        return amino_acid_probs, transcript_base_probs, assignment, assignment_logits, {
            "emit_logits": emit_logits,
            "emit_prob": emit_prob,
            "assignment_entropy_loss": assignment_entropy,
            "assignment_entropy": assignment_entropy.detach(),
            "assignment_sharpness": assignment_sharpness.detach(),
            "emit_count_by_example": emit_count.detach(),
            "emit_count": emit_count.detach().mean(),
            "mean_emit_probability": emit_prob.detach().mean(),
            "mean_splice_site_assignment": mean_splice_site_assignment.detach(),
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
    if start_weight <= 0:
        return 0.0
    if end_step <= 0:
        return start_weight
    if step >= end_step:
        return 0.0
    return start_weight * (1.0 - float(step) / float(end_step))


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def estimate_max_genome_length(args: argparse.Namespace) -> int:
    if args.exon_length_mode == "sampled":
        max_cds_bases = args.max_exon_count * args.max_exon_bases
    else:
        max_cds_bases = args.max_protein_codons * 3
    max_utr_bases = args.max_utr5_length + args.max_utr3_length
    max_intronic_bases = max(0, args.max_exon_count - 1) * args.max_intron_length
    return max_cds_bases + max_utr_bases + max_intronic_bases


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


def load_model_weights(path: Path, *, model: nn.Module, device: torch.device) -> None:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


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


def empty_failure_stats() -> dict[str, float]:
    return {
        "examples": 0.0,
        "failed_examples": 0.0,
        "failed_target_bases_sum": 0.0,
        "failed_exon_count_sum": 0.0,
        "failed_max_intron_sum": 0.0,
        "failed_first_error_sum": 0.0,
        "failed_first_error_count": 0.0,
        "pointer_error_distance_sum": 0.0,
        "pointer_error_count": 0.0,
        "junction_pointer_correct": 0.0,
        "junction_pointer_total": 0.0,
    }


def merge_failure_stats(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def exon_junction_positions(exon_lengths: list[int], window: int) -> set[int]:
    if window < 0:
        window = 0
    junction_positions = set()
    cursor = 0
    for exon_length in exon_lengths[:-1]:
        cursor += int(exon_length)
        for position in range(cursor - window, cursor + window + 1):
            if position >= 0:
                junction_positions.add(position)
    return junction_positions


def summarize_failures(
    *,
    predicted_bases: torch.Tensor,
    base_target: torch.Tensor,
    base_mask: torch.Tensor,
    predicted_pointer: torch.Tensor,
    pointer_target: torch.Tensor,
    examples: list[dict[str, object]],
    junction_window: int,
) -> dict[str, float]:
    stats = empty_failure_stats()
    stats["examples"] = float(len(examples))

    predicted_bases_cpu = predicted_bases.detach().cpu()
    base_target_cpu = base_target.detach().cpu()
    base_mask_cpu = base_mask.detach().cpu()
    predicted_pointer_cpu = predicted_pointer.detach().cpu()
    pointer_target_cpu = pointer_target.detach().cpu()

    base_exact = ((predicted_bases_cpu == base_target_cpu) | ~base_mask_cpu).all(dim=1)
    for index, example in enumerate(examples):
        valid_mask = base_mask_cpu[index]
        valid_positions = torch.nonzero(valid_mask, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        pointer_wrong = (predicted_pointer_cpu[index] != pointer_target_cpu[index]) & valid_mask
        if pointer_wrong.any():
            pointer_distances = (
                predicted_pointer_cpu[index][pointer_wrong] - pointer_target_cpu[index][pointer_wrong]
            ).abs()
            stats["pointer_error_distance_sum"] += float(pointer_distances.sum().item())
            stats["pointer_error_count"] += float(pointer_distances.numel())

        junction_positions = exon_junction_positions(list(example["exon_lengths"]), junction_window)
        if junction_positions:
            near_junction = torch.zeros_like(valid_mask)
            for position in junction_positions:
                if position < near_junction.numel():
                    near_junction[position] = True
            near_junction &= valid_mask
            if near_junction.any():
                junction_correct = predicted_pointer_cpu[index][near_junction] == pointer_target_cpu[index][near_junction]
                stats["junction_pointer_correct"] += float(junction_correct.sum().item())
                stats["junction_pointer_total"] += float(junction_correct.numel())

        if bool(base_exact[index]):
            continue

        base_wrong = (predicted_bases_cpu[index] != base_target_cpu[index]) & valid_mask
        first_error = torch.nonzero(base_wrong, as_tuple=False).flatten()
        stats["failed_examples"] += 1.0
        stats["failed_target_bases_sum"] += float(valid_positions.numel())
        stats["failed_exon_count_sum"] += float(example["exon_count"])
        intron_lengths = list(example["intron_lengths"])
        stats["failed_max_intron_sum"] += float(max(intron_lengths) if intron_lengths else 0)
        if first_error.numel() > 0:
            stats["failed_first_error_sum"] += float(first_error[0].item())
            stats["failed_first_error_count"] += 1.0

    return stats


def failure_metrics(stats: dict[str, float]) -> dict[str, float]:
    failed_examples = max(1.0, stats["failed_examples"])
    pointer_error_count = max(1.0, stats["pointer_error_count"])
    junction_total = max(1.0, stats["junction_pointer_total"])
    first_error_count = max(1.0, stats["failed_first_error_count"])
    return {
        "failure_rate": stats["failed_examples"] / max(1.0, stats["examples"]),
        "failed_target_bases_mean": stats["failed_target_bases_sum"] / failed_examples,
        "failed_exon_count_mean": stats["failed_exon_count_sum"] / failed_examples,
        "failed_max_intron_mean": stats["failed_max_intron_sum"] / failed_examples,
        "failed_first_error_mean": stats["failed_first_error_sum"] / first_error_count,
        "pointer_error_distance_mean": stats["pointer_error_distance_sum"] / pointer_error_count,
        "junction_pointer_accuracy": stats["junction_pointer_correct"] / junction_total,
        "junction_pointer_total": stats["junction_pointer_total"],
    }


def format_report_metrics(label: str, metrics: dict[str, float | int]) -> str:
    return (
        f"{label:<10} "
        f"loss total={metrics['loss']:.3f} "
        f"(protein={metrics['aa_loss']:.3f}, cds={metrics['nt_loss']:.3f}, emit_skip={metrics['emit_loss']:.3f})\n"
        f"{'':<10} "
        f"protein exact={metrics['exact_match']:.3f}, protein token={metrics['token_accuracy']:.3f}, "
        f"CDS exact={metrics['nucleotide_exact_match']:.3f}, CDS base={metrics['nucleotide_accuracy']:.3f}\n"
        f"{'':<10} "
        f"splice route exact={metrics['assignment_exact_match']:.3f}, route base={metrics['assignment_accuracy']:.3f}, "
        f"junction route={metrics['junction_pointer_accuracy']:.3f} (n={metrics['junction_pointer_total']:.0f})\n"
        f"{'':<10} "
        f"confidence protein={metrics['aa_confidence']:.3f}, CDS={metrics['nucleotide_confidence']:.3f}, "
        f"route={metrics['assignment_confidence']:.3f}; route entropy={metrics['assignment_entropy']:.3f}, "
        f"sharpness={metrics['assignment_sharpness']:.3f}\n"
        f"{'':<10} "
        f"emit mass predicted={metrics['emit_count']:.1f} bases, mean_abs_error={metrics['emit_mass_error']:.1f}, "
        f"splice-site attention={metrics['mean_splice_site_assignment']:.3f}"
    )


def format_failure_report(label: str, metrics: dict[str, float | int]) -> str:
    return (
        f"{label:<10} "
        f"failed examples={metrics['failure_rate']:.3f}; "
        f"failed length={metrics['failed_target_bases_mean']:.1f} bases, "
        f"failed exons={metrics['failed_exon_count_mean']:.2f}, "
        f"failed max intron={metrics['failed_max_intron_mean']:.0f} bp, "
        f"first CDS error={metrics['failed_first_error_mean']:.1f}, "
        f"mean route miss={metrics['pointer_error_distance_mean']:.1f} bp"
    )


def evaluate_model_batch(
    *,
    model: nn.Module,
    batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        list[dict[str, object]],
    ],
    args: argparse.Namespace,
    emit_loss_weight: float,
) -> dict[str, float]:
    dna, splice_tracks, target, target_mask, base_target, base_mask, emit_target, pointer_target, transcript_bases, examples = batch
    amino_acid_probs, transcript_base_probs, assignment, assignment_logits, diagnostics = model(
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
    genome_mask = dna.sum(dim=-1) > 0
    per_emit_loss = F.binary_cross_entropy_with_logits(
        diagnostics["emit_logits"],
        emit_target,
        reduction="none",
    )
    emit_loss = (per_emit_loss * genome_mask.to(per_emit_loss.dtype)).sum() / genome_mask.sum().clamp_min(1)
    loss = (
        aa_loss
        + args.nucleotide_loss_weight * nt_loss
        + emit_loss_weight * emit_loss
        + args.assignment_entropy_weight * diagnostics["assignment_entropy_loss"]
    )

    predicted = amino_acid_probs.argmax(dim=-1)
    predicted_bases = transcript_base_probs.argmax(dim=-1)
    predicted_assignment = assignment_logits.argmax(dim=-1)
    token_correct = ((predicted == target) & target_mask).sum()
    base_correct = ((predicted_bases == base_target) & base_mask).sum()
    assignment_correct = ((predicted_assignment == pointer_target) & base_mask).sum()
    exact = ((predicted == target) | ~target_mask).all(dim=1).float().mean()
    nucleotide_exact = ((predicted_bases == base_target) | ~base_mask).all(dim=1).float().mean()
    assignment_exact = ((predicted_assignment == pointer_target) | ~base_mask).all(dim=1).float().mean()
    aa_confidence = (
        amino_acid_probs.max(dim=-1).values * target_mask.to(amino_acid_probs.dtype)
    ).sum() / target_mask.sum().clamp_min(1)
    nucleotide_confidence = (
        transcript_base_probs.max(dim=-1).values * base_mask.to(transcript_base_probs.dtype)
    ).sum() / base_mask.sum().clamp_min(1)
    assignment_confidence = (
        assignment.max(dim=-1).values * base_mask.to(assignment.dtype)
    ).sum() / base_mask.sum().clamp_min(1)

    stats = summarize_failures(
        predicted_bases=predicted_bases,
        base_target=base_target,
        base_mask=base_mask,
        predicted_pointer=predicted_assignment,
        pointer_target=pointer_target,
        examples=examples,
        junction_window=args.failure_junction_window,
    )
    failure = failure_metrics(stats)
    target_emit_count = base_mask.sum(dim=1).to(diagnostics["emit_count_by_example"].dtype)
    emit_mass_error = (diagnostics["emit_count_by_example"] - target_emit_count).abs().mean()

    return {
        "loss": float(loss.item()),
        "aa_loss": float(aa_loss.item()),
        "nt_loss": float(nt_loss.item()),
        "emit_loss": float(emit_loss.item()),
        "token_accuracy": float(token_correct.item()) / max(1, int(target_mask.sum().item())),
        "exact_match": float(exact.item()),
        "nucleotide_accuracy": float(base_correct.item()) / max(1, int(base_mask.sum().item())),
        "nucleotide_exact_match": float(nucleotide_exact.item()),
        "assignment_accuracy": float(assignment_correct.item()) / max(1, int(base_mask.sum().item())),
        "assignment_exact_match": float(assignment_exact.item()),
        "aa_confidence": float(aa_confidence.item()),
        "nucleotide_confidence": float(nucleotide_confidence.item()),
        "assignment_confidence": float(assignment_confidence.item()),
        "assignment_entropy": float(diagnostics["assignment_entropy"].item()),
        "assignment_sharpness": float(diagnostics["assignment_sharpness"].item()),
        "emit_count": float(diagnostics["emit_count"].item()),
        "emit_mass_error": float(emit_mass_error.item()),
        "mean_emit_probability": float(diagnostics["mean_emit_probability"].item()),
        "mean_splice_site_assignment": float(diagnostics["mean_splice_site_assignment"].item()),
        **failure,
    }


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
    if args.max_exon_bases < args.min_exon_bases:
        raise ValueError("--max-exon-bases must be >= --min-exon-bases")
    if not args.min_exon_bases <= args.median_exon_bases <= args.max_exon_bases:
        raise ValueError("--median-exon-bases must be between --min-exon-bases and --max-exon-bases")
    if not 0 <= args.rare_short_exon_prob <= 1:
        raise ValueError("--rare-short-exon-prob must be in [0, 1]")
    if args.min_intron_length < 4:
        raise ValueError("--min-intron-length must be at least 4")
    if args.max_intron_length < args.min_intron_length:
        raise ValueError("--max-intron-length must be >= --min-intron-length")
    if args.eval_protein_codons < 1:
        raise ValueError("--eval-protein-codons must be at least 1")
    if args.eval_exon_count < 1:
        raise ValueError("--eval-exon-count must be at least 1")
    if args.exon_length_mode == "split" and args.eval_exon_count * args.min_exon_bases > args.eval_protein_codons * 3:
        raise ValueError("--eval-exon-count is too large for --eval-protein-codons and --min-exon-bases")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.validation_batch_size < 0:
        raise ValueError("--validation-batch-size must be non-negative; use 0 to disable fixed validation")
    if args.failure_junction_window < 0:
        raise ValueError("--failure-junction-window must be non-negative")
    if args.micro_batch_size < 0:
        raise ValueError("--micro-batch-size must be non-negative; use 0 for auto")
    if args.max_micro_batch_tokens < 1:
        raise ValueError("--max-micro-batch-tokens must be at least 1")
    if args.length_bucket_size < 0:
        raise ValueError("--length-bucket-size must be non-negative")
    if args.emit_loss_weight < 0:
        raise ValueError("--emit-loss-weight must be non-negative")
    if args.emit_loss_anneal_steps < 0:
        raise ValueError("--emit-loss-anneal-steps must be non-negative")
    if args.assignment_entropy_weight < 0:
        raise ValueError("--assignment-entropy-weight must be non-negative")
    if args.not_coding_weight < 0:
        raise ValueError("--not-coding-weight must be non-negative")
    if args.utr_weight < 0:
        raise ValueError("--utr-weight must be non-negative")
    if args.min_utr5_length < 0 or args.min_utr3_length < 0:
        raise ValueError("minimum UTR lengths must be non-negative")
    if args.max_utr5_length < args.min_utr5_length:
        raise ValueError("--max-utr5-length must be >= --min-utr5-length")
    if args.max_utr3_length < args.min_utr3_length:
        raise ValueError("--max-utr3-length must be >= --min-utr3-length")
    if args.local_conv_kernel < 1 or args.local_conv_kernel % 2 == 0:
        raise ValueError("--local-conv-kernel must be a positive odd integer")
    if args.head_conv_kernel < 1 or args.head_conv_kernel % 2 == 0:
        raise ValueError("--head-conv-kernel must be a positive odd integer")
    if args.attention_layers < 0:
        raise ValueError("--attention-layers must be non-negative")
    if args.attention_heads < 1:
        raise ValueError("--attention-heads must be at least 1")
    if args.hidden_dim % args.attention_heads != 0:
        raise ValueError("--hidden-dim must be divisible by --attention-heads")
    if args.attention_window < 1 or args.attention_window % 2 == 0:
        raise ValueError("--attention-window must be a positive odd integer")
    if args.max_assignment_sharpness <= 0:
        raise ValueError("--max-assignment-sharpness must be positive")
    if not 0 <= args.loss_ema_beta < 1:
        raise ValueError("--loss-ema-beta must be in [0, 1)")
    if not 0 < args.success_lr_decay <= 1:
        raise ValueError("--success-lr-decay must be in (0, 1]")
    if not 0 <= args.success_exact_threshold <= 1:
        raise ValueError("--success-exact-threshold must be in [0, 1]")
    if not 0 <= args.success_nucleotide_exact_threshold <= 1:
        raise ValueError("--success-nucleotide-exact-threshold must be in [0, 1]")
    if args.init_from and (args.resume_from or args.auto_resume):
        raise ValueError("--init-from cannot be combined with --resume-from or --auto-resume")
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
    print(f"Mamba/Triton cache root: {MAMBA_CACHE_ROOT}")
    print(f"model tracks: {MODEL_TRACK_NAMES}")
    print(f"annotation tracks generated: {ANNOTATION_TRACK_NAMES}")
    print("true_transcript_rank is used only for monotonic assignment diagnostics")
    print(
        "training data: "
        f"exons={args.min_exon_count}-{args.max_exon_count}, "
        f"introns={args.min_intron_length}-{args.max_intron_length} bp, "
        f"exon lengths mode={args.exon_length_mode}"
    )
    print(
        "exon length target: "
        f"min={args.min_exon_bases} bp, median~{args.median_exon_bases} bp, "
        f"max={args.max_exon_bases} bp, rare short exon probability={args.rare_short_exon_prob:.3f}"
    )
    print(
        "UTR length target: "
        f"5prime={args.min_utr5_length}-{args.max_utr5_length} bp, "
        f"3prime={args.min_utr3_length}-{args.max_utr3_length} bp"
    )
    print(
        "protein length controls: "
        f"codons={args.min_protein_codons}-{args.max_protein_codons} "
        "(ignored when exon length mode is sampled)"
    )
    effective_micro_batch_size = resolve_micro_batch_size(args)
    micro_batch_mode = "auto" if args.micro_batch_size == 0 else "manual"
    print(
        f"effective batch size: {args.batch_size}; micro batch size: {effective_micro_batch_size}; "
        f"micro batch mode: {micro_batch_mode}; length bucket: {args.length_bucket_size} bp"
    )
    print(
        f"learning rate: base={args.learning_rate}, min={args.min_learning_rate}, "
        f"success_decay={args.success_lr_decay}"
    )
    print(
        "emit/skip mode: prior-masked emit probabilities"
        if args.use_prior_emit_mask
        else "emit/skip mode: unconstrained emit probabilities"
    )
    print(
        "emit loss schedule: "
        f"start={args.emit_loss_weight}, anneal_steps={args.emit_loss_anneal_steps}"
    )

    model = MambaEmitSkipTranslator(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        chunk_size=args.chunk_size,
        headdim=args.headdim,
        use_prior_emit_mask=args.use_prior_emit_mask,
        max_assignment_sharpness=args.max_assignment_sharpness,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    if args.init_from:
        init_path = resolve_checkpoint_path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"Initial checkpoint does not exist: {init_path}")
        load_model_weights(init_path, model=model, device=device)
        print(f"initialized model weights from: {init_path}")

    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir)
    checkpointing_enabled = (
        args.checkpoint_every > 0 or args.checkpoint_keep_every > 0 or args.save_best_checkpoint
    )
    if checkpointing_enabled:
        print(f"checkpoint directory: {checkpoint_dir}")

    validation_batch = None
    if args.validation_batch_size > 0:
        validation_batch = make_batch(
            batch_size=args.validation_batch_size,
            device=device,
            min_protein_codons=args.min_protein_codons,
            max_protein_codons=args.max_protein_codons,
            min_exon_count=args.min_exon_count,
            max_exon_count=args.max_exon_count,
            min_exon_bases=args.min_exon_bases,
            max_exon_bases=args.max_exon_bases,
            median_exon_bases=args.median_exon_bases,
            rare_short_exon_prob=args.rare_short_exon_prob,
            exon_length_mode=args.exon_length_mode,
            min_utr5_length=args.min_utr5_length,
            max_utr5_length=args.max_utr5_length,
            min_utr3_length=args.min_utr3_length,
            max_utr3_length=args.max_utr3_length,
            min_intron_length=args.min_intron_length,
            max_intron_length=args.max_intron_length,
            length_bucket_size=args.length_bucket_size,
            seed=args.validation_seed,
        )
        print(
            f"fixed validation batch: size={args.validation_batch_size}; seed={args.validation_seed}; "
            f"junction window={args.failure_junction_window} transcript bases"
        )

    start_step = 0
    best_loss = float("inf")
    final_metrics = None
    loss_ema = None
    success_lr_multiplier = 1.0
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
        if final_metrics is not None:
            success_lr_multiplier = float(final_metrics.get("next_lr_multiplier", success_lr_multiplier))
            if (
                final_metrics.get("exact_match", 0.0) >= args.success_exact_threshold
                and final_metrics.get("nucleotide_exact_match", 0.0) >= args.success_nucleotide_exact_threshold
            ):
                success_lr_multiplier = min(success_lr_multiplier, args.success_lr_decay)

    if start_step >= args.steps:
        print(f"checkpoint has already reached requested --steps={args.steps}; skipping training loop")

    for step in range(start_step, args.steps):
        scheduled_lr = get_lr(
            step=step,
            total_steps=args.steps,
            base_lr=args.learning_rate,
            min_lr=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
        )
        current_lr_multiplier = success_lr_multiplier
        lr = scheduled_lr * current_lr_multiplier
        for group in optimizer.param_groups:
            group["lr"] = lr
        emit_loss_weight = get_linear_annealed_weight(
            step=step,
            start_weight=args.emit_loss_weight,
            end_step=args.emit_loss_anneal_steps,
        )

        optimizer.zero_grad(set_to_none=True)
        micro_batch_count = math.ceil(args.batch_size / effective_micro_batch_size)
        total_examples = 0
        loss_sum = 0.0
        aa_loss_sum = 0.0
        nt_loss_sum = 0.0
        emit_loss_sum = 0.0
        token_correct = 0
        token_total = 0
        exact_sum = 0.0
        nucleotide_correct = 0
        nucleotide_total = 0
        nucleotide_exact_sum = 0.0
        assignment_correct_total = 0
        assignment_total = 0
        assignment_exact_sum = 0.0
        target_lengths_all = []
        genome_lengths = []
        intron_lengths = []
        utr5_lengths = []
        utr3_lengths = []
        assignment_entropy_sum = 0.0
        aa_confidence_sum = 0.0
        nucleotide_confidence_sum = 0.0
        assignment_confidence_sum = 0.0
        assignment_sharpness_sum = 0.0
        emit_count_sum = 0.0
        emit_mass_error_sum = 0.0
        mean_emit_probability_sum = 0.0
        mean_splice_site_assignment_sum = 0.0
        train_failure_stats = empty_failure_stats()

        for micro_batch_index in range(micro_batch_count):
            micro_batch_size = min(
                effective_micro_batch_size,
                args.batch_size - micro_batch_index * effective_micro_batch_size,
            )
            dna, splice_tracks, target, target_mask, base_target, base_mask, emit_target, pointer_target, transcript_bases, examples = make_batch(
                batch_size=micro_batch_size,
                device=device,
                min_protein_codons=args.min_protein_codons,
                max_protein_codons=args.max_protein_codons,
                min_exon_count=args.min_exon_count,
                max_exon_count=args.max_exon_count,
                min_exon_bases=args.min_exon_bases,
                max_exon_bases=args.max_exon_bases,
                median_exon_bases=args.median_exon_bases,
                rare_short_exon_prob=args.rare_short_exon_prob,
                exon_length_mode=args.exon_length_mode,
                min_intron_length=args.min_intron_length,
                max_intron_length=args.max_intron_length,
                length_bucket_size=args.length_bucket_size,
                seed=args.batch_seed_offset + step * micro_batch_count + micro_batch_index,
            )
            amino_acid_probs, transcript_base_probs, assignment, assignment_logits, diagnostics = model(
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
            genome_mask = dna.sum(dim=-1) > 0
            per_emit_loss = F.binary_cross_entropy_with_logits(
                diagnostics["emit_logits"],
                emit_target,
                reduction="none",
            )
            emit_loss = (per_emit_loss * genome_mask.to(per_emit_loss.dtype)).sum() / genome_mask.sum().clamp_min(1)
            entropy_loss = args.assignment_entropy_weight * diagnostics["assignment_entropy_loss"]
            loss = (
                aa_loss
                + args.nucleotide_loss_weight * nt_loss
                + emit_loss_weight * emit_loss
                + entropy_loss
            )
            loss_scale = float(micro_batch_size) / float(args.batch_size)
            (loss * loss_scale).backward()

            with torch.no_grad():
                predicted = amino_acid_probs.argmax(dim=-1)
                correct = (predicted == target) & target_mask
                aa_confidence = amino_acid_probs.max(dim=-1).values
                predicted_bases = transcript_base_probs.argmax(dim=-1)
                base_correct = (predicted_bases == base_target) & base_mask
                nucleotide_confidence = transcript_base_probs.max(dim=-1).values
                predicted_assignment = assignment_logits.argmax(dim=-1)
                assignment_correct = (predicted_assignment == pointer_target) & base_mask
                assignment_confidence = assignment.max(dim=-1).values
                target_lengths = target_mask.sum(dim=1)

                total_examples += micro_batch_size
                loss_sum += loss.item() * micro_batch_size
                aa_loss_sum += aa_loss.item() * micro_batch_size
                nt_loss_sum += nt_loss.item() * micro_batch_size
                emit_loss_sum += emit_loss.item() * micro_batch_size
                token_correct += int(correct.sum().item())
                token_total += int(target_mask.sum().item())
                exact_sum += float(((predicted == target) | ~target_mask).all(dim=1).float().sum().item())
                nucleotide_correct += int(base_correct.sum().item())
                nucleotide_total += int(base_mask.sum().item())
                nucleotide_exact_sum += float(((predicted_bases == base_target) | ~base_mask).all(dim=1).float().sum().item())
                assignment_correct_total += int(assignment_correct.sum().item())
                assignment_total += int(base_mask.sum().item())
                assignment_exact_sum += float(((predicted_assignment == pointer_target) | ~base_mask).all(dim=1).float().sum().item())
                merge_failure_stats(
                    train_failure_stats,
                    summarize_failures(
                        predicted_bases=predicted_bases,
                        base_target=base_target,
                        base_mask=base_mask,
                        predicted_pointer=predicted_assignment,
                        pointer_target=pointer_target,
                        examples=examples,
                        junction_window=args.failure_junction_window,
                    ),
                )
                aa_confidence_sum += float(
                    (aa_confidence * target_mask.to(aa_confidence.dtype)).sum().item()
                )
                nucleotide_confidence_sum += float(
                    (nucleotide_confidence * base_mask.to(nucleotide_confidence.dtype)).sum().item()
                )
                assignment_confidence_sum += float(
                    (assignment_confidence * base_mask.to(assignment_confidence.dtype)).sum().item()
                )
                target_lengths_all.extend(int(length) for length in target_lengths.tolist())
                genome_lengths.extend(len(str(example["genome"])) for example in examples)
                intron_lengths.extend(
                    int(intron_length)
                    for example in examples
                    for intron_length in example["intron_lengths"]
                )
                assignment_entropy_sum += diagnostics["assignment_entropy"].item() * micro_batch_size
                assignment_sharpness_sum += diagnostics["assignment_sharpness"].item() * micro_batch_size
                emit_count_sum += diagnostics["emit_count"].item() * micro_batch_size
                target_emit_count = base_mask.sum(dim=1).to(diagnostics["emit_count_by_example"].dtype)
                emit_mass_error = (diagnostics["emit_count_by_example"] - target_emit_count).abs().mean()
                emit_mass_error_sum += emit_mass_error.item() * micro_batch_size
                mean_emit_probability_sum += diagnostics["mean_emit_probability"].item() * micro_batch_size
                mean_splice_site_assignment_sum += diagnostics["mean_splice_site_assignment"].item() * micro_batch_size

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = loss_sum / max(1, total_examples)
        aa_loss_value = aa_loss_sum / max(1, total_examples)
        nt_loss_value = nt_loss_sum / max(1, total_examples)
        emit_loss_value = emit_loss_sum / max(1, total_examples)
        token_accuracy = token_correct / max(1, token_total)
        exact = exact_sum / max(1, total_examples)
        nucleotide_accuracy = nucleotide_correct / max(1, nucleotide_total)
        nucleotide_exact = nucleotide_exact_sum / max(1, total_examples)
        assignment_accuracy = assignment_correct_total / max(1, assignment_total)
        assignment_exact = assignment_exact_sum / max(1, total_examples)
        aa_confidence = aa_confidence_sum / max(1, token_total)
        nucleotide_confidence = nucleotide_confidence_sum / max(1, nucleotide_total)
        assignment_confidence = assignment_confidence_sum / max(1, assignment_total)
        mean_target_length = sum(target_lengths_all) / max(1, len(target_lengths_all))
        max_target_length = max(target_lengths_all) if target_lengths_all else 0
        mean_genome_length = sum(genome_lengths) / max(1, len(genome_lengths))
        max_genome_length = max(genome_lengths) if genome_lengths else 0
        mean_intron_length = sum(intron_lengths) / len(intron_lengths) if intron_lengths else 0.0
        max_intron_length = max(intron_lengths) if intron_lengths else 0
        assignment_entropy = assignment_entropy_sum / max(1, total_examples)
        assignment_sharpness = assignment_sharpness_sum / max(1, total_examples)
        emit_count = emit_count_sum / max(1, total_examples)
        emit_mass_error = emit_mass_error_sum / max(1, total_examples)
        mean_emit_probability = mean_emit_probability_sum / max(1, total_examples)
        mean_splice_site_assignment = mean_splice_site_assignment_sum / max(1, total_examples)
        train_failure = failure_metrics(train_failure_stats)
        loss_ema = loss_value if loss_ema is None else args.loss_ema_beta * loss_ema + (1.0 - args.loss_ema_beta) * loss_value
        if (
            success_lr_multiplier > args.success_lr_decay
            and exact >= args.success_exact_threshold
            and nucleotide_exact >= args.success_nucleotide_exact_threshold
        ):
            success_lr_multiplier = args.success_lr_decay
            print(
                f"success lr brake armed: future lr multiplier={success_lr_multiplier:.3f}",
                flush=True,
            )

        final_metrics = {
            "step": step,
            "scheduled_lr": scheduled_lr,
            "lr_multiplier": current_lr_multiplier,
            "next_lr_multiplier": success_lr_multiplier,
            "lr": lr,
            "loss": loss_value,
            "loss_ema": loss_ema,
            "aa_loss": aa_loss_value,
            "nt_loss": nt_loss_value,
            "emit_loss": emit_loss_value,
            "emit_loss_weight": emit_loss_weight,
            "micro_batch_size": effective_micro_batch_size,
            "length_bucket_size": args.length_bucket_size,
            "token_accuracy": token_accuracy,
            "exact_match": exact,
            "nucleotide_accuracy": nucleotide_accuracy,
            "nucleotide_exact_match": nucleotide_exact,
            "assignment_accuracy": assignment_accuracy,
            "assignment_exact_match": assignment_exact,
            "aa_confidence": aa_confidence,
            "nucleotide_confidence": nucleotide_confidence,
            "assignment_confidence": assignment_confidence,
            "mean_target_length": mean_target_length,
            "max_target_length": max_target_length,
            "mean_genome_length": mean_genome_length,
            "max_genome_length": max_genome_length,
            "mean_intron_length": mean_intron_length,
            "max_intron_length": max_intron_length,
            "assignment_entropy": assignment_entropy,
            "assignment_sharpness": assignment_sharpness,
            "emit_count": emit_count,
            "emit_mass_error": emit_mass_error,
            "mean_emit_probability": mean_emit_probability,
            "mean_splice_site_assignment": mean_splice_site_assignment,
            "failure_rate": train_failure["failure_rate"],
            "failed_target_bases_mean": train_failure["failed_target_bases_mean"],
            "failed_exon_count_mean": train_failure["failed_exon_count_mean"],
            "failed_max_intron_mean": train_failure["failed_max_intron_mean"],
            "failed_first_error_mean": train_failure["failed_first_error_mean"],
            "pointer_error_distance_mean": train_failure["pointer_error_distance_mean"],
            "junction_pointer_accuracy": train_failure["junction_pointer_accuracy"],
            "junction_pointer_total": train_failure["junction_pointer_total"],
        }

        is_final_step = step == args.steps - 1
        is_report_step = step % args.print_every == 0 or is_final_step
        validation_metrics = None
        if is_report_step and validation_batch is not None:
            model.eval()
            with torch.no_grad():
                validation_metrics = evaluate_model_batch(
                    model=model,
                    batch=validation_batch,
                    args=args,
                    emit_loss_weight=emit_loss_weight,
                )
            model.train()
            final_metrics.update({f"validation_{key}": value for key, value in validation_metrics.items()})

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
            report_lines = [
                (
                    f"\nStep {step:06d} | learning_rate={lr:.2e} "
                    f"(schedule_mult={current_lr_multiplier:.3f}) | smoothed_loss={loss_ema:.3f}"
                ),
                (
                    f"Batch shape | proteins mean/max={mean_target_length:.1f}/{max_target_length} aa, "
                    f"genomes mean/max={mean_genome_length:.0f}/{max_genome_length} bp, "
                    f"introns mean/max={mean_intron_length:.0f}/{max_intron_length} bp, "
                    f"microbatch={effective_micro_batch_size}/{args.batch_size}, emit_loss_weight={emit_loss_weight:.3f}"
                ),
                format_report_metrics("train", final_metrics),
                format_failure_report("train errors", final_metrics),
            ]
            if validation_metrics is not None:
                report_lines.extend(
                    [
                        format_report_metrics("validation", validation_metrics),
                        format_failure_report("val errors", validation_metrics),
                    ]
                )
            print("\n".join(report_lines), flush=True)

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

    dna, splice_tracks, _target, target_mask, base_target, base_mask, _emit_target, _pointer_target, transcript_bases, examples = make_batch(
        batch_size=1,
        device=device,
        min_protein_codons=args.eval_protein_codons,
        max_protein_codons=args.eval_protein_codons,
        min_exon_count=args.eval_exon_count,
        max_exon_count=args.eval_exon_count,
        min_exon_bases=args.min_exon_bases,
        max_exon_bases=args.max_exon_bases,
        median_exon_bases=args.median_exon_bases,
        rare_short_exon_prob=args.rare_short_exon_prob,
        exon_length_mode=args.exon_length_mode,
        min_intron_length=args.min_intron_length,
        max_intron_length=args.max_intron_length,
        length_bucket_size=args.length_bucket_size,
        seed=args.eval_seed,
    )
    model.eval()
    with torch.no_grad():
        amino_acid_probs, transcript_base_probs, assignment, _assignment_logits, diagnostics = model(
            dna,
            splice_tracks,
            transcript_bases=transcript_bases,
        )
    eval_length = int(target_mask[0].sum().item())
    eval_base_length = int(base_mask[0].sum().item())
    prediction = "".join(AMINO_ACIDS[index] for index in amino_acid_probs.argmax(dim=-1)[0, :eval_length].tolist())
    predicted_cds = "".join(DNA_BASES[index] for index in transcript_base_probs.argmax(dim=-1)[0, :eval_base_length].tolist())
    eval_aa_confidence = amino_acid_probs.max(dim=-1).values[0, :eval_length].mean().item()
    eval_nucleotide_confidence = transcript_base_probs.max(dim=-1).values[0, :eval_base_length].mean().item()
    eval_assignment_confidence = assignment.max(dim=-1).values[0, :eval_base_length].mean().item()
    eval_emit_mass_error = (
        diagnostics["emit_count_by_example"][0] - base_mask.sum(dim=1).to(diagnostics["emit_count_by_example"].dtype)[0]
    ).abs().item()

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
    print("assignment entropy:", diagnostics["assignment_entropy"].item())
    print("amino-acid confidence:", eval_aa_confidence)
    print("nucleotide confidence:", eval_nucleotide_confidence)
    print("assignment confidence:", eval_assignment_confidence)
    print("emit count:", diagnostics["emit_count"].item())
    print("emit mass error:", eval_emit_mass_error)
    print("assignment sharpness:", diagnostics["assignment_sharpness"].item())
    print("mean emit probability:", diagnostics["mean_emit_probability"].item())
    print("mean splice-site assignment:", diagnostics["mean_splice_site_assignment"].item())


PHASE_LABELS = (
    "exon_phase_0",
    "exon_phase_1",
    "exon_phase_2",
    "intron_carry_0",
    "intron_carry_1",
    "intron_carry_2",
    "utr_5",
    "utr_3",
)
DEFAULT_INTRON_CARRY_INDEX = 3
UTR5_INDEX = 6
UTR3_INDEX = 7


class MambaPhaseTrackPredictor(nn.Module):
    """Genome-registered CDS phase predictor.

    The output has shape B x L x 8 and stays aligned to the input DNA bases:
    exon phase 0/1/2, intron carrying the next exon phase 0/1/2, 5' UTR, or 3' UTR.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        layers: int = 3,
        chunk_size: int = 32,
        headdim: int = 8,
        local_conv_kernel: int = 9,
        head_conv_kernel: int = 7,
        bidirectional: bool = False,
        attention_layers: int = 2,
        attention_heads: int = 4,
        attention_window: int = 129,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.local_context = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=local_conv_kernel,
                padding=local_conv_kernel // 2,
                groups=hidden_dim,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.scan_blocks = OfficialMamba2Encoder(
            hidden_dim=hidden_dim,
            layers=layers,
            chunk_size=chunk_size,
            headdim=headdim,
        )
        if self.bidirectional:
            self.reverse_scan_blocks = OfficialMamba2Encoder(
                hidden_dim=hidden_dim,
                layers=layers,
                chunk_size=chunk_size,
                headdim=headdim,
            )
            self.bidirectional_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_blocks = nn.ModuleList(
            [
                LocalWindowAttentionBlock(
                    hidden_dim=hidden_dim,
                    heads=attention_heads,
                    window_size=attention_window,
                )
                for _ in range(attention_layers)
            ]
        )
        self.phase_norm = nn.LayerNorm(hidden_dim)
        self.phase_head = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=head_conv_kernel,
                padding=head_conv_kernel // 2,
                groups=hidden_dim,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, len(PHASE_LABELS), kernel_size=1),
        )

    def seed_reverse_from_forward(self) -> None:
        if self.bidirectional:
            self.reverse_scan_blocks.load_state_dict(self.scan_blocks.state_dict())

    @staticmethod
    def _reverse_valid_prefix(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_length, channels = x.shape
        positions = torch.arange(max_length, device=x.device)
        lengths = valid_mask.long().sum(dim=1)
        reverse_positions = lengths[:, None] - 1 - positions[None, :]
        gather_positions = reverse_positions.clamp(min=0, max=max_length - 1)
        gather_positions = gather_positions.unsqueeze(-1).expand(batch_size, max_length, channels)
        reversed_x = torch.gather(x, dim=1, index=gather_positions)
        prefix_mask = positions[None, :] < lengths[:, None]
        return reversed_x.masked_fill(~prefix_mask.unsqueeze(-1), 0.0)

    def forward(self, dna_one_hot: torch.Tensor, splice_tracks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([dna_one_hot, splice_tracks], dim=-1)
        encoded = self.input_projection(features)
        local_features = self.local_norm(encoded).transpose(1, 2)
        encoded = encoded + self.local_context(local_features).transpose(1, 2)
        forward_encoded = self.scan_blocks(encoded)
        valid_mask = dna_one_hot.sum(dim=-1) > 0
        if self.bidirectional:
            reverse_input = self._reverse_valid_prefix(encoded, valid_mask)
            reverse_encoded = self.reverse_scan_blocks(reverse_input)
            reverse_encoded = self._reverse_valid_prefix(reverse_encoded, valid_mask)
            encoded = self.bidirectional_fusion(torch.cat([forward_encoded, reverse_encoded], dim=-1))
        else:
            encoded = forward_encoded
        for attention_block in self.attention_blocks:
            encoded = attention_block(encoded, valid_mask=valid_mask)
        phase_features = self.phase_norm(encoded).transpose(1, 2)
        phase_logits = self.phase_head(phase_features).transpose(1, 2)
        return phase_logits, encoded


def phase_targets_from_examples(
    examples: list[dict[str, object]],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    mask_rows = []
    for example in examples:
        annotations = example["annotations"]
        target = torch.full((max_length,), DEFAULT_INTRON_CARRY_INDEX, dtype=torch.long)
        genome_mask = torch.zeros(max_length, dtype=torch.bool)
        genome_length = annotations.shape[0]
        genome_mask[:genome_length] = True
        transcript_rank = annotations[:, 3]
        region_code = annotations[:, 4] if annotations.shape[1] > 4 else torch.where(
            transcript_rank >= 0,
            torch.full_like(transcript_rank, REGION_CDS),
            torch.full_like(transcript_rank, REGION_INTRON),
        )
        coding_mask = transcript_rank >= 0
        intron_mask = region_code == REGION_INTRON
        target[:genome_length][region_code == REGION_UTR5] = UTR5_INDEX
        target[:genome_length][region_code == REGION_UTR3] = UTR3_INDEX
        coding_indices = torch.nonzero(coding_mask, as_tuple=False).flatten()
        if coding_indices.numel() > 0:
            coding_phase = transcript_rank[coding_mask].long() % 3
            target[:genome_length][coding_mask] = coding_phase
            first_coding_index = int(coding_indices[0].item())
            last_seen_next_phase = 0
            if first_coding_index > 0:
                prefix_mask = intron_mask[:first_coding_index]
                target[:first_coding_index][prefix_mask] = 3
            previous_index = None
            for genomic_index, phase in zip(coding_indices.tolist(), coding_phase.tolist()):
                if previous_index is not None and genomic_index > previous_index + 1:
                    gap = slice(previous_index + 1, genomic_index)
                    gap_intron_mask = intron_mask[gap]
                    target[gap][gap_intron_mask] = 3 + last_seen_next_phase
                last_seen_next_phase = (int(phase) + 1) % 3
                previous_index = int(genomic_index)
            if previous_index is not None and previous_index + 1 < genome_length:
                suffix = slice(previous_index + 1, genome_length)
                suffix_intron_mask = intron_mask[suffix]
                target[suffix][suffix_intron_mask] = 3
        rows.append(target)
        mask_rows.append(genome_mask)
    return torch.stack(rows).to(device), torch.stack(mask_rows).to(device)


def phase_metrics(logits: torch.Tensor, target: torch.Tensor, genome_mask: torch.Tensor) -> dict[str, float]:
    predicted = logits.argmax(dim=-1)
    coding_mask = genome_mask & (target < 3)
    intron_carry_mask = genome_mask & (target >= 3) & (target < 6)
    utr5_mask = genome_mask & (target == UTR5_INDEX)
    utr3_mask = genome_mask & (target == UTR3_INDEX)
    noncoding_mask = genome_mask & (target >= 3)
    correct = (predicted == target) & genome_mask
    coding_correct = (predicted == target) & coding_mask
    intron_carry_correct = (predicted == target) & intron_carry_mask
    utr5_correct = (predicted == target) & utr5_mask
    utr3_correct = (predicted == target) & utr3_mask
    collapsed_predicted = torch.where(predicted < 3, predicted, torch.full_like(predicted, 3))
    collapsed_target = torch.where(target < 3, target, torch.full_like(target, 3))
    collapsed_correct = (collapsed_predicted == collapsed_target) & genome_mask
    collapsed_noncoding_correct = collapsed_correct & noncoding_mask
    exact = ((predicted == target) | ~genome_mask).all(dim=1).float().mean()
    coding_exact = ((predicted == target) | ~coding_mask).all(dim=1).float().mean()
    confidence = logits.softmax(dim=-1).max(dim=-1).values
    phase_metrics_by_class = {}
    for phase_index in range(3):
        phase_mask = coding_mask & (target == phase_index)
        phase_correct = (predicted == target) & phase_mask
        phase_metrics_by_class[f"exon_phase_{phase_index}_accuracy"] = (
            float(phase_correct.sum().item()) / max(1, int(phase_mask.sum().item()))
        )
        carry_mask = intron_carry_mask & (target == phase_index + 3)
        carry_correct = (predicted == target) & carry_mask
        phase_metrics_by_class[f"intron_carry_{phase_index}_accuracy"] = (
            float(carry_correct.sum().item()) / max(1, int(carry_mask.sum().item()))
        )
    return {
        "accuracy": float(correct.sum().item()) / max(1, int(genome_mask.sum().item())),
        "collapsed_accuracy": float(collapsed_correct.sum().item()) / max(1, int(genome_mask.sum().item())),
        "coding_phase_accuracy": float(coding_correct.sum().item()) / max(1, int(coding_mask.sum().item())),
        "exon_phase_accuracy": float(coding_correct.sum().item()) / max(1, int(coding_mask.sum().item())),
        "intron_carry_accuracy": float(intron_carry_correct.sum().item()) / max(1, int(intron_carry_mask.sum().item())),
        "utr5_accuracy": float(utr5_correct.sum().item()) / max(1, int(utr5_mask.sum().item())),
        "utr3_accuracy": float(utr3_correct.sum().item()) / max(1, int(utr3_mask.sum().item())),
        "not_coding_accuracy": float(collapsed_noncoding_correct.sum().item()) / max(1, int(noncoding_mask.sum().item())),
        "exact_match": float(exact.item()),
        "coding_exact_match": float(coding_exact.item()),
        "confidence": float((confidence * genome_mask.to(confidence.dtype)).sum().item())
        / max(1, int(genome_mask.sum().item())),
        **phase_metrics_by_class,
    }


def format_phase_report(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label:<10} loss={metrics['loss']:.4f}\n"
        f"{'':<10} genome exact={metrics['exact_match']:.3f}, "
        f"8-state accuracy={metrics['accuracy']:.3f}, collapsed 4-track={metrics['collapsed_accuracy']:.3f}\n"
        f"{'':<10} exon phase exact={metrics['coding_exact_match']:.3f}, "
        f"exon phase base={metrics['exon_phase_accuracy']:.3f}, "
        f"phase0={metrics['exon_phase_0_accuracy']:.3f}, "
        f"phase1={metrics['exon_phase_1_accuracy']:.3f}, "
        f"phase2={metrics['exon_phase_2_accuracy']:.3f}, "
        f"not-coding collapsed={metrics['not_coding_accuracy']:.3f}\n"
        f"{'':<10} intron carry base={metrics['intron_carry_accuracy']:.3f}, "
        f"carry0={metrics['intron_carry_0_accuracy']:.3f}, "
        f"carry1={metrics['intron_carry_1_accuracy']:.3f}, "
        f"carry2={metrics['intron_carry_2_accuracy']:.3f}\n"
        f"{'':<10} UTR base 5prime={metrics['utr5_accuracy']:.3f}, 3prime={metrics['utr3_accuracy']:.3f}\n"
        f"{'':<10} confidence={metrics['confidence']:.3f}"
    )


def timed(message: str, start_time: float) -> None:
    print(f"{message} ({time.perf_counter() - start_time:.1f}s)", flush=True)


def load_matching_model_weights(path: Path, *, model: nn.Module, device: torch.device) -> set[str]:
    start_time = time.perf_counter()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    timed("checkpoint loaded on CPU", start_time)
    source_state = checkpoint["model_state_dict"]
    target_state = model.state_dict()
    adapted_state = dict(target_state)
    exact_count = 0
    partial_count = 0
    partial_names = []
    for key, source_value in source_state.items():
        if key not in target_state:
            continue
        target_value = target_state[key]
        if tuple(source_value.shape) == tuple(target_value.shape):
            adapted_state[key] = source_value
            exact_count += 1
            continue
        if source_value.ndim != target_value.ndim:
            continue
        if any(source_dim <= 0 or target_dim <= 0 for source_dim, target_dim in zip(source_value.shape, target_value.shape)):
            continue
        copied_value = target_value.clone()
        if key == "input_projection.weight" and target_value.shape[1] > source_value.shape[1]:
            copied_value[:, source_value.shape[1] :] = 0.0
        slices = tuple(slice(0, min(source_dim, target_dim)) for source_dim, target_dim in zip(source_value.shape, target_value.shape))
        copied_value[slices] = source_value[slices]
        adapted_state[key] = copied_value
        partial_count += 1
        partial_names.append(key)
    copy_start = time.perf_counter()
    model.load_state_dict(adapted_state)
    timed(f"copied {exact_count} exact tensors and {partial_count} partial tensors into phase model", copy_start)
    if partial_names:
        print("partially initialized widened tensors: " + ", ".join(partial_names), flush=True)
    print(f"loaded compatible tensors from: {path}", flush=True)
    return set(source_state)


def slice_training_batch(
    batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        list[dict[str, object]],
    ],
    start: int,
    end: int,
) -> tuple[
    torch.Tensor,
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
    dna, splice_tracks, target, target_mask, base_target, base_mask, emit_target, pointer_target, transcript_bases, examples = batch
    return (
        dna[start:end],
        splice_tracks[start:end],
        target[start:end],
        target_mask[start:end],
        base_target[start:end],
        base_mask[start:end],
        emit_target[start:end],
        pointer_target[start:end],
        transcript_bases,
        examples[start:end],
    )


def train_phase(args: argparse.Namespace) -> None:
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
    print(f"Mamba/Triton cache root: {MAMBA_CACHE_ROOT}")
    print("task: genome-aligned carried-phase track")
    print(f"output classes: {PHASE_LABELS}")
    print(
        "training data: "
        f"exons={args.min_exon_count}-{args.max_exon_count}, "
        f"introns={args.min_intron_length}-{args.max_intron_length} bp, "
        f"exon lengths mode={args.exon_length_mode}"
    )
    print(
        "exon length target: "
        f"min={args.min_exon_bases} bp, median~{args.median_exon_bases} bp, "
        f"max={args.max_exon_bases} bp, rare short exon probability={args.rare_short_exon_prob:.3f}"
    )
    print(
        "UTR length target: "
        f"5prime={args.min_utr5_length}-{args.max_utr5_length} bp, "
        f"3prime={args.min_utr3_length}-{args.max_utr3_length} bp"
    )
    print(
        "architecture: "
        f"local conv + {'bidirectional' if args.bidirectional else 'forward-only'} Mamba2 "
        f"+ {args.attention_layers} local attention blocks, "
        f"attention_heads={args.attention_heads}, attention_window={args.attention_window}, "
        f"local_conv={args.local_conv_kernel}, head_conv={args.head_conv_kernel}"
    )

    stage_start = time.perf_counter()
    print("building phase Mamba model...", flush=True)
    model = MambaPhaseTrackPredictor(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        chunk_size=args.chunk_size,
        headdim=args.headdim,
        local_conv_kernel=args.local_conv_kernel,
        head_conv_kernel=args.head_conv_kernel,
        bidirectional=args.bidirectional,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
        attention_window=args.attention_window,
    ).to(device)
    timed("model is on device", stage_start)
    stage_start = time.perf_counter()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    timed("optimizer built", stage_start)
    if args.init_from:
        init_path = resolve_checkpoint_path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"Initial checkpoint does not exist: {init_path}")
        print(f"loading compatible weights from: {init_path}", flush=True)
        loaded_weight_names = load_matching_model_weights(init_path, model=model, device=device)
        if args.bidirectional:
            if any(name.startswith("reverse_scan_blocks.") for name in loaded_weight_names):
                print("loaded reverse Mamba blocks from checkpoint", flush=True)
            else:
                model.seed_reverse_from_forward()
                print("seeded reverse Mamba blocks from loaded forward blocks", flush=True)

    checkpoint_dir = resolve_checkpoint_path(args.checkpoint_dir)
    print(f"checkpoint directory: {checkpoint_dir}")
    effective_micro_batch_size = resolve_micro_batch_size(args)
    print(f"effective batch size: {args.batch_size}; microbatch={effective_micro_batch_size}/{args.batch_size}")

    validation_batch = None
    if args.validation_batch_size > 0:
        stage_start = time.perf_counter()
        print(f"building fixed validation batch with {args.validation_batch_size} examples...", flush=True)
        validation_batch = make_batch(
            batch_size=args.validation_batch_size,
            device=device,
            min_protein_codons=args.min_protein_codons,
            max_protein_codons=args.max_protein_codons,
            min_exon_count=args.min_exon_count,
            max_exon_count=args.max_exon_count,
            min_exon_bases=args.min_exon_bases,
            max_exon_bases=args.max_exon_bases,
            median_exon_bases=args.median_exon_bases,
            rare_short_exon_prob=args.rare_short_exon_prob,
            exon_length_mode=args.exon_length_mode,
            min_utr5_length=args.min_utr5_length,
            max_utr5_length=args.max_utr5_length,
            min_utr3_length=args.min_utr3_length,
            max_utr3_length=args.max_utr3_length,
            min_intron_length=args.min_intron_length,
            max_intron_length=args.max_intron_length,
            length_bucket_size=args.length_bucket_size,
            seed=args.validation_seed,
        )
        timed("fixed validation batch ready", stage_start)
    else:
        print("fixed validation disabled", flush=True)

    class_weights = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            args.not_coding_weight,
            args.not_coding_weight,
            args.not_coding_weight,
            args.utr_weight,
            args.utr_weight,
        ],
        dtype=torch.float32,
        device=device,
    )
    best_loss = float("inf")
    loss_ema = None
    final_metrics = None

    for step in range(args.steps):
        if step == 0:
            print("starting training loop; first Mamba call may compile Triton kernels", flush=True)
        scheduled_lr = get_lr(
            step=step,
            total_steps=args.steps,
            base_lr=args.learning_rate,
            min_lr=args.min_learning_rate,
            warmup_steps=args.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr

        optimizer.zero_grad(set_to_none=True)
        micro_batch_count = math.ceil(args.batch_size / effective_micro_batch_size)
        total_examples = 0
        loss_sum = 0.0
        metric_sums: dict[str, float] = {}
        target_lengths_all = []
        genome_lengths = []
        intron_lengths = []
        full_batch = make_batch(
            batch_size=args.batch_size,
            device=device,
            min_protein_codons=args.min_protein_codons,
            max_protein_codons=args.max_protein_codons,
            min_exon_count=args.min_exon_count,
            max_exon_count=args.max_exon_count,
            min_exon_bases=args.min_exon_bases,
            max_exon_bases=args.max_exon_bases,
            median_exon_bases=args.median_exon_bases,
            rare_short_exon_prob=args.rare_short_exon_prob,
            exon_length_mode=args.exon_length_mode,
            min_utr5_length=args.min_utr5_length,
            max_utr5_length=args.max_utr5_length,
            min_utr3_length=args.min_utr3_length,
            max_utr3_length=args.max_utr3_length,
            min_intron_length=args.min_intron_length,
            max_intron_length=args.max_intron_length,
            length_bucket_size=args.length_bucket_size,
            seed=args.batch_seed_offset + step,
        )

        for micro_batch_index in range(micro_batch_count):
            micro_batch_size = min(
                effective_micro_batch_size,
                args.batch_size - micro_batch_index * effective_micro_batch_size,
            )
            start_index = micro_batch_index * effective_micro_batch_size
            end_index = start_index + micro_batch_size
            dna, splice_tracks, _target, _target_mask, _base_target, _base_mask, _emit_target, _pointer_target, _transcript_bases, examples = slice_training_batch(
                full_batch,
                start_index,
                end_index,
            )
            phase_target, genome_mask = phase_targets_from_examples(
                examples,
                max_length=dna.shape[1],
                device=device,
            )
            first_forward_start = time.perf_counter() if step == 0 and micro_batch_index == 0 else None
            logits, _encoded = model(dna, splice_tracks)
            if first_forward_start is not None:
                timed("first Mamba forward complete", first_forward_start)
            per_base_loss = F.cross_entropy(
                logits.reshape(-1, len(PHASE_LABELS)),
                phase_target.reshape(-1),
                weight=class_weights,
                reduction="none",
            ).reshape_as(phase_target)
            loss = (per_base_loss * genome_mask.to(per_base_loss.dtype)).sum() / genome_mask.sum().clamp_min(1)
            loss_scale = float(micro_batch_size) / float(args.batch_size)
            first_backward_start = time.perf_counter() if step == 0 and micro_batch_index == 0 else None
            (loss * loss_scale).backward()
            if first_backward_start is not None:
                timed("first Mamba backward complete", first_backward_start)

            with torch.no_grad():
                metrics = phase_metrics(logits, phase_target, genome_mask)
                loss_sum += loss.item() * micro_batch_size
                for key in metrics:
                    metric_sums.setdefault(key, 0.0)
                    metric_sums[key] += metrics[key] * micro_batch_size
                total_examples += micro_batch_size
                target_lengths_all.extend(int(example["protein_codons"]) for example in examples)
                genome_lengths.extend(len(str(example["genome"])) for example in examples)
                utr5_lengths.extend(int(example["utr5_length"]) for example in examples)
                utr3_lengths.extend(int(example["utr3_length"]) for example in examples)
                intron_lengths.extend(
                    int(intron_length)
                    for example in examples
                    for intron_length in example["intron_lengths"]
                )

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_value = loss_sum / max(1, total_examples)
        loss_ema = loss_value if loss_ema is None else args.loss_ema_beta * loss_ema + (1.0 - args.loss_ema_beta) * loss_value
        final_metrics = {
            "loss": loss_value,
            "lr": scheduled_lr,
            "mean_target_length": sum(target_lengths_all) / max(1, len(target_lengths_all)),
            "max_target_length": max(target_lengths_all) if target_lengths_all else 0,
            "mean_genome_length": sum(genome_lengths) / max(1, len(genome_lengths)),
            "max_genome_length": max(genome_lengths) if genome_lengths else 0,
            "mean_intron_length": sum(intron_lengths) / len(intron_lengths) if intron_lengths else 0.0,
            "max_intron_length": max(intron_lengths) if intron_lengths else 0,
            "mean_utr5_length": sum(utr5_lengths) / max(1, len(utr5_lengths)),
            "max_utr5_length": max(utr5_lengths) if utr5_lengths else 0,
            "mean_utr3_length": sum(utr3_lengths) / max(1, len(utr3_lengths)),
            "max_utr3_length": max(utr3_lengths) if utr3_lengths else 0,
            **{key: value / max(1, total_examples) for key, value in metric_sums.items()},
        }

        is_final_step = step == args.steps - 1
        is_report_step = step % args.print_every == 0 or is_final_step
        validation_metrics = None
        if is_report_step and validation_batch is not None:
            model.eval()
            with torch.no_grad():
                dna, splice_tracks, *_unused, examples = validation_batch
                phase_target, genome_mask = phase_targets_from_examples(
                    examples,
                    max_length=dna.shape[1],
                    device=device,
                )
                logits, _encoded = model(dna, splice_tracks)
                per_base_loss = F.cross_entropy(
                    logits.reshape(-1, len(PHASE_LABELS)),
                    phase_target.reshape(-1),
                    weight=class_weights,
                    reduction="none",
                ).reshape_as(phase_target)
                val_loss = (
                    per_base_loss * genome_mask.to(per_base_loss.dtype)
                ).sum() / genome_mask.sum().clamp_min(1)
                validation_metrics = {"loss": float(val_loss.item()), **phase_metrics(logits, phase_target, genome_mask)}
            model.train()

        if args.save_best_checkpoint and is_report_step and loss_value < best_loss:
            best_loss = loss_value
            save_checkpoint(
                checkpoint_dir / "best.pt",
                checkpoint_payload(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    final_metrics=final_metrics,
                    best_loss=best_loss,
                ),
            )
            print(f"saved best checkpoint: {checkpoint_dir / 'best.pt'}", flush=True)

        if is_report_step:
            report_lines = [
                f"\nStep {step:06d} | learning_rate={scheduled_lr:.2e} | smoothed_loss={loss_ema:.4f}",
                (
                    f"Batch shape | proteins mean/max={final_metrics['mean_target_length']:.1f}/"
                    f"{final_metrics['max_target_length']} aa, genomes mean/max="
                    f"{final_metrics['mean_genome_length']:.0f}/{final_metrics['max_genome_length']} bp, "
                    f"introns mean/max={final_metrics['mean_intron_length']:.0f}/"
                    f"{final_metrics['max_intron_length']} bp, microbatch="
                    f"{effective_micro_batch_size}/{args.batch_size}"
                ),
                (
                    f"UTRs       5prime mean/max={final_metrics['mean_utr5_length']:.0f}/"
                    f"{final_metrics['max_utr5_length']} bp, 3prime mean/max="
                    f"{final_metrics['mean_utr3_length']:.0f}/{final_metrics['max_utr3_length']} bp"
                ),
                format_phase_report("train", final_metrics),
            ]
            if validation_metrics is not None:
                report_lines.append(format_phase_report("validation", validation_metrics))
            print("\n".join(report_lines), flush=True)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hybrid conv + Mamba2 + local-attention carried-phase predictor on synthetic spliced DNA."
    )
    parser.add_argument("--device", default="auto", help="Device to use. Default requires CUDA via auto.")
    parser.add_argument("--steps", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=0, help="Per-forward batch size. Use 0 for auto.")
    parser.add_argument("--max-micro-batch-tokens", type=int, default=500_000)
    parser.add_argument("--min-protein-codons", type=int, default=24)
    parser.add_argument("--max-protein-codons", type=int, default=96)
    parser.add_argument("--min-exon-count", type=int, default=1)
    parser.add_argument("--max-exon-count", type=int, default=4)
    parser.add_argument("--min-exon-bases", type=int, default=9)
    parser.add_argument("--max-exon-bases", type=int, default=300)
    parser.add_argument("--median-exon-bases", type=int, default=50)
    parser.add_argument("--rare-short-exon-prob", type=float, default=0.05)
    parser.add_argument("--exon-length-mode", choices=("split", "sampled"), default="split")
    parser.add_argument("--min-utr5-length", type=int, default=0)
    parser.add_argument("--max-utr5-length", type=int, default=0)
    parser.add_argument("--min-utr3-length", type=int, default=0)
    parser.add_argument("--max-utr3-length", type=int, default=0)
    parser.add_argument("--eval-protein-codons", type=int, default=48)
    parser.add_argument("--eval-exon-count", type=int, default=3)
    parser.add_argument("--min-intron-length", type=int, default=10)
    parser.add_argument("--max-intron-length", type=int, default=120)
    parser.add_argument("--length-bucket-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=8)
    parser.add_argument("--local-conv-kernel", type=int, default=9)
    parser.add_argument("--head-conv-kernel", type=int, default=7)
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-window", type=int, default=129)
    parser.add_argument("--use-prior-emit-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--not-coding-weight", type=float, default=0.25)
    parser.add_argument("--utr-weight", type=float, default=0.5)
    parser.add_argument("--nucleotide-loss-weight", type=float, default=0.5)
    parser.add_argument("--emit-loss-weight", type=float, default=1.0)
    parser.add_argument("--emit-loss-anneal-steps", type=int, default=0)
    parser.add_argument("--assignment-entropy-weight", type=float, default=0.0)
    parser.add_argument("--max-assignment-sharpness", type=float, default=10.0)
    parser.add_argument("--loss-ema-beta", type=float, default=0.95)
    parser.add_argument("--success-lr-decay", type=float, default=0.05)
    parser.add_argument("--success-exact-threshold", type=float, default=0.95)
    parser.add_argument("--success-nucleotide-exact-threshold", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-seed-offset", type=int, default=10_000)
    parser.add_argument("--validation-seed", type=int, default=424_242)
    parser.add_argument("--eval-seed", type=int, default=123_456)
    parser.add_argument("--failure-junction-window", type=int, default=3)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints/synthetic_splice_official_mamba2_phase_track")
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-keep-every", type=int, default=10_000)
    parser.add_argument("--save-best-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init-from", default="", help="Load model weights only, then start a fresh run.")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from", default="")
    return parser.parse_args()


def main() -> None:
    train_phase(parse_args())


if __name__ == "__main__":
    main()
