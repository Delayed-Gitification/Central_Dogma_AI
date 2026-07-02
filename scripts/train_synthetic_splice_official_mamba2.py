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
    max_intron_length: int,
    rng: random.Random,
) -> dict[str, object]:
    protein = random_protein(protein_codons, rng)
    cds = reverse_translate(protein, rng)
    exon_lengths = random_split_lengths(len(cds), exon_count, rng)

    genome_parts = []
    exon_prior = []
    donor_track = []
    acceptor_track = []
    true_transcript_rank = []

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
            intron_length = rng.randint(8, max_intron_length)
            intron = "GT" + random_dna(intron_length - 4, rng) + "AG"
            for base in intron:
                genome_parts.append(base)
                exon_prior.append(0.0)
                donor_track.append(0.0)
                acceptor_track.append(0.0)
                true_transcript_rank.append(-1.0)

    genome = "".join(genome_parts)
    target = torch.tensor([AA_TO_INDEX[amino_acid] for amino_acid in protein], dtype=torch.long)
    model_tracks = torch.tensor([[value] for value in exon_prior], dtype=torch.float32)
    annotations = torch.tensor(
        list(zip(exon_prior, donor_track, acceptor_track, true_transcript_rank)),
        dtype=torch.float32,
    )
    return {
        "genome": genome,
        "protein": protein,
        "dna": one_hot_dna(genome),
        "tracks": model_tracks,
        "annotations": annotations,
        "target": target,
    }


def make_batch(
    batch_size: int,
    protein_codons: int,
    device: torch.device,
    exon_count: int = 3,
    max_intron_length: int = 36,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, object]]]:
    rng = random.Random(seed)
    examples = [
        make_synthetic_example(protein_codons, exon_count, max_intron_length, rng)
        for _ in range(batch_size)
    ]
    max_length = max(example["dna"].shape[0] for example in examples)

    dna_rows = []
    track_rows = []
    target_rows = []
    for example in examples:
        dna = example["dna"]
        tracks = example["tracks"]
        padding = max_length - dna.shape[0]
        dna_rows.append(torch.cat([dna, torch.zeros(padding, 4)], dim=0))
        track_rows.append(torch.cat([tracks, torch.zeros(padding, tracks.shape[1])], dim=0))
        target_rows.append(example["target"])

    return (
        torch.stack(dna_rows).to(device),
        torch.stack(track_rows).to(device),
        torch.stack(target_rows).to(device),
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
        transcript_bases: int,
        input_dim: int = 5,
        hidden_dim: int = 32,
        layers: int = 3,
        chunk_size: int = 16,
        headdim: int = 8,
    ):
        super().__init__()
        self.transcript_bases = transcript_bases
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
        self.query = nn.Embedding(transcript_bases, hidden_dim)
        self.coordinate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.content_scale = nn.Parameter(torch.tensor(0.2))
        self.log_coordinate_sharpness = nn.Parameter(torch.tensor(-1.0))
        self.exon_prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, dna_one_hot: torch.Tensor, splice_tracks: torch.Tensor):
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

        query = self.query.weight
        content_logits = torch.einsum("bld,td->btl", encoded, query) / math.sqrt(encoded.shape[-1])

        latent_coordinate = torch.sigmoid(self.coordinate_head(encoded).squeeze(-1))
        target_coordinate = torch.linspace(0, 1, self.transcript_bases, device=dna_one_hot.device)
        coordinate_sharpness = F.softplus(self.log_coordinate_sharpness)
        coordinate_bias = -coordinate_sharpness * (
            latent_coordinate[:, None, :] - target_coordinate[None, :, None]
        ).pow(2)
        exon_bias = self.exon_prior_scale.abs() * torch.log(
            exon_prior.squeeze(-1).clamp_min(1e-4)
        )[:, None, :]

        pointer_logits = self.content_scale * content_logits + coordinate_bias + exon_bias
        attention = pointer_logits.softmax(dim=-1)

        transcript_bases = torch.einsum("btl,blc->btc", attention, dna_one_hot)
        codon_bases = transcript_bases.reshape(batch_size, -1, 3, 4)
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

        return amino_acid_probs, attention, {
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


def train(args: argparse.Namespace) -> None:
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
    print(f"annotation tracks generated but not used: {ANNOTATION_TRACK_NAMES}")

    transcript_bases = args.protein_codons * 3
    model = MambaSplicePointerTranslator(
        transcript_bases=transcript_bases,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        chunk_size=args.chunk_size,
        headdim=args.headdim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    final_metrics = None
    for step in range(args.steps):
        dna, splice_tracks, target, _examples = make_batch(
            batch_size=args.batch_size,
            protein_codons=args.protein_codons,
            device=device,
            exon_count=args.exon_count,
            max_intron_length=args.max_intron_length,
            seed=args.batch_seed_offset + step,
        )
        amino_acid_probs, _attention, diagnostics = model(dna, splice_tracks)
        aa_loss = F.nll_loss(torch.log(amino_acid_probs).reshape(-1, len(AMINO_ACIDS)), target.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        aa_loss.backward()
        optimizer.step()

        with torch.no_grad():
            predicted = amino_acid_probs.argmax(dim=-1)
            exact = (predicted == target).all(dim=1).float().mean().item()
            token_accuracy = (predicted == target).float().mean().item()

        final_metrics = {
            "step": step,
            "loss": aa_loss.item(),
            "token_accuracy": token_accuracy,
            "exact_match": exact,
            "attention_entropy": diagnostics["attention_entropy"].item(),
            "coordinate_sharpness": diagnostics["coordinate_sharpness"].item(),
            "coordinate_span": diagnostics["coordinate_span"].item(),
            "mean_exon_attention": diagnostics["mean_exon_attention"].item(),
        }

        if step % args.print_every == 0 or step == args.steps - 1:
            print(
                f"step={step:03d} loss={aa_loss.item():.3f} "
                f"token_acc={token_accuracy:.3f} exact={exact:.3f} "
                f"entropy={diagnostics['attention_entropy'].item():.3f} "
                f"coord_span={diagnostics['coordinate_span'].item():.2f} "
                f"coord_sharp={diagnostics['coordinate_sharpness'].item():.3f} "
                f"exon_attention={diagnostics['mean_exon_attention'].item():.3f}",
                flush=True,
            )

    dna, splice_tracks, _target, examples = make_batch(
        batch_size=1,
        protein_codons=args.protein_codons,
        device=device,
        exon_count=args.exon_count,
        max_intron_length=args.max_intron_length,
        seed=args.eval_seed,
    )
    model.eval()
    with torch.no_grad():
        amino_acid_probs, _attention, diagnostics = model(dna, splice_tracks)
    prediction = "".join(AMINO_ACIDS[index] for index in amino_acid_probs.argmax(dim=-1)[0].tolist())

    print("\nFinal training metrics:")
    if final_metrics is not None:
        for key, value in final_metrics.items():
            print(f"{key}: {value}")

    print("\nHeld-out synthetic example:")
    print("target:    ", examples[0]["protein"])
    print("predicted: ", prediction)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--protein-codons", type=int, default=24)
    parser.add_argument("--exon-count", type=int, default=3)
    parser.add_argument("--max-intron-length", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--headdim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-seed-offset", type=int, default=10_000)
    parser.add_argument("--eval-seed", type=int, default=123_456)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()