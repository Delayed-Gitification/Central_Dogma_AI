from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DNA_BASES = "ACGT"
DNA_TO_INDEX = {base: index for index, base in enumerate(DNA_BASES)}


def add_bipangolin_to_path(root: str) -> None:
    root_path = Path(root).expanduser().resolve()
    src_path = root_path / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"Could not find biPangolin src directory: {src_path}")
    sys.path.insert(0, str(src_path))


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested MPS, but PyTorch MPS is not available.")
    return device


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def random_splice_rich_dna(length: int, rng: random.Random) -> str:
    """Synthetic DNA with occasional canonical splice-like motifs.

    The teacher is still real biPangolin; this just gives it more interesting
    sequence contexts than uniform random DNA.
    """
    chars = list(random_dna(length, rng))
    motif_count = max(1, length // 96)
    motifs = ("GT", "AG", "CAG", "GTAAGT", "TTTTTTTT", "CCCCCCCC")
    for _ in range(motif_count):
        motif = rng.choice(motifs)
        if len(motif) >= length:
            continue
        start = rng.randint(0, length - len(motif))
        chars[start : start + len(motif)] = list(motif)
    return "".join(chars)


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTUNacgtun", "TGCAANtgcaan"))[::-1].upper().replace("U", "T")


def one_hot_sequence(sequence: str, device: torch.device) -> torch.Tensor:
    out = torch.zeros(len(sequence), 4, device=device)
    for index, base in enumerate(sequence.upper()):
        if base in DNA_TO_INDEX:
            out[index, DNA_TO_INDEX[base]] = 1.0
    return out


class GenomeSampler:
    def __init__(self, fasta_path: str, chroms: str, seq_len: int, min_non_n_frac: float):
        import pyfastx  # noqa: PLC0415

        self.fasta = pyfastx.Fasta(str(Path(fasta_path).expanduser()))
        available = list(self.fasta.keys())
        if chroms:
            requested = [chrom.strip() for chrom in chroms.split(",") if chrom.strip()]
        else:
            requested = available
        self.chroms = [chrom for chrom in requested if chrom in self.fasta and len(self.fasta[chrom]) >= seq_len]
        if not self.chroms:
            raise ValueError(
                f"No usable chromosomes found for seq_len={seq_len}. "
                f"Requested={requested[:8]}, available examples={available[:8]}"
            )
        self.seq_len = seq_len
        self.min_non_n_frac = min_non_n_frac

    def sample(self, rng: random.Random) -> str:
        for _attempt in range(200):
            chrom = rng.choice(self.chroms)
            chrom_len = len(self.fasta[chrom])
            start = rng.randint(0, chrom_len - self.seq_len)
            sequence = self.fasta[chrom][start : start + self.seq_len].seq.upper().replace("U", "T")
            non_n_frac = sum(base in DNA_TO_INDEX for base in sequence) / max(1, len(sequence))
            if non_n_frac >= self.min_non_n_frac:
                return sequence
        return sequence


def extract_gtf_attr(attrs: str, key: str) -> str | None:
    needle = key + ' "'
    start = attrs.find(needle)
    if start < 0:
        return None
    end = attrs.find('"', start + len(needle))
    return attrs[start + len(needle) : end] if end > 0 else None


class GtfSiteGenomeSampler(GenomeSampler):
    """Sample real genomic windows centred near true transcript splice sites."""

    def __init__(
        self,
        fasta_path: str,
        gtf_path: str,
        chroms: str,
        strands: str,
        seq_len: int,
        min_non_n_frac: float,
        max_sites: int,
        seed: int,
    ):
        super().__init__(fasta_path, chroms, seq_len, min_non_n_frac)
        wanted_chroms = set(self.chroms)
        wanted_strands = {strand.strip() for strand in strands.split(",") if strand.strip()}
        transcript_exons: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        with open(Path(gtf_path).expanduser()) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                chrom, _source, feature, start, end, _score, strand, _frame, attrs = parts[:9]
                if feature != "exon" or chrom not in wanted_chroms or strand not in wanted_strands:
                    continue
                transcript_id = extract_gtf_attr(attrs, "transcript_id")
                if transcript_id is None:
                    continue
                transcript_exons.setdefault((chrom, transcript_id, strand), []).append((int(start), int(end)))

        records: list[tuple[str, int, str]] = []
        for (chrom, _transcript_id, strand), exons in transcript_exons.items():
            if len(exons) < 2:
                continue
            exons_sorted = sorted(exons, key=lambda item: item[0], reverse=(strand == "-"))
            for index, (start, end) in enumerate(exons_sorted):
                if strand == "+":
                    acceptor_pos = start - 1
                    donor_pos = end - 1
                else:
                    acceptor_pos = end - 1
                    donor_pos = start - 1
                if index > 0:
                    records.append((chrom, acceptor_pos, strand))
                if index < len(exons_sorted) - 1:
                    records.append((chrom, donor_pos, strand))

        if not records:
            raise ValueError(f"No GTF splice-site records found in {gtf_path} for chroms={sorted(wanted_chroms)}")
        if max_sites > 0 and len(records) > max_sites:
            rng = random.Random(seed)
            records = rng.sample(records, max_sites)
        self.records = records
        print(
            f"GTF sampler: {len(self.records):,} annotated splice-site centres "
            f"across {len({record[0] for record in self.records})} chroms"
        )

    def sample(self, rng: random.Random) -> str:
        for _attempt in range(200):
            chrom, site_pos, strand = rng.choice(self.records)
            chrom_len = len(self.fasta[chrom])
            offset = rng.randint(max(0, self.seq_len // 4), max(0, (3 * self.seq_len) // 4))
            start = min(max(0, site_pos - offset), max(0, chrom_len - self.seq_len))
            sequence = self.fasta[chrom][start : start + self.seq_len].seq.upper().replace("U", "T")
            if strand == "-":
                sequence = reverse_complement(sequence)
            non_n_frac = sum(base in DNA_TO_INDEX for base in sequence) / max(1, len(sequence))
            if non_n_frac >= self.min_non_n_frac:
                return sequence
        return sequence


def make_batch(
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    batch_size: int | None = None,
    genome_sampler: GenomeSampler | None = None,
) -> tuple[list[str], torch.Tensor]:
    rng = random.Random(seed)
    batch = args.batch_size if batch_size is None else batch_size
    sequences = []
    rows = []
    for _ in range(batch):
        if genome_sampler is not None:
            sequence = genome_sampler.sample(rng)
        elif rng.random() < args.splice_rich_fraction:
            sequence = random_splice_rich_dna(args.seq_len, rng)
        else:
            sequence = random_dna(args.seq_len, rng)
        sequences.append(sequence)
        rows.append(one_hot_sequence(sequence, device))
    return sequences, torch.stack(rows)


def soften_one_hot(
    one_hot: torch.Tensor,
    *,
    eps_min: float,
    eps_max: float,
    logit_noise_std: float,
    temperature: float,
) -> torch.Tensor:
    eps = torch.empty((*one_hot.shape[:2], 1), dtype=one_hot.dtype, device=one_hot.device).uniform_(eps_min, eps_max)
    random_probs = torch.rand_like(one_hot)
    random_probs = random_probs / random_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    soft = (1.0 - eps) * one_hot + eps * random_probs
    if logit_noise_std > 0:
        logits = soft.clamp_min(1e-8).log() + torch.randn_like(soft) * logit_noise_std
        soft = (logits / max(temperature, 1e-6)).softmax(dim=-1)
    return soft / soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def dna_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()


def pad_for_pangolin(dna_probs: torch.Tensor, crop: int) -> torch.Tensor:
    batch, _length, channels = dna_probs.shape
    padding = torch.zeros(batch, crop, channels, dtype=dna_probs.dtype, device=dna_probs.device)
    padded = torch.cat([padding, dna_probs, padding], dim=1)
    return padded.transpose(1, 2).contiguous()


class DifferentiableBiPangolinPair(nn.Module):
    def __init__(
        self,
        *,
        pangolin_model: nn.Module,
        probe: nn.Module,
        probe_cfg: dict,
        probe_layers: list[str],
        tissue_index: int,
        correction_k: float | None,
        crop: int,
        prob_channel_per_tissue: list[int],
        attach_hooks,
    ):
        super().__init__()
        self.pangolin_model = pangolin_model
        self.probe = probe
        self.probe_cfg = probe_cfg
        self.tissue_index = int(tissue_index)
        self.correction_k = correction_k
        self.crop = int(crop)
        self.prob_channel = int(prob_channel_per_tissue[self.tissue_index])
        self.handles = attach_hooks(self.pangolin_model, probe_layers)

    def corrected_probe_probs(self, probe_logits: torch.Tensor) -> torch.Tensor:
        probs = probe_logits.softmax(dim=1)
        if self.correction_k is not None and self.correction_k != 1.0:
            corrected = probs.clone()
            corrected[:, 0] = corrected[:, 0] * float(self.correction_k)
            probs = corrected / corrected.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return probs

    def forward(self, dna_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_tensor = pad_for_pangolin(dna_probs, self.crop)
        pangolin_out = self.pangolin_model(seq_tensor)
        length = dna_probs.shape[1]
        pangolin_out = pangolin_out[..., :length]

        gathered = []
        for layer_name, handle in self.handles.items():
            acts = handle["cache"]["activations"]
            if not handle["is_cropped"]:
                acts = acts[..., self.crop : self.crop + length]
            gathered.append(acts)
        if self.probe_cfg.get("include_sequence", False):
            gathered.append(seq_tensor[..., self.crop : self.crop + length])

        combined = torch.cat(gathered, dim=1)
        probe_logits = self.probe(combined)
        probe_probs = self.corrected_probe_probs(probe_logits)
        pangolin_prob = pangolin_out[:, self.prob_channel, :length]

        acceptor = pangolin_prob * probe_probs[:, 1]
        donor = pangolin_prob * probe_probs[:, 2]
        routed_soft = torch.stack([acceptor, donor], dim=-1)
        return {
            "pangolin_prob": pangolin_prob,
            "probe_probs": probe_probs.transpose(1, 2),
            "routed_soft": routed_soft,
        }


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def load_pair(args: argparse.Namespace, device: torch.device):
    from bipangolin.runner import (  # noqa: PLC0415
        BiPangolinRunner,
        PANGOLIN_CROP,
        PROB_CHANNEL_PER_TISSUE,
        attach_hooks,
        load_probe,
        parse_probe_layers,
    )
    from bipangolin.model import AR, L, W, Pangolin  # noqa: PLC0415

    runner = BiPangolinRunner(
        pangolin_model_dir=args.pangolin_model_dir or None,
        probe_dir=args.probe_dir or None,
        device=device,
        tissue=args.tissue,
        n_models_per_tissue=1,
        ensemble=False,
        output_unscaled_values=args.output_unscaled_values,
    )
    pangolin_path, probe_path, tissue_index = runner._pair_specs[0]
    map_location = device if device.type == "cuda" else torch.device("cpu")

    def make_trainable_pair() -> DifferentiableBiPangolinPair:
        pangolin = Pangolin(L, W, AR).to(device)
        pangolin.load_state_dict(torch.load(pangolin_path, map_location=map_location))
        probe, cfg = load_probe(probe_path, device)
        set_trainable(probe, True)
        layers = parse_probe_layers(cfg["probe_layer"])
        return DifferentiableBiPangolinPair(
            pangolin_model=pangolin,
            probe=probe,
            probe_cfg=cfg,
            probe_layers=layers,
            tissue_index=tissue_index,
            correction_k=runner.correction_k,
            crop=PANGOLIN_CROP,
            prob_channel_per_tissue=PROB_CHANNEL_PER_TISSUE,
            attach_hooks=attach_hooks,
        ).to(device)

    teacher = make_trainable_pair()
    student = make_trainable_pair()
    teacher.eval()
    set_trainable(teacher, False)
    set_trainable(student.pangolin_model, args.train_pangolin)
    set_trainable(student.probe, args.train_probe)

    print(f"loaded real biPangolin pair: {pangolin_path.name} + {probe_path.name}")
    print(f"tissue={runner.tissue_names[0]}; correction_k={runner.correction_k}")
    return teacher, student


def mse_dict(student: dict[str, torch.Tensor], teacher: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "routed_mse": F.mse_loss(student["routed_soft"], teacher["routed_soft"]),
        "pangolin_mse": F.mse_loss(student["pangolin_prob"], teacher["pangolin_prob"]),
        "probe_mse": F.mse_loss(student["probe_probs"], teacher["probe_probs"]),
    }


def index_outputs(outputs: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, index) for key, value in outputs.items()}


@torch.no_grad()
def sample_teacher_enriched_batch(
    args: argparse.Namespace,
    device: torch.device,
    teacher: DifferentiableBiPangolinPair,
    seed: int,
    genome_sampler: GenomeSampler | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    multiplier = max(1, args.peak_enrichment_candidates)
    candidate_count = args.batch_size * multiplier
    _seqs, hard = make_batch(args, device, seed, batch_size=candidate_count, genome_sampler=genome_sampler)
    teacher_outputs = teacher(hard)
    if multiplier == 1:
        return hard, teacher_outputs
    peaks = teacher_outputs["routed_soft"].amax(dim=(1, 2))
    selected = peaks.topk(k=args.batch_size, largest=True).indices
    return hard.index_select(0, selected), index_outputs(teacher_outputs, selected)


@torch.no_grad()
def track_metrics(pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> dict[str, float]:
    routed_gap = (pred["routed_soft"] - target["routed_soft"]).abs()
    pred_peak = pred["routed_soft"].amax(dim=(1, 2))
    target_peak = target["routed_soft"].amax(dim=(1, 2))
    peak_gap = (pred_peak - target_peak).abs()
    return {
        "mae": float(routed_gap.mean().item()),
        "max_abs": float(routed_gap.max().item()),
        "peak_gap": float(peak_gap.mean().item()),
        "target_peak": float(target_peak.mean().item()),
        "pred_peak": float(pred_peak.mean().item()),
    }


def save_checkpoint(path: Path, student: DifferentiableBiPangolinPair, args: argparse.Namespace, step: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "args": vars(args),
            "step": step,
            "metrics": metrics,
            "note": "Student is a real Pangolin+biPangolin probe pair fine-tuned for soft DNA probabilities.",
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    add_bipangolin_to_path(args.bipangolin_root)
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    teacher, student = load_pair(args, device)
    genome_sampler = None
    if args.fasta:
        if args.gtf:
            genome_sampler = GtfSiteGenomeSampler(
                args.fasta,
                args.gtf,
                args.chroms,
                args.gtf_strands,
                args.seq_len,
                args.min_non_n_frac,
                args.max_gtf_sites,
                args.seed,
            )
        else:
            genome_sampler = GenomeSampler(args.fasta, args.chroms, args.seq_len, args.min_non_n_frac)

    param_groups = []
    if args.train_pangolin:
        param_groups.append({"params": student.pangolin_model.parameters(), "lr": args.pangolin_lr})
    if args.train_probe:
        param_groups.append({"params": student.probe.parameters(), "lr": args.probe_lr})
    if not param_groups:
        raise ValueError("Nothing is trainable. Pass --train-pangolin and/or --train-probe.")
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    checkpoint_dir = Path(args.checkpoint_dir)
    best_loss = float("inf")
    print("real biPangolin soft-DNA distillation")
    print(
        f"device={device}; seq_len={args.seq_len}; batch={args.batch_size}; "
        f"train_pangolin={args.train_pangolin}; train_probe={args.train_probe}"
    )
    print(
        f"soft eps={args.soft_eps_min}-{args.soft_eps_max}; logit_noise={args.soft_logit_noise_std}; "
        f"loss weights soft/hard/probe={args.soft_loss_weight}/{args.hard_anchor_weight}/{args.probe_loss_weight}"
    )

    for step in range(args.steps):
        student.train()
        hard, teacher_hard = sample_teacher_enriched_batch(args, device, teacher, args.seed + step, genome_sampler)
        soft = soften_one_hot(
            hard,
            eps_min=args.soft_eps_min,
            eps_max=args.soft_eps_max,
            logit_noise_std=args.soft_logit_noise_std,
            temperature=args.soft_temperature,
        )

        student_soft = student(soft)
        student_hard = student(hard)
        soft_parts = mse_dict(student_soft, teacher_hard)
        hard_parts = mse_dict(student_hard, teacher_hard)
        loss = (
            args.soft_loss_weight * (soft_parts["routed_mse"] + soft_parts["pangolin_mse"])
            + args.probe_loss_weight * soft_parts["probe_mse"]
            + args.hard_anchor_weight * (hard_parts["routed_mse"] + hard_parts["pangolin_mse"] + hard_parts["probe_mse"])
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            student.eval()
            val_hard, val_teacher = sample_teacher_enriched_batch(
                args, device, teacher, args.seed + 10_000 + step, genome_sampler
            )
            val_soft = soften_one_hot(
                val_hard,
                eps_min=args.soft_eps_min,
                eps_max=args.soft_eps_max,
                logit_noise_std=args.soft_logit_noise_std,
                temperature=args.soft_temperature,
            )
            with torch.no_grad():
                val_student_soft = student(val_soft)
                val_student_hard = student(val_hard)
                val_soft_parts = mse_dict(val_student_soft, val_teacher)
                val_hard_parts = mse_dict(val_student_hard, val_teacher)
                val_loss = (
                    args.soft_loss_weight * (val_soft_parts["routed_mse"] + val_soft_parts["pangolin_mse"])
                    + args.probe_loss_weight * val_soft_parts["probe_mse"]
                    + args.hard_anchor_weight * (
                        val_hard_parts["routed_mse"] + val_hard_parts["pangolin_mse"] + val_hard_parts["probe_mse"]
                    )
                )
                soft_metrics = track_metrics(val_student_soft, val_teacher)
                hard_metrics = track_metrics(val_student_hard, val_teacher)
            if val_loss.item() < best_loss:
                best_loss = float(val_loss.item())
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    student,
                    args,
                    step,
                    {"val_loss": best_loss, **{f"soft_{k}": v for k, v in soft_metrics.items()}},
                )
            print(
                f"\nstep {step:06d} train {loss.item():.6f} val {val_loss.item():.6f} best {best_loss:.6f} "
                f"entropy {dna_entropy(val_soft).item():.3f}"
            )
            print(
                f"soft input  mae {soft_metrics['mae']:.5f} max {soft_metrics['max_abs']:.5f} "
                f"peak_gap {soft_metrics['peak_gap']:.5f} pred/target_peak {soft_metrics['pred_peak']:.4f}/{soft_metrics['target_peak']:.4f}"
            )
            print(
                f"hard anchor mae {hard_metrics['mae']:.5f} max {hard_metrics['max_abs']:.5f} "
                f"| mse soft routed/prob/probe "
                f"{val_soft_parts['routed_mse'].item():.6g}/{val_soft_parts['pangolin_mse'].item():.6g}/{val_soft_parts['probe_mse'].item():.6g}"
            )

    save_checkpoint(checkpoint_dir / "latest.pt", student, args, args.steps - 1, {"best_loss": best_loss})
    print(f"saved latest checkpoint: {checkpoint_dir / 'latest.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune real biPangolin/Pangolin to tolerate fixed-length soft DNA probabilities.")
    parser.add_argument("--bipangolin-root", default="/Users/ogw/Documents/GitHub/biPangolin")
    parser.add_argument("--pangolin-model-dir", default="")
    parser.add_argument("--probe-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tissue", default="brain", choices=("heart", "liver", "brain", "testis"))
    parser.add_argument("--output-unscaled-values", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--splice-rich-fraction", type=float, default=0.75)
    parser.add_argument("--fasta", default="", help="Reference genome FASTA. If provided, train on real genomic windows.")
    parser.add_argument("--gtf", default="", help="Optional GTF. If provided, sample windows around annotated splice sites.")
    parser.add_argument("--chroms", default="", help="Comma-separated chromosomes to sample. Empty means all FASTA chromosomes.")
    parser.add_argument("--gtf-strands", default="+,-", help="Comma-separated GTF strands to sample.")
    parser.add_argument("--max-gtf-sites", type=int, default=200_000)
    parser.add_argument("--min-non-n-frac", type=float, default=0.95)
    parser.add_argument(
        "--peak-enrichment-candidates",
        type=int,
        default=1,
        help="Generate this many candidate batches and keep examples with the strongest frozen biPangolin peaks.",
    )

    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--pangolin-lr", type=float, default=1e-6)
    parser.add_argument("--probe-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-pangolin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-probe", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--soft-eps-min", type=float, default=0.02)
    parser.add_argument("--soft-eps-max", type=float, default=0.35)
    parser.add_argument("--soft-logit-noise-std", type=float, default=0.35)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--soft-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-anchor-weight", type=float, default=0.25)
    parser.add_argument("--probe-loss-weight", type=float, default=0.25)

    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", default="checkpoints/bipangolin_soft_dna_distill")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
