from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DNA_BASES = "ACGT"
DNA_TO_INDEX = {base: index for index, base in enumerate(DNA_BASES)}
TRACK_NAMES = ("donor", "acceptor")
DONOR, ACCEPTOR = 0, 1


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
    return device


def load_mamba2():
    try:
        from mamba_ssm import Mamba2
    except ImportError as exc:
        try:
            from mamba_ssm.modules.mamba2 import Mamba2
        except ImportError:
            raise RuntimeError(
                "mamba_ssm is required for this script. On the cluster, activate the CUDA env "
                "you used for the Dogmamba/Mamba2 scripts."
            ) from exc
    return Mamba2


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTUNacgtun", "TGCAANtgcaan"))[::-1].upper().replace("U", "T")


def one_hot_sequence(sequence: str, device: torch.device) -> torch.Tensor:
    out = torch.zeros(len(sequence), 4, dtype=torch.float32, device=device)
    for index, base in enumerate(sequence.upper()):
        if base in DNA_TO_INDEX:
            out[index, DNA_TO_INDEX[base]] = 1.0
    return out


def extract_gtf_attr(attrs: str, key: str) -> str | None:
    needle = key + ' "'
    start = attrs.find(needle)
    if start < 0:
        return None
    end = attrs.find('"', start + len(needle))
    return attrs[start + len(needle) : end] if end > 0 else None


def parse_chroms(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = {part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()}
    return values or None


class GtfSpliceWindowSampler:
    def __init__(
        self,
        *,
        fasta_path: str,
        gtf_path: str,
        seq_len: int,
        chroms: set[str] | None,
        strands: set[str],
        max_sites: int,
        min_non_n_frac: float,
        seed: int,
    ):
        import pyfastx  # noqa: PLC0415

        self.fasta = pyfastx.Fasta(str(Path(fasta_path).expanduser()))
        self.seq_len = seq_len
        self.min_non_n_frac = min_non_n_frac
        available_chroms = set(self.fasta.keys())
        if chroms is None:
            chroms = available_chroms
        self.chroms = {chrom for chrom in chroms if chrom in available_chroms and len(self.fasta[chrom]) >= seq_len}
        if not self.chroms:
            raise ValueError("No requested chromosomes are present in the FASTA and long enough for --seq-len.")

        transcript_exons: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        line_count = 0
        exon_count = 0
        with open(Path(gtf_path).expanduser()) as handle:
            for line in handle:
                line_count += 1
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                chrom, _source, feature, start, end, _score, strand, _frame, attrs = parts[:9]
                if feature != "exon" or chrom not in self.chroms or strand not in strands:
                    continue
                transcript_id = extract_gtf_attr(attrs, "transcript_id")
                if transcript_id is None:
                    continue
                transcript_exons.setdefault((chrom, transcript_id, strand), []).append((int(start), int(end)))
                exon_count += 1
                if line_count % 1_000_000 == 0:
                    print(f"parsed {line_count:,} GTF lines, kept {exon_count:,} exons")

        self.sites_by_chrom: dict[str, dict[int, int]] = {}
        records: list[tuple[str, int, int, str]] = []
        skipped_tss = 0
        skipped_tts = 0
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
                is_first = index == 0
                is_last = index == len(exons_sorted) - 1
                if not is_first:
                    self._add_site(chrom, acceptor_pos, ACCEPTOR, records, strand)
                else:
                    skipped_tss += 1
                if not is_last:
                    self._add_site(chrom, donor_pos, DONOR, records, strand)
                else:
                    skipped_tts += 1

        if not records:
            raise ValueError("No splice sites parsed from GTF.")
        if max_sites > 0 and len(records) > max_sites:
            rng = random.Random(seed)
            records = rng.sample(records, max_sites)
        self.records = records
        self.sorted_sites = {chrom: sorted(sites.keys()) for chrom, sites in self.sites_by_chrom.items()}
        print(
            f"GTF windows: {len(self.records):,} site-centred records; "
            f"{sum(len(v) for v in self.sites_by_chrom.values()):,} unique clean sites; "
            f"chroms={len(self.sites_by_chrom)}; skipped TSS/TTS {skipped_tss:,}/{skipped_tts:,}"
        )

    def _add_site(
        self,
        chrom: str,
        position: int,
        label: int,
        records: list[tuple[str, int, int, str]],
        strand: str,
    ) -> None:
        sites = self.sites_by_chrom.setdefault(chrom, {})
        previous = sites.get(position)
        if previous is None or previous == label:
            sites[position] = label
            records.append((chrom, position, label, strand))

    def sample(self, rng: random.Random) -> tuple[str, torch.Tensor]:
        for _attempt in range(200):
            chrom, centre, _label, strand = rng.choice(self.records)
            chrom_len = len(self.fasta[chrom])
            offset = rng.randint(self.seq_len // 3, (2 * self.seq_len) // 3)
            start = min(max(0, centre - offset), max(0, chrom_len - self.seq_len))
            end = start + self.seq_len
            sequence = self.fasta[chrom][start:end].seq.upper().replace("U", "T")
            non_n = sum(base in DNA_TO_INDEX for base in sequence)
            if non_n / max(1, len(sequence)) < self.min_non_n_frac:
                continue

            labels = torch.zeros(self.seq_len, len(TRACK_NAMES), dtype=torch.float32)
            site_positions = self.sorted_sites.get(chrom, [])
            site_labels = self.sites_by_chrom.get(chrom, {})
            lo = bisect_left(site_positions, start)
            hi = bisect_right(site_positions, end - 1)
            for pos in site_positions[lo:hi]:
                slot = pos - start
                labels[slot, site_labels[pos]] = 1.0

            if strand == "-":
                sequence = reverse_complement(sequence)
                labels = labels.flip(0)
            return sequence, labels
        return sequence, labels


def soften_one_hot(one_hot: torch.Tensor, eps_min: float, eps_max: float, noise_std: float, temperature: float) -> torch.Tensor:
    eps = torch.empty((*one_hot.shape[:2], 1), device=one_hot.device, dtype=one_hot.dtype).uniform_(eps_min, eps_max)
    random_probs = torch.rand_like(one_hot)
    random_probs = random_probs / random_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    soft = (1.0 - eps) * one_hot + eps * random_probs
    if noise_std > 0:
        logits = soft.clamp_min(1e-8).log() + torch.randn_like(soft) * noise_std
        soft = (logits / max(temperature, 1e-6)).softmax(dim=-1)
    return soft / soft.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def add_existence_junk(
    dna: torch.Tensor,
    labels: torch.Tensor,
    *,
    junk_slots_per_base: int,
    junk_exist_max: float,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, _channels = dna.shape
    rows_dna = []
    rows_labels = []
    rows_exist = []
    rows_mask = []
    for row in range(batch):
        dna_parts = []
        label_parts = []
        exist_parts = []
        mask_parts = []
        for pos in range(length):
            dna_parts.append(dna[row, pos])
            label_parts.append(labels[row, pos])
            exist_parts.append(torch.ones((), dtype=dna.dtype, device=dna.device))
            mask_parts.append(torch.ones((), dtype=dna.dtype, device=dna.device))
            for _ in range(junk_slots_per_base):
                junk = torch.rand(4, dtype=dna.dtype, device=dna.device)
                junk = junk / junk.sum().clamp_min(1e-8)
                dna_parts.append(junk)
                label_parts.append(torch.zeros(labels.shape[-1], dtype=labels.dtype, device=labels.device))
                exist_parts.append(torch.rand((), dtype=dna.dtype, device=dna.device) * junk_exist_max)
                mask_parts.append(torch.ones((), dtype=dna.dtype, device=dna.device))
        row_dna = torch.stack(dna_parts)[:max_length]
        row_labels = torch.stack(label_parts)[:max_length]
        row_exist = torch.stack(exist_parts)[:max_length]
        row_mask = torch.stack(mask_parts)[:max_length]
        pad = max_length - row_dna.shape[0]
        if pad > 0:
            row_dna = torch.cat([row_dna, torch.zeros(pad, 4, dtype=dna.dtype, device=dna.device)], dim=0)
            row_labels = torch.cat([row_labels, torch.zeros(pad, labels.shape[-1], dtype=labels.dtype, device=labels.device)], dim=0)
            row_exist = torch.cat([row_exist, torch.zeros(pad, dtype=dna.dtype, device=dna.device)], dim=0)
            row_mask = torch.cat([row_mask, torch.zeros(pad, dtype=dna.dtype, device=dna.device)], dim=0)
        rows_dna.append(row_dna)
        rows_labels.append(row_labels)
        rows_exist.append(row_exist)
        rows_mask.append(row_mask)
    return torch.stack(rows_dna), torch.stack(rows_labels), torch.stack(rows_exist), torch.stack(rows_mask)


class OfficialMamba2Block(nn.Module):
    def __init__(self, hidden_dim: int, chunk_size: int, headdim: int):
        super().__init__()
        Mamba2 = load_mamba2()
        self.chunk_size = chunk_size
        self.norm = nn.LayerNorm(hidden_dim)
        d_inner = 2 * hidden_dim
        if d_inner % headdim != 0:
            raise ValueError("2 * hidden_dim must be divisible by headdim")
        self.mamba = Mamba2(
            d_model=hidden_dim,
            d_state=32,
            d_conv=4,
            expand=2,
            headdim=headdim,
            chunk_size=chunk_size,
        )

    def _pad_to_chunk(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        remainder = x.shape[1] % self.chunk_size
        if remainder == 0:
            return x, 0
        pad = self.chunk_size - remainder
        return torch.cat([x, torch.zeros(x.shape[0], pad, x.shape[2], dtype=x.dtype, device=x.device)], dim=1), pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y, pad = self._pad_to_chunk(self.norm(x))
        y = self.mamba(y)
        if isinstance(y, tuple):
            y = y[0]
        if pad:
            y = y[:, :-pad]
        return residual + y


class MambaSpliceSoftExistPredictor(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, chunk_size: int, headdim: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [OfficialMamba2Block(hidden_dim, chunk_size=chunk_size, headdim=headdim) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, len(TRACK_NAMES))

    def forward(self, dna_probs: torch.Tensor, existence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        effective_pos = torch.cumsum(existence * mask, dim=1)
        effective_pos = effective_pos / effective_pos[:, -1:].clamp_min(1.0)
        features = torch.cat(
            [
                dna_probs * existence[..., None],
                existence[..., None],
                effective_pos[..., None],
            ],
            dim=-1,
        )
        x = self.input_projection(features)
        x = x * mask[..., None]
        for block in self.blocks:
            x = block(x) * mask[..., None]
        return self.head(self.dropout(self.norm(x)))


def make_batch(args: argparse.Namespace, sampler: GtfSpliceWindowSampler, device: torch.device, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(args.seed + step)
    dna_rows = []
    label_rows = []
    for _ in range(args.batch_size):
        sequence, labels = sampler.sample(rng)
        dna_rows.append(one_hot_sequence(sequence, device))
        label_rows.append(labels.to(device))
    return torch.stack(dna_rows), torch.stack(label_rows)


def prepare_input(args: argparse.Namespace, dna: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if random.random() < args.soft_augment_prob:
        dna = soften_one_hot(dna, args.soft_eps_min, args.soft_eps_max, args.soft_logit_noise_std, args.soft_temperature)
    existence = torch.ones(dna.shape[:2], dtype=dna.dtype, device=dna.device)
    mask = torch.ones_like(existence)
    if args.junk_slots_per_base > 0 and random.random() < args.exist_augment_prob:
        max_length = dna.shape[1] * (1 + args.junk_slots_per_base)
        dna, labels, existence, mask = add_existence_junk(
            dna,
            labels,
            junk_slots_per_base=args.junk_slots_per_base,
            junk_exist_max=args.junk_exist_max,
            max_length=max_length,
        )
    return dna, labels, existence, mask


def splice_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, positive_weight: float) -> torch.Tensor:
    pos_weight = torch.full((labels.shape[-1],), positive_weight, dtype=logits.dtype, device=logits.device)
    raw = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none").mean(dim=-1)
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5) & (mask[..., None] > 0)
    true = (labels >= 0.5) & (mask[..., None] > 0)
    tp = (pred & true).sum(dim=(0, 1)).float()
    fp = (pred & ~true).sum(dim=(0, 1)).float()
    fn = (~pred & true).sum(dim=(0, 1)).float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    peak_true = labels.amax(dim=(1, 2)).mean()
    peak_pred = probs.amax(dim=(1, 2)).mean()
    return {
        "don_f1": float(f1[DONOR].item()),
        "acc_f1": float(f1[ACCEPTOR].item()),
        "don_rec": float(recall[DONOR].item()),
        "acc_rec": float(recall[ACCEPTOR].item()),
        "peak_true": float(peak_true.item()),
        "peak_pred": float(peak_pred.item()),
        "mean_prob": float((probs * mask[..., None]).sum().item() / (mask.sum().item() * labels.shape[-1])),
    }


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, step: int, val_loss: float, val_metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": step,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "track_names": TRACK_NAMES,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    chroms = parse_chroms(args.chroms)
    strands = {part.strip() for part in args.strands.split(",") if part.strip()}
    sampler = GtfSpliceWindowSampler(
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        seq_len=args.seq_len,
        chroms=chroms,
        strands=strands,
        max_sites=args.max_sites,
        min_non_n_frac=args.min_non_n_frac,
        seed=args.seed,
    )
    model = MambaSpliceSoftExistPredictor(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        chunk_size=args.chunk_size,
        headdim=args.headdim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir)
    best_loss = float("inf")

    print("Mamba splice predictor with soft DNA + existence inputs")
    print(
        f"device={device}; seq_len={args.seq_len}; batch={args.batch_size}; hidden={args.hidden_dim}; "
        f"layers={args.layers}; params={sum(p.numel() for p in model.parameters())}"
    )
    print(
        f"soft_prob={args.soft_augment_prob}; exist_prob={args.exist_augment_prob}; "
        f"junk/base={args.junk_slots_per_base}; explicit input=[soft_base*exist, exist, effective_pos]"
    )

    for step in range(args.steps):
        model.train()
        dna, labels = make_batch(args, sampler, device, step)
        dna_in, labels_in, existence, mask = prepare_input(args, dna, labels)
        logits = model(dna_in, existence, mask)
        loss = splice_loss(logits, labels_in, mask, args.positive_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                val_dna, val_labels = make_batch(args, sampler, device, 100_000 + step)
                val_dna_in, val_labels_in, val_exist, val_mask = prepare_input(args, val_dna, val_labels)
                val_logits = model(val_dna_in, val_exist, val_mask)
                val_loss = splice_loss(val_logits, val_labels_in, val_mask, args.positive_weight)
                val_metrics = metrics(val_logits, val_labels_in, val_mask)
            if val_loss.item() < best_loss:
                best_loss = float(val_loss.item())
                save_checkpoint(checkpoint_dir / "best.pt", model, args, step, best_loss, val_metrics)
            print(
                f"\nstep {step:06d} loss {loss.item():.4f} val {val_loss.item():.4f} best {best_loss:.4f} "
                f"f1 donor/acceptor {val_metrics['don_f1']:.3f}/{val_metrics['acc_f1']:.3f} "
                f"recall {val_metrics['don_rec']:.3f}/{val_metrics['acc_rec']:.3f}"
            )
            print(
                f"peaks pred/true {val_metrics['peak_pred']:.3f}/{val_metrics['peak_true']:.3f} "
                f"mean_prob {val_metrics['mean_prob']:.5f} input_len {val_dna_in.shape[1]} "
                f"exist_sum {val_exist.sum(dim=1).mean().item():.1f}"
            )

    save_checkpoint(checkpoint_dir / "latest.pt", model, args, args.steps - 1, best_loss, {})
    print(f"saved latest checkpoint: {checkpoint_dir / 'latest.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Mamba splice-site predictor on GTF/FASTA with soft DNA and existence-column augmentation.")
    parser.add_argument("--fasta", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa")
    parser.add_argument("--gtf", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf")
    parser.add_argument("--chroms", default="chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22")
    parser.add_argument("--strands", default="+,-")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sites", type=int, default=300_000)
    parser.add_argument("--min-non-n-frac", type=float, default=0.95)

    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=300.0)

    parser.add_argument("--soft-augment-prob", type=float, default=0.5)
    parser.add_argument("--soft-eps-min", type=float, default=0.01)
    parser.add_argument("--soft-eps-max", type=float, default=0.25)
    parser.add_argument("--soft-logit-noise-std", type=float, default=0.25)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--exist-augment-prob", type=float, default=0.25)
    parser.add_argument("--junk-slots-per-base", type=int, default=1)
    parser.add_argument("--junk-exist-max", type=float, default=0.05)

    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints/mamba_splice_soft_exist")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
