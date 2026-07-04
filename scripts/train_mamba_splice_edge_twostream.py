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
        offset_min_frac: float = 1.0 / 3.0,
        offset_max_frac: float = 2.0 / 3.0,
    ):
        import pyfastx  # noqa: PLC0415

        self.fasta = pyfastx.Fasta(str(Path(fasta_path).expanduser()))
        self.seq_len = seq_len
        self.min_non_n_frac = min_non_n_frac
        self.offset_min_frac = offset_min_frac
        self.offset_max_frac = offset_max_frac
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
            offset_min = int(round(self.seq_len * self.offset_min_frac))
            offset_max = int(round(self.seq_len * self.offset_max_frac))
            offset_min = min(max(0, offset_min), self.seq_len)
            offset_max = min(max(offset_min, offset_max), self.seq_len)
            offset = rng.randint(offset_min, offset_max)
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
    def __init__(
        self,
        hidden_dim: int,
        chunk_size: int,
        headdim: int,
        d_conv: int,
        layerscale_init: float,
    ):
        super().__init__()
        Mamba2 = load_mamba2()
        self.chunk_size = chunk_size
        self.norm = nn.LayerNorm(hidden_dim)
        self.res_scale = nn.Parameter(torch.tensor(float(layerscale_init)))
        d_inner = 2 * hidden_dim
        if d_inner % headdim != 0:
            raise ValueError("2 * hidden_dim must be divisible by headdim")
        nheads = d_inner // headdim
        if nheads % 8 != 0:
            raise ValueError(
                "Mamba2 Triton kernels are happiest when nheads=(2 * hidden_dim / headdim) "
                f"is a multiple of 8. Got hidden_dim={hidden_dim}, headdim={headdim}, "
                f"nheads={nheads}. Try --hidden-dim 192 --headdim 16 or --hidden-dim 160 --headdim 8."
            )
        self.mamba = Mamba2(
            d_model=hidden_dim,
            d_state=32,
            d_conv=d_conv,
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
        return residual + self.res_scale * y


class TwoStreamMambaLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        chunk_size: int,
        headdim: int,
        d_conv: int,
        layerscale_init: float,
    ):
        super().__init__()
        self.fwd = OfficialMamba2Block(
            hidden_dim,
            chunk_size=chunk_size,
            headdim=headdim,
            d_conv=d_conv,
            layerscale_init=layerscale_init,
        )
        self.rev = OfficialMamba2Block(
            hidden_dim,
            chunk_size=chunk_size,
            headdim=headdim,
            d_conv=d_conv,
            layerscale_init=layerscale_init,
        )
        mix_dim = 2 * hidden_dim + 2
        self.mix_norm = nn.LayerNorm(mix_dim)
        self.mix_gate = nn.Linear(mix_dim, hidden_dim)
        self.mix_proj = nn.Sequential(
            nn.Linear(mix_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mix_scale = nn.Parameter(torch.tensor(float(layerscale_init)))

    def forward(
        self,
        fwd: torch.Tensor,
        rev: torch.Tensor,
        existence: torch.Tensor,
        effective_pos: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fwd = self.fwd(fwd) * mask[..., None]
        rev = torch.flip(self.rev(torch.flip(rev, dims=[1])), dims=[1]) * mask[..., None]
        mix = torch.cat([fwd, rev, existence[..., None], effective_pos[..., None]], dim=-1)
        mix = self.mix_norm(mix)
        gate = torch.sigmoid(self.mix_gate(mix))
        shared = self.mix_proj(mix) * mask[..., None]
        fwd = (fwd + self.mix_scale * gate * shared) * mask[..., None]
        rev = (rev + self.mix_scale * (1.0 - gate) * shared) * mask[..., None]
        return fwd, rev


def shift_right(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


def shift_left(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, 1:], torch.zeros_like(x[:, -1:])], dim=1)


class EdgeHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        edge_dim = 4 * hidden_dim + 4
        self.net = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        fwd: torch.Tensor,
        rev: torch.Tensor,
        existence: torch.Tensor,
        effective_pos: torch.Tensor,
        pos_sin: torch.Tensor,
        pos_cos: torch.Tensor,
    ) -> torch.Tensor:
        edge = torch.cat(
            [
                fwd,
                rev,
                shift_right(fwd),
                shift_left(rev),
                existence[..., None],
                effective_pos[..., None],
                pos_sin[..., None],
                pos_cos[..., None],
            ],
            dim=-1,
        )
        return self.net(edge)


class SoftExistBiMambaSplicePredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        shared_layers: int,
        branch_layers: int,
        chunk_size: int,
        headdim: int,
        d_conv: int,
        layerscale_init: float,
        dropout: float,
        site_prior: float,
    ):
        super().__init__()
        self.fwd_input_projection = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rev_input_projection = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer_kwargs = {
            "hidden_dim": hidden_dim,
            "chunk_size": chunk_size,
            "headdim": headdim,
            "d_conv": d_conv,
            "layerscale_init": layerscale_init,
        }
        self.shared_layers = nn.ModuleList(
            [TwoStreamMambaLayer(**layer_kwargs) for _ in range(shared_layers)]
        )
        self.donor_layers = nn.ModuleList(
            [TwoStreamMambaLayer(**layer_kwargs) for _ in range(branch_layers)]
        )
        self.acceptor_layers = nn.ModuleList(
            [TwoStreamMambaLayer(**layer_kwargs) for _ in range(branch_layers)]
        )
        self.donor_head = EdgeHead(hidden_dim, dropout=dropout)
        self.acceptor_head = EdgeHead(hidden_dim, dropout=dropout)
        self.donor_out = nn.Linear(hidden_dim, 1)
        self.acceptor_out = nn.Linear(hidden_dim, 1)
        prior = min(max(site_prior, 1e-8), 1.0 - 1e-8)
        bias = math.log(prior / (1.0 - prior))
        nn.init.constant_(self.donor_out.bias, bias)
        nn.init.constant_(self.acceptor_out.bias, bias)

    def forward(self, dna_probs: torch.Tensor, existence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        effective_pos = torch.cumsum(existence * mask, dim=1)
        effective_pos = effective_pos / effective_pos[:, -1:].clamp_min(1.0)
        pos_angle = effective_pos * (2.0 * math.pi)
        pos_sin = torch.sin(pos_angle)
        pos_cos = torch.cos(pos_angle)
        features = torch.cat(
            [
                dna_probs * existence[..., None],
                existence[..., None],
                effective_pos[..., None],
                pos_sin[..., None],
                pos_cos[..., None],
            ],
            dim=-1,
        )
        fwd = self.fwd_input_projection(features) * mask[..., None]
        rev = self.rev_input_projection(features) * mask[..., None]
        for layer in self.shared_layers:
            fwd, rev = layer(fwd, rev, existence, effective_pos, mask)

        donor_fwd, donor_rev = fwd, rev
        for layer in self.donor_layers:
            donor_fwd, donor_rev = layer(donor_fwd, donor_rev, existence, effective_pos, mask)
        acceptor_fwd, acceptor_rev = fwd, rev
        for layer in self.acceptor_layers:
            acceptor_fwd, acceptor_rev = layer(acceptor_fwd, acceptor_rev, existence, effective_pos, mask)

        donor_edge = self.donor_head(donor_fwd, donor_rev, existence, effective_pos, pos_sin, pos_cos)
        acceptor_edge = self.acceptor_head(acceptor_fwd, acceptor_rev, existence, effective_pos, pos_sin, pos_cos)
        donor = self.donor_out(donor_edge)
        acceptor = self.acceptor_out(acceptor_edge)
        return torch.cat([donor, acceptor], dim=-1)


def make_batch(
    args: argparse.Namespace,
    sampler: GtfSpliceWindowSampler,
    device: torch.device,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    rng = random.Random(args.seed + step)
    sequences = []
    dna_rows = []
    label_rows = []
    for _ in range(args.batch_size):
        sequence, labels = sampler.sample(rng)
        sequences.append(sequence)
        dna_rows.append(one_hot_sequence(sequence, device))
        label_rows.append(labels.to(device))
    return torch.stack(dna_rows), torch.stack(label_rows), sequences


def prepare_input(
    args: argparse.Namespace,
    dna: torch.Tensor,
    labels: torch.Tensor,
    *,
    allow_augment: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, bool]]:
    soft_augmented = allow_augment and random.random() < args.soft_augment_prob
    exist_augmented = False
    if soft_augmented:
        dna = soften_one_hot(dna, args.soft_eps_min, args.soft_eps_max, args.soft_logit_noise_std, args.soft_temperature)
    existence = torch.ones(dna.shape[:2], dtype=dna.dtype, device=dna.device)
    mask = torch.ones_like(existence)
    if allow_augment and args.junk_slots_per_base > 0 and random.random() < args.exist_augment_prob:
        exist_augmented = True
        max_length = dna.shape[1] * (1 + args.junk_slots_per_base)
        dna, labels, existence, mask = add_existence_junk(
            dna,
            labels,
            junk_slots_per_base=args.junk_slots_per_base,
            junk_exist_max=args.junk_exist_max,
            max_length=max_length,
        )
    return dna, labels, existence, mask, {"soft": soft_augmented, "exist": exist_augmented}


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
    topk_values = {}
    for channel, prefix in ((DONOR, "don"), (ACCEPTOR, "acc")):
        valid = mask.reshape(-1) > 0
        channel_probs = probs[..., channel].reshape(-1)
        channel_true = (labels[..., channel].reshape(-1) >= 0.5) & valid
        valid_count = int(valid.sum().item())
        true_count = int(channel_true.sum().item())
        if true_count > 0 and valid_count > 0:
            masked_probs = channel_probs.masked_fill(~valid, -1.0)
            k = min(true_count, valid_count)
            top_index = masked_probs.topk(k=k, largest=True).indices
            top_hits = channel_true[top_index].float().sum()
            k2 = min(2 * true_count, valid_count)
            top2_index = masked_probs.topk(k=k2, largest=True).indices
            top2_hits = channel_true[top2_index].float().sum()
            topk_values[f"{prefix}_topk"] = float((top_hits / float(k)).item())
            topk_values[f"{prefix}_top2k_rec"] = float((top2_hits / float(true_count)).item())
            topk_values[f"{prefix}_true_n"] = float(true_count)
        else:
            topk_values[f"{prefix}_topk"] = 0.0
            topk_values[f"{prefix}_top2k_rec"] = 0.0
            topk_values[f"{prefix}_true_n"] = float(true_count)
    return {
        "don_f1": float(f1[DONOR].item()),
        "acc_f1": float(f1[ACCEPTOR].item()),
        "don_rec": float(recall[DONOR].item()),
        "acc_rec": float(recall[ACCEPTOR].item()),
        "peak_true": float(peak_true.item()),
        "peak_pred": float(peak_pred.item()),
        "mean_prob": float((probs * mask[..., None]).sum().item() / (mask.sum().item() * labels.shape[-1])),
        **topk_values,
    }


@torch.no_grad()
def motif_sanity(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    sequences: list[str],
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    out: dict[str, float] = {}
    for channel, prefix, motif_side, motif in (
        (DONOR, "don", "right", "GT"),
        (ACCEPTOR, "acc", "left", "AG"),
    ):
        valid = mask.reshape(-1) > 0
        channel_probs = probs[..., channel].reshape(-1)
        channel_true = (labels[..., channel].reshape(-1) >= 0.5) & valid
        true_count = int(channel_true.sum().item())
        if true_count <= 0:
            out[f"{prefix}_top_motif"] = 0.0
            out[f"{prefix}_true_motif"] = 0.0
            continue
        masked_probs = channel_probs.masked_fill(~valid, -1.0)
        top_indices = masked_probs.topk(k=min(true_count, int(valid.sum().item())), largest=True).indices.cpu().tolist()
        true_indices = torch.nonzero(channel_true, as_tuple=False).flatten().cpu().tolist()

        def motif_fraction(flat_indices: list[int]) -> float:
            hits = 0
            total = 0
            length = labels.shape[1]
            for flat in flat_indices:
                row = flat // length
                pos = flat % length
                if row >= len(sequences):
                    continue
                sequence = sequences[row]
                if motif_side == "right":
                    if pos + 2 >= len(sequence):
                        continue
                    context = sequence[pos + 1 : pos + 3]
                else:
                    if pos - 2 < 0:
                        continue
                    context = sequence[pos - 2 : pos]
                total += 1
                hits += int(context == motif)
            return hits / max(1, total)

        out[f"{prefix}_top_motif"] = motif_fraction(top_indices)
        out[f"{prefix}_true_motif"] = motif_fraction(true_indices)
    return out


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    model: nn.Module,
    sampler: GtfSpliceWindowSampler,
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, float], int, float, dict[str, bool]]:
    logits_rows = []
    label_rows = []
    mask_rows = []
    exist_means = []
    sequences: list[str] = []
    losses = []
    val_aug = {"soft": False, "exist": False}
    for batch_index in range(args.val_batches):
        val_dna, val_labels, val_sequences = make_batch(args, sampler, device, 100_000 + batch_index)
        val_dna_in, val_labels_in, val_exist, val_mask, batch_aug = prepare_input(
            args,
            val_dna,
            val_labels,
            allow_augment=args.val_augment,
        )
        val_logits = model(val_dna_in, val_exist, val_mask)
        losses.append(splice_loss(val_logits, val_labels_in, val_mask, args.positive_weight))
        logits_rows.append(val_logits)
        label_rows.append(val_labels_in)
        mask_rows.append(val_mask)
        exist_means.append(val_exist.sum(dim=1).mean())
        sequences.extend(val_sequences)
        val_aug["soft"] = val_aug["soft"] or batch_aug["soft"]
        val_aug["exist"] = val_aug["exist"] or batch_aug["exist"]
    logits = torch.cat(logits_rows, dim=0)
    labels = torch.cat(label_rows, dim=0)
    mask = torch.cat(mask_rows, dim=0)
    val_loss = torch.stack(losses).mean()
    val_metrics = metrics(logits, labels, mask)
    val_motifs = motif_sanity(logits, labels, mask, sequences)
    exist_sum = float(torch.stack(exist_means).mean().item())
    return float(val_loss.item()), val_metrics, val_motifs, logits.shape[1], exist_sum, val_aug


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
    model = SoftExistBiMambaSplicePredictor(
        hidden_dim=args.hidden_dim,
        shared_layers=args.shared_layers,
        branch_layers=args.branch_layers,
        chunk_size=args.chunk_size,
        headdim=args.headdim,
        d_conv=args.d_conv,
        layerscale_init=args.layerscale_init,
        dropout=args.dropout,
        site_prior=args.site_prior,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir)
    best_loss = float("inf")

    print("Pure Mamba two-stream edge splice predictor")
    print(
        f"device={device}; seq_len={args.seq_len}; batch={args.batch_size}; hidden={args.hidden_dim}; "
        f"shared_layers={args.shared_layers}; branch_layers={args.branch_layers}; d_conv={args.d_conv}; "
        f"layerscale={args.layerscale_init}; site_prior={args.site_prior}; "
        f"val_batches={args.val_batches}; params={sum(p.numel() for p in model.parameters())}"
    )
    print(
        f"soft_prob={args.soft_augment_prob}; exist_prob={args.exist_augment_prob}; "
        f"junk/base={args.junk_slots_per_base}; input=[soft_base*exist, exist, effective_pos, sin/cos]; "
        f"head=edge[fwd_i, rev_i, fwd_i-1, rev_i+1]"
    )

    for step in range(args.steps):
        model.train()
        dna, labels, _train_sequences = make_batch(args, sampler, device, step)
        dna_in, labels_in, existence, mask, train_aug = prepare_input(args, dna, labels, allow_augment=True)
        logits = model(dna_in, existence, mask)
        loss = splice_loss(logits, labels_in, mask, args.positive_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_motifs, val_input_len, val_exist_sum, val_aug = evaluate(
                    args,
                    model,
                    sampler,
                    device,
                )
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(checkpoint_dir / "best.pt", model, args, step, best_loss, val_metrics)
            print(
                f"\nstep {step:06d} loss {loss.item():.4f} val {val_loss:.4f} best {best_loss:.4f} "
                f"topK donor/acceptor {val_metrics['don_topk']:.3f}/{val_metrics['acc_topk']:.3f} "
                f"top2Krec {val_metrics['don_top2k_rec']:.3f}/{val_metrics['acc_top2k_rec']:.3f}"
            )
            print(
                f"threshold f1 donor/acceptor {val_metrics['don_f1']:.3f}/{val_metrics['acc_f1']:.3f} "
                f"recall {val_metrics['don_rec']:.3f}/{val_metrics['acc_rec']:.3f} "
                f"true sites donor/acceptor {val_metrics['don_true_n']:.0f}/{val_metrics['acc_true_n']:.0f}"
            )
            print(
                f"motif sanity topK donor_GT/acceptor_AG "
                f"{val_motifs['don_top_motif']:.3f}/{val_motifs['acc_top_motif']:.3f} "
                f"| true donor_GT/acceptor_AG "
                f"{val_motifs['don_true_motif']:.3f}/{val_motifs['acc_true_motif']:.3f}"
            )
            print(
                f"peaks pred/true {val_metrics['peak_pred']:.3f}/{val_metrics['peak_true']:.3f} "
                f"mean_prob {val_metrics['mean_prob']:.5f} input_len {val_input_len} "
                f"exist_sum {val_exist_sum:.1f} "
                f"aug train soft/exist {int(train_aug['soft'])}/{int(train_aug['exist'])} "
                f"val {int(val_aug['soft'])}/{int(val_aug['exist'])}"
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
    parser.add_argument("--shared-layers", type=int, default=4)
    parser.add_argument("--branch-layers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--headdim", type=int, default=8)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--layerscale-init", type=float, default=0.1)
    parser.add_argument("--site-prior", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=150.0)

    parser.add_argument("--soft-augment-prob", type=float, default=0.0)
    parser.add_argument("--soft-eps-min", type=float, default=0.01)
    parser.add_argument("--soft-eps-max", type=float, default=0.25)
    parser.add_argument("--soft-logit-noise-std", type=float, default=0.25)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--exist-augment-prob", type=float, default=0.0)
    parser.add_argument("--junk-slots-per-base", type=int, default=1)
    parser.add_argument("--junk-exist-max", type=float, default=0.05)

    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=8, help="Fixed validation batches to average at each report.")
    parser.add_argument("--val-augment", action="store_true", help="Apply soft/existence augmentation during validation too.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/mamba_splice_edge_twostream")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
