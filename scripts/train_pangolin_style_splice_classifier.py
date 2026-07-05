from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mamba_splice_soft_exist import (
    ACCEPTOR,
    DONOR,
    GtfSpliceWindowSampler,
    make_batch,
    metrics,
    motif_sanity,
    parse_chroms,
    pick_device,
    prepare_input,
)


DEFAULT_W = (11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 41, 41, 41, 41)
DEFAULT_AR = (1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25)
CLASS_NAMES = ("none", "acceptor", "donor")


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.replace(" ", ",").split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return x + out


class PangolinStyleSpliceClassifier(nn.Module):
    """Pangolin-like residual dilated conv net with a 3-class logit head.

    Input:  dna_probs [B, L, 4]
    Output: logits [B, L - 2 * crop, 3] in class order none/acceptor/donor.
    """

    def __init__(
        self,
        *,
        channels: int,
        kernels: tuple[int, ...],
        atrous_rates: tuple[int, ...],
        dropout: float,
        site_prior: float,
    ):
        super().__init__()
        if len(kernels) != len(atrous_rates):
            raise ValueError("--kernels and --atrous-rates must have the same length.")
        for kernel in kernels:
            if kernel % 2 == 0:
                raise ValueError("Pangolin-style same-length convs require odd kernel sizes.")

        self.channels = channels
        self.crop = int(sum(rate * (kernel - 1) for kernel, rate in zip(kernels, atrous_rates)))
        self.conv1 = nn.Conv1d(4, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)
        self.resblocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for index, (kernel, rate) in enumerate(zip(kernels, atrous_rates)):
            self.resblocks.append(ResBlock(channels, kernel, rate))
            if (index + 1) % 4 == 0 or index + 1 == len(kernels):
                self.skip_convs.append(nn.Conv1d(channels, channels, 1))
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Conv1d(channels, len(CLASS_NAMES), 1)

        prior = min(max(site_prior, 1e-8), 0.49)
        with torch.no_grad():
            self.final.bias[0] = math.log(max(1.0 - 2.0 * prior, 1e-8))
            self.final.bias[1] = math.log(prior)
            self.final.bias[2] = math.log(prior)

    def forward(self, dna_probs: torch.Tensor) -> torch.Tensor:
        if dna_probs.shape[1] <= 2 * self.crop:
            raise ValueError(
                f"Input length {dna_probs.shape[1]} is too short for this Pangolin-style receptive field. "
                f"Need > {2 * self.crop}; use --seq-len {2 * self.crop + 2048} or larger."
            )
        x = dna_probs.transpose(1, 2).contiguous()
        conv = self.conv1(x)
        skip = self.skip(conv)
        skip_index = 0
        for index, block in enumerate(self.resblocks):
            conv = block(conv)
            if (index + 1) % 4 == 0 or index + 1 == len(self.resblocks):
                skip = skip + self.skip_convs[skip_index](conv)
                skip_index += 1
        skip = skip[:, :, self.crop : -self.crop]
        logits = self.final(self.dropout(skip))
        return logits.transpose(1, 2).contiguous()


def crop_to_model(labels: torch.Tensor, mask: torch.Tensor, crop: int) -> tuple[torch.Tensor, torch.Tensor]:
    if crop <= 0:
        return labels, mask
    return labels[:, crop:-crop], mask[:, crop:-crop]


def labels_to_targets(labels: torch.Tensor) -> torch.Tensor:
    targets = torch.zeros(labels.shape[:2], dtype=torch.long, device=labels.device)
    targets[labels[..., ACCEPTOR] >= 0.5] = 1
    targets[labels[..., DONOR] >= 0.5] = 2
    return targets


def class_logits_to_binary_logits(class_logits: torch.Tensor) -> torch.Tensor:
    probs = class_logits.softmax(dim=-1)
    donor_acceptor = torch.stack([probs[..., 2], probs[..., 1]], dim=-1)
    donor_acceptor = donor_acceptor.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(donor_acceptor)


def splice_class_loss(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float,
    none_weight: float,
) -> torch.Tensor:
    targets = labels_to_targets(labels)
    weights = torch.tensor(
        [none_weight, positive_weight, positive_weight],
        dtype=class_logits.dtype,
        device=class_logits.device,
    )
    raw = F.cross_entropy(class_logits.transpose(1, 2), targets, weight=weights, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    model: PangolinStyleSpliceClassifier,
    sampler: GtfSpliceWindowSampler,
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, float], int, float, dict[str, bool]]:
    logits_rows = []
    label_rows = []
    mask_rows = []
    sequences: list[str] = []
    losses = []
    val_aug = {"soft": False, "exist": False}
    for batch_index in range(args.val_batches):
        val_dna, val_labels, val_sequences = make_batch(args, sampler, device, 100_000 + batch_index)
        val_dna_in, val_labels_in, _val_exist, val_mask, batch_aug = prepare_input(
            args,
            val_dna,
            val_labels,
            allow_augment=args.val_augment,
        )
        class_logits = model(val_dna_in)
        cropped_labels, cropped_mask = crop_to_model(val_labels_in, val_mask, model.crop)
        losses.append(
            splice_class_loss(
                class_logits,
                cropped_labels,
                cropped_mask,
                positive_weight=args.positive_weight,
                none_weight=args.none_weight,
            )
        )
        logits_rows.append(class_logits_to_binary_logits(class_logits))
        label_rows.append(cropped_labels)
        mask_rows.append(cropped_mask)
        if model.crop > 0:
            sequences.extend(sequence[model.crop : -model.crop] for sequence in val_sequences)
        else:
            sequences.extend(val_sequences)
        val_aug["soft"] = val_aug["soft"] or batch_aug["soft"]
        val_aug["exist"] = val_aug["exist"] or batch_aug["exist"]

    logits = torch.cat(logits_rows, dim=0)
    labels = torch.cat(label_rows, dim=0)
    mask = torch.cat(mask_rows, dim=0)
    val_loss = torch.stack(losses).mean()
    val_metrics = metrics(logits, labels, mask)
    val_motifs = motif_sanity(logits, labels, mask, sequences)
    return float(val_loss.item()), val_metrics, val_motifs, logits.shape[1], float(logits.shape[1]), val_aug


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    step: int,
    val_loss: float,
    val_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": step,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "class_names": CLASS_NAMES,
            "binary_track_names": ("donor", "acceptor"),
            "crop": model.crop,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.exist_augment_prob > 0 or args.junk_slots_per_base > 0:
        raise ValueError(
            "This Pangolin-style conv baseline supports fixed-length hard/soft DNA only. "
            "Use --exist-augment-prob 0 and --junk-slots-per-base 0."
        )
    kernels = parse_int_list(args.kernels)
    atrous_rates = parse_int_list(args.atrous_rates)
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
    model = PangolinStyleSpliceClassifier(
        channels=args.channels,
        kernels=kernels,
        atrous_rates=atrous_rates,
        dropout=args.dropout,
        site_prior=args.site_prior,
    ).to(device)
    if args.seq_len <= 2 * model.crop:
        raise ValueError(
            f"--seq-len {args.seq_len} is too short for crop={model.crop} each side. "
            f"Use --seq-len > {2 * model.crop}."
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir)
    best_loss = float("inf")

    print("Pangolin-style splice classifier")
    print(
        f"device={device}; seq_len={args.seq_len}; output_len={args.seq_len - 2 * model.crop}; "
        f"crop_each_side={model.crop}; batch={args.batch_size}; channels={args.channels}; "
        f"blocks={len(kernels)}; params={sum(p.numel() for p in model.parameters())}"
    )
    print(
        f"classes={CLASS_NAMES}; soft_prob={args.soft_augment_prob}; "
        f"loss weights none/positive={args.none_weight}/{args.positive_weight}"
    )

    for step in range(args.steps):
        model.train()
        dna, labels, _train_sequences = make_batch(args, sampler, device, step)
        dna_in, labels_in, _existence, mask, train_aug = prepare_input(args, dna, labels, allow_augment=True)
        class_logits = model(dna_in)
        cropped_labels, cropped_mask = crop_to_model(labels_in, mask, model.crop)
        loss = splice_class_loss(
            class_logits,
            cropped_labels,
            cropped_mask,
            positive_weight=args.positive_weight,
            none_weight=args.none_weight,
        )
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
                f"mean_prob {val_metrics['mean_prob']:.5f} input_len {args.seq_len} "
                f"output_len {val_input_len} exist_sum {val_exist_sum:.1f} "
                f"aug train soft/exist {int(train_aug['soft'])}/0 val {int(val_aug['soft'])}/0"
            )

    save_checkpoint(checkpoint_dir / "latest.pt", model, args, args.steps - 1, best_loss, {})
    print(f"saved latest checkpoint: {checkpoint_dir / 'latest.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Pangolin-style 3-class splice-site classifier on GTF/FASTA windows.")
    parser.add_argument("--fasta", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa")
    parser.add_argument("--gtf", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf")
    parser.add_argument("--chroms", default="chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22")
    parser.add_argument("--strands", default="+,-")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sites", type=int, default=300_000)
    parser.add_argument("--min-non-n-frac", type=float, default=0.95)

    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--kernels", default=",".join(str(value) for value in DEFAULT_W))
    parser.add_argument("--atrous-rates", default=",".join(str(value) for value in DEFAULT_AR))
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--site-prior", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=100.0)
    parser.add_argument("--none-weight", type=float, default=1.0)

    parser.add_argument("--soft-augment-prob", type=float, default=0.0)
    parser.add_argument("--soft-eps-min", type=float, default=0.01)
    parser.add_argument("--soft-eps-max", type=float, default=0.25)
    parser.add_argument("--soft-logit-noise-std", type=float, default=0.25)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--exist-augment-prob", type=float, default=0.0)
    parser.add_argument("--junk-slots-per-base", type=int, default=0)
    parser.add_argument("--junk-exist-max", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-augment", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/pangolin_style_splice_classifier")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
