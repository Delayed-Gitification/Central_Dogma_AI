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
FixedBatch = tuple[torch.Tensor, torch.Tensor, list[str]]


def average_precision_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.float()
    positives = labels.sum()
    if positives <= 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    precision = sorted_labels.cumsum(dim=0) / torch.arange(
        1,
        sorted_labels.numel() + 1,
        dtype=scores.dtype,
        device=scores.device,
    )
    return float((precision * sorted_labels).sum().div(positives).item())


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.replace(" ", ",").split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def parse_milestones(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return parse_int_list(raw)


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


@torch.no_grad()
def class_metrics(class_logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    binary_logits = class_logits_to_binary_logits(class_logits)
    out = metrics(binary_logits, labels, mask)
    class_probs = class_logits.softmax(dim=-1)
    valid = mask.reshape(-1) > 0
    for label_channel, class_channel, prefix in ((DONOR, 2, "don"), (ACCEPTOR, 1, "acc")):
        scores = class_probs[..., class_channel].reshape(-1)[valid]
        truth = (labels[..., label_channel].reshape(-1)[valid] >= 0.5).float()
        out[f"{prefix}_ap"] = average_precision_from_scores(scores, truth)
    return out


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


def make_fixed_batches(
    args: argparse.Namespace,
    sampler: GtfSpliceWindowSampler,
    device: torch.device,
    *,
    seed_offset: int,
    batch_count: int,
) -> list[FixedBatch]:
    return [make_batch(args, sampler, device, seed_offset + batch_index) for batch_index in range(batch_count)]


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    model: PangolinStyleSpliceClassifier,
    fixed_batches: list[FixedBatch],
) -> tuple[float, dict[str, float], dict[str, float], int, float, dict[str, bool]]:
    class_logits_rows = []
    binary_logits_rows = []
    label_rows = []
    mask_rows = []
    sequences: list[str] = []
    losses = []
    val_aug = {"soft": False, "exist": False}
    for val_dna, val_labels, val_sequences in fixed_batches:
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
        class_logits_rows.append(class_logits)
        binary_logits_rows.append(class_logits_to_binary_logits(class_logits))
        label_rows.append(cropped_labels)
        mask_rows.append(cropped_mask)
        if model.crop > 0:
            sequences.extend(sequence[model.crop : -model.crop] for sequence in val_sequences)
        else:
            sequences.extend(val_sequences)
        val_aug["soft"] = val_aug["soft"] or batch_aug["soft"]
        val_aug["exist"] = val_aug["exist"] or batch_aug["exist"]

    class_logits = torch.cat(class_logits_rows, dim=0)
    binary_logits = torch.cat(binary_logits_rows, dim=0)
    labels = torch.cat(label_rows, dim=0)
    mask = torch.cat(mask_rows, dim=0)
    val_loss = torch.stack(losses).mean()
    val_metrics = class_metrics(class_logits, labels, mask)
    val_motifs = motif_sanity(binary_logits, labels, mask, sequences)
    return float(val_loss.item()), val_metrics, val_motifs, class_logits.shape[1], float(class_logits.shape[1]), val_aug


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
    train_chroms = parse_chroms(args.train_chroms or args.chroms)
    val_chroms = parse_chroms(args.val_chroms or args.chroms)
    strands = {part.strip() for part in args.strands.split(",") if part.strip()}
    train_sampler = GtfSpliceWindowSampler(
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        seq_len=args.seq_len,
        chroms=train_chroms,
        strands=strands,
        max_sites=args.max_sites,
        min_non_n_frac=args.min_non_n_frac,
        seed=args.seed,
        canonical_only=not args.allow_noncanonical_sites,
    )
    val_sampler = GtfSpliceWindowSampler(
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        seq_len=args.seq_len,
        chroms=val_chroms,
        strands=strands,
        max_sites=args.max_sites,
        min_non_n_frac=args.min_non_n_frac,
        seed=args.seed + 10_000,
        canonical_only=not args.allow_noncanonical_sites,
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
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    milestones = parse_milestones(args.lr_milestones)
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones), gamma=args.lr_gamma)
        if milestones
        else None
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    best_loss = float("inf")
    fixed_val_batches = make_fixed_batches(
        args,
        val_sampler,
        device,
        seed_offset=args.val_seed_offset,
        batch_count=args.val_batches,
    )
    total_steps = args.steps
    if args.epochs > 0:
        total_steps = args.epochs * args.steps_per_epoch

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
    print(
        f"train_chroms={','.join(train_chroms)}; val_chroms={','.join(val_chroms)}; "
        f"fixed_val_batches={len(fixed_val_batches)}; optimizer={args.optimizer}; lr={args.lr}; "
        f"epochs={args.epochs}; steps_per_epoch={args.steps_per_epoch}; lr_milestones={milestones}"
    )

    for step in range(total_steps):
        epoch = step // args.steps_per_epoch if args.epochs > 0 else 0
        step_in_epoch = step % args.steps_per_epoch if args.epochs > 0 else step
        model.train()
        dna, labels, _train_sequences = make_batch(args, train_sampler, device, step)
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
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        end_of_epoch = args.epochs > 0 and step_in_epoch == args.steps_per_epoch - 1
        if step % args.print_every == 0 or step == total_steps - 1 or end_of_epoch:
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_motifs, val_input_len, val_exist_sum, val_aug = evaluate(
                    args,
                    model,
                    fixed_val_batches,
                )
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(checkpoint_dir / "best.pt", model, args, step, best_loss, val_metrics)
            print(
                f"\nstep {step:06d} epoch {epoch + 1:02d} step {step_in_epoch:04d}/{args.steps_per_epoch} "
                f"lr {optimizer.param_groups[0]['lr']:.2e} loss {loss.item():.4f} "
                f"val {val_loss:.4f} best {best_loss:.4f} "
                f"topK donor/acceptor {val_metrics['don_topk']:.3f}/{val_metrics['acc_topk']:.3f} "
                f"top2Krec {val_metrics['don_top2k_rec']:.3f}/{val_metrics['acc_top2k_rec']:.3f} "
                f"AP {val_metrics['don_ap']:.3f}/{val_metrics['acc_ap']:.3f}"
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
        if args.epochs > 0 and end_of_epoch and scheduler is not None:
            scheduler.step()

    save_checkpoint(checkpoint_dir / "latest.pt", model, args, total_steps - 1, best_loss, {})
    print(f"saved latest checkpoint: {checkpoint_dir / 'latest.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Pangolin-style 3-class splice-site classifier on GTF/FASTA windows.")
    parser.add_argument("--fasta", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa")
    parser.add_argument("--gtf", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf")
    parser.add_argument("--chroms", default="chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22")
    parser.add_argument("--train-chroms", default=None)
    parser.add_argument("--val-chroms", default=None)
    parser.add_argument("--strands", default="+,-")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sites", type=int, default=300_000)
    parser.add_argument("--min-non-n-frac", type=float, default=0.95)
    parser.add_argument("--allow-noncanonical-sites", action="store_true")

    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernels", default=",".join(str(value) for value in DEFAULT_W))
    parser.add_argument("--atrous-rates", default=",".join(str(value) for value in DEFAULT_AR))
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--site-prior", type=float, default=1.0 / 3.0)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--none-weight", type=float, default=1.0)
    parser.add_argument("--lr-milestones", default="6,7,8,9")
    parser.add_argument("--lr-gamma", type=float, default=0.5)

    parser.add_argument("--soft-augment-prob", type=float, default=0.0)
    parser.add_argument("--soft-eps-min", type=float, default=0.01)
    parser.add_argument("--soft-eps-max", type=float, default=0.25)
    parser.add_argument("--soft-logit-noise-std", type=float, default=0.25)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--exist-augment-prob", type=float, default=0.0)
    parser.add_argument("--junk-slots-per-base", type=int, default=0)
    parser.add_argument("--junk-exist-max", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1_000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-seed-offset", type=int, default=100_000)
    parser.add_argument("--val-augment", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/pangolin_style_splice_classifier")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
