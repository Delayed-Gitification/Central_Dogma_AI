from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_mamba_splice_soft_exist import (
    GtfSpliceWindowSampler,
    make_batch,
    metrics,
    motif_sanity,
    parse_chroms,
    pick_device,
    prepare_input,
)
from train_pangolin_style_splice_classifier import (
    CLASS_NAMES,
    class_logits_to_binary_logits,
    class_metrics,
    parse_milestones,
    splice_class_loss,
)


FixedBatch = tuple[torch.Tensor, torch.Tensor, list[str]]


def make_fixed_batches(
    args: argparse.Namespace,
    sampler: GtfSpliceWindowSampler,
    device: torch.device,
    *,
    seed_offset: int,
    batch_count: int,
) -> list[FixedBatch]:
    return [make_batch(args, sampler, device, seed_offset + batch_index) for batch_index in range(batch_count)]


def center_crop(
    x: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    sequences: list[str],
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    if target_length <= 0 or target_length == x.shape[1]:
        return x, labels, mask, sequences
    if target_length > x.shape[1]:
        raise ValueError(f"--target-length {target_length} exceeds output length {x.shape[1]}.")
    start = (x.shape[1] - target_length) // 2
    end = start + target_length
    return x[:, start:end], labels[:, start:end], mask[:, start:end], [seq[start:end] for seq in sequences]


class ExistAwareAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_mult: int,
        dropout: float,
        local_window: float,
        relative_buckets: int,
        relative_bucket_size: float,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("--hidden-dim must be divisible by --heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.local_window = local_window
        self.relative_buckets = relative_buckets
        self.relative_bucket_size = relative_bucket_size

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_mult * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * hidden_dim, hidden_dim),
        )
        if relative_buckets > 0:
            self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * relative_buckets + 1))
        else:
            self.relative_bias = None

    def _attention_bias(
        self,
        existence: torch.Tensor,
        mask: torch.Tensor,
        effective_pos: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key_real = (mask > 0) & (existence > 1e-6)
        bias = torch.zeros(
            existence.shape[0],
            1,
            existence.shape[1],
            existence.shape[1],
            device=existence.device,
            dtype=dtype,
        )
        bias = bias + existence.clamp_min(1e-6).log().to(dtype)[:, None, None, :]
        bias = bias.masked_fill(~key_real[:, None, None, :], torch.finfo(dtype).min)

        signed_dist = None
        if self.local_window > 0 or self.relative_bias is not None:
            signed_dist = effective_pos[:, :, None] - effective_pos[:, None, :]
        if self.local_window > 0:
            bias = bias.masked_fill(signed_dist.abs()[:, None] > self.local_window, torch.finfo(dtype).min)
        if self.relative_bias is not None:
            bucket = torch.round(signed_dist / max(self.relative_bucket_size, 1e-6)).long()
            bucket = bucket.clamp(-self.relative_buckets, self.relative_buckets) + self.relative_buckets
            rel = self.relative_bias[:, bucket].permute(1, 0, 2, 3).to(dtype)
            bias = bias + rel
        return bias

    def forward(
        self,
        x: torch.Tensor,
        existence: torch.Tensor,
        mask: torch.Tensor,
        effective_pos: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        qkv = self.qkv(y).view(y.shape[0], y.shape[1], 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        v = v * existence[:, None, :, None]
        attn_bias = self._attention_bias(existence, mask, effective_pos, q.dtype)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.hidden_dim)
        x = residual + self.dropout(self.out(y))
        x = x * mask[..., None]
        x = x + self.dropout(self.mlp(self.norm2(x))) * mask[..., None]
        return x * mask[..., None]


class TinyExistTransformer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        layers: int,
        heads: int,
        mlp_mult: int,
        dropout: float,
        local_window: float,
        relative_buckets: int,
        relative_bucket_size: float,
        head_kernel: int,
    ):
        super().__init__()
        if head_kernel > 0 and head_kernel % 2 == 0:
            raise ValueError("--head-kernel must be odd, or 0 to disable the local head path.")
        self.base_embedding = nn.Parameter(torch.empty(4, hidden_dim))
        nn.init.normal_(self.base_embedding, std=0.02)
        self.scalar_projection = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ExistAwareAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=heads,
                    mlp_mult=mlp_mult,
                    dropout=dropout,
                    local_window=local_window,
                    relative_buckets=relative_buckets,
                    relative_bucket_size=relative_bucket_size,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        if head_kernel > 0:
            padding = head_kernel // 2
            self.local_head = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, head_kernel, padding=padding),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, head_kernel, padding=padding),
                nn.GELU(),
            )
        else:
            self.local_head = None
        self.final = nn.Linear(hidden_dim, len(CLASS_NAMES))
        with torch.no_grad():
            self.final.bias.zero_()

    def forward(self, dna_probs: torch.Tensor, existence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        effective_pos = torch.cumsum(existence * mask, dim=1)
        effective_pos_norm = effective_pos / effective_pos[:, -1:].clamp_min(1.0)
        base = dna_probs @ self.base_embedding
        x = base * existence[..., None]
        scalars = torch.stack([existence, effective_pos_norm], dim=-1)
        x = x + self.scalar_projection(scalars)
        x = x * mask[..., None]
        for block in self.blocks:
            x = block(x, existence, mask, effective_pos)
        x = self.norm(x)
        if self.local_head is not None:
            x = x + self.local_head(x.transpose(1, 2)).transpose(1, 2) * mask[..., None]
        return self.final(x)


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    model: TinyExistTransformer,
    fixed_batches: list[FixedBatch],
) -> tuple[float, dict[str, float], dict[str, float], int, float, dict[str, bool]]:
    class_logits_rows = []
    binary_logits_rows = []
    label_rows = []
    mask_rows = []
    sequences: list[str] = []
    losses = []
    exist_means = []
    val_aug = {"soft": False, "exist": False}
    for val_dna, val_labels, val_sequences in fixed_batches:
        val_dna_in, val_labels_in, val_exist, val_mask, batch_aug = prepare_input(
            args,
            val_dna,
            val_labels,
            allow_augment=args.val_augment,
        )
        class_logits = model(val_dna_in, val_exist, val_mask)
        class_logits, cropped_labels, cropped_mask, cropped_sequences = center_crop(
            class_logits,
            val_labels_in,
            val_mask,
            val_sequences,
            args.target_length,
        )
        losses.append(splice_class_loss(class_logits, cropped_labels, cropped_mask, args.positive_weight, args.none_weight))
        class_logits_rows.append(class_logits)
        binary_logits_rows.append(class_logits_to_binary_logits(class_logits))
        label_rows.append(cropped_labels)
        mask_rows.append(cropped_mask)
        sequences.extend(cropped_sequences)
        exist_means.append(val_exist.sum(dim=1).mean())
        val_aug["soft"] = val_aug["soft"] or batch_aug["soft"]
        val_aug["exist"] = val_aug["exist"] or batch_aug["exist"]

    class_logits_all = torch.cat(class_logits_rows, dim=0)
    binary_logits = torch.cat(binary_logits_rows, dim=0)
    labels = torch.cat(label_rows, dim=0)
    mask = torch.cat(mask_rows, dim=0)
    val_loss = torch.stack(losses).mean()
    val_metrics = class_metrics(class_logits_all, labels, mask)
    val_motifs = motif_sanity(binary_logits, labels, mask, sequences)
    exist_sum = float(torch.stack(exist_means).mean().item())
    return float(val_loss.item()), val_metrics, val_motifs, class_logits_all.shape[1], exist_sum, val_aug


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
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

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
    model = TinyExistTransformer(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        local_window=args.local_window,
        relative_buckets=args.relative_buckets,
        relative_bucket_size=args.relative_bucket_size,
        head_kernel=args.head_kernel,
    ).to(device)

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
    total_steps = args.steps if args.epochs <= 0 else args.epochs * args.steps_per_epoch

    print("TinyExistTransformer splice classifier")
    print(
        f"device={device}; seq_len={args.seq_len}; target_len={args.target_length}; batch={args.batch_size}; "
        f"hidden={args.hidden_dim}; layers={args.layers}; heads={args.heads}; "
        f"local_window={args.local_window}; relative_buckets={args.relative_buckets}; "
        f"params={sum(p.numel() for p in model.parameters())}"
    )
    print(
        f"classes={CLASS_NAMES}; soft_prob={args.soft_augment_prob}; exist_prob={args.exist_augment_prob}; "
        f"junk/base={args.junk_slots_per_base}; loss weights none/positive={args.none_weight}/{args.positive_weight}"
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
        dna, labels, train_sequences = make_batch(args, train_sampler, device, step)
        dna_in, labels_in, existence, mask, train_aug = prepare_input(args, dna, labels, allow_augment=True)
        class_logits = model(dna_in, existence, mask)
        class_logits, cropped_labels, cropped_mask, _cropped_sequences = center_crop(
            class_logits,
            labels_in,
            mask,
            train_sequences,
            args.target_length,
        )
        loss = splice_class_loss(class_logits, cropped_labels, cropped_mask, args.positive_weight, args.none_weight)
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
                f"mean_prob {val_metrics['mean_prob']:.5f} input_len {val_input_len} "
                f"exist_sum {val_exist_sum:.1f} "
                f"aug train soft/exist {int(train_aug['soft'])}/{int(train_aug['exist'])} "
                f"val {int(val_aug['soft'])}/{int(val_aug['exist'])}"
            )
        if args.epochs > 0 and end_of_epoch and scheduler is not None:
            scheduler.step()

    save_checkpoint(checkpoint_dir / "latest.pt", model, args, total_steps - 1, best_loss, {})
    print(f"saved latest checkpoint: {checkpoint_dir / 'latest.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an existence-aware Transformer splice-site classifier.")
    parser.add_argument("--fasta", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa")
    parser.add_argument("--gtf", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf")
    parser.add_argument("--chroms", default="chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY")
    parser.add_argument("--train-chroms", default=None)
    parser.add_argument("--val-chroms", default=None)
    parser.add_argument("--strands", default="+,-")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--target-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-sites", type=int, default=300_000)
    parser.add_argument("--min-non-n-frac", type=float, default=0.95)
    parser.add_argument("--allow-noncanonical-sites", action="store_true")

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--local-window", type=float, default=0.0, help="Effective-position attention window. 0 means full attention.")
    parser.add_argument("--relative-buckets", type=int, default=32)
    parser.add_argument("--relative-bucket-size", type=float, default=16.0)
    parser.add_argument("--head-kernel", type=int, default=9)

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
    parser.add_argument("--junk-exist-max", type=float, default=0.05)

    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1_000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-seed-offset", type=int, default=100_000)
    parser.add_argument("--val-augment", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/exist_transformer_splice")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
