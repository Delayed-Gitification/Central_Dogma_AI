from __future__ import annotations

import argparse

import torch

from train_mamba_splice_soft_exist import (
    GtfSpliceWindowSampler,
    make_batch,
    metrics,
    motif_sanity,
    parse_chroms,
    pick_device,
)


def probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probs)


@torch.no_grad()
def score_batch_with_bipangolin(runner, sequences: list[str], score_source: str) -> torch.Tensor:
    rows = []
    for sequence in sequences:
        result = runner.score_sequence(sequence)
        if score_source == "probe":
            donor = result.probe_donor.float()
            acceptor = result.probe_acceptor.float()
        elif score_source == "routed_prob":
            routed_prob, _routed_psi = result.routed_tracks()
            # biPangolin routed tracks are acceptor, donor; average tissues.
            acceptor = routed_prob[0].float().mean(dim=0)
            donor = routed_prob[1].float().mean(dim=0)
        else:
            raise ValueError(f"Unknown score source: {score_source}")
        rows.append(torch.stack([donor, acceptor], dim=-1))
    return torch.stack(rows, dim=0)


def build_runner(args: argparse.Namespace, device: torch.device):
    from bipangolin import BiPangolinRunner  # PyPI/active-env import on cluster.

    kwargs = {
        "device": str(device),
        "ensemble": not args.no_ensemble,
        "tissue": args.tissue,
        "n_models_per_tissue": args.n_models_per_tissue,
    }
    if args.pangolin_model_dir:
        kwargs["pangolin_model_dir"] = args.pangolin_model_dir
    if args.probe_dir:
        kwargs["probe_dir"] = args.probe_dir
    if args.correction_k is not None:
        kwargs["correction_k"] = args.correction_k
    if args.correction_file:
        kwargs["correction_file"] = args.correction_file
    return BiPangolinRunner(**kwargs)


def benchmark(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    chroms = parse_chroms(args.chroms)
    strands = {part.strip() for part in args.strands.split(",") if part.strip()}
    context_len = args.context_len if args.context_len > 0 else args.seq_len + 2 * args.context_flank
    if context_len < args.seq_len:
        raise ValueError(f"--context-len must be >= --seq-len, got {context_len} < {args.seq_len}")
    roi_start = (context_len - args.seq_len) // 2
    roi_end = roi_start + args.seq_len
    offset_min_frac = roi_start / float(context_len)
    offset_max_frac = max(roi_start, roi_end - 1) / float(context_len)
    sampler = GtfSpliceWindowSampler(
        fasta_path=args.fasta,
        gtf_path=args.gtf,
        seq_len=context_len,
        chroms=chroms,
        strands=strands,
        max_sites=args.max_sites,
        min_non_n_frac=args.min_non_n_frac,
        seed=args.seed,
        offset_min_frac=offset_min_frac,
        offset_max_frac=offset_max_frac,
    )
    runner = build_runner(args, device)

    logits_rows = []
    label_rows = []
    mask_rows = []
    sequences_all: list[str] = []

    print("biPangolin exact-validation benchmark")
    print(
        f"device={device}; roi_len={args.seq_len}; context_len={context_len}; "
        f"crop={roi_start}:{roi_end}; batch={args.batch_size}; "
        f"val_batches={args.val_batches}; score_source={args.score_source}"
    )
    print(
        f"validation seeds are identical to train_mamba_splice_soft_exist.py: "
        f"seed + 100000 + batch_index"
    )

    for batch_index in range(args.val_batches):
        _dna, labels, sequences = make_batch(args, sampler, torch.device("cpu"), 100_000 + batch_index)
        probs = score_batch_with_bipangolin(runner, sequences, args.score_source).cpu()
        probs = probs[:, roi_start:roi_end]
        labels = labels[:, roi_start:roi_end]
        roi_sequences = [sequence[roi_start:roi_end] for sequence in sequences]
        logits_rows.append(probs_to_logits(probs))
        label_rows.append(labels.cpu())
        mask_rows.append(torch.ones(labels.shape[:2], dtype=torch.float32))
        sequences_all.extend(roi_sequences)
        print(f"scored val batch {batch_index + 1}/{args.val_batches}")

    logits = torch.cat(logits_rows, dim=0)
    labels = torch.cat(label_rows, dim=0)
    mask = torch.cat(mask_rows, dim=0)

    val_metrics = metrics(logits, labels, mask)
    val_motifs = motif_sanity(logits, labels, mask, sequences_all)

    print(
        f"\nbiPangolin same-val topK donor/acceptor "
        f"{val_metrics['don_topk']:.3f}/{val_metrics['acc_topk']:.3f} "
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
        f"mean_prob {val_metrics['mean_prob']:.5f} input_len {context_len} roi_len {args.seq_len} "
        f"exist_sum {float(args.seq_len):.1f} aug train soft/exist 0/0 val 0/0"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyPI biPangolin on the exact same fixed validation windows as the Mamba splice script.")
    parser.add_argument("--fasta", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/GRCh38.primary_assembly.genome.fa")
    parser.add_argument("--gtf", default="/camp/home/wilkino/home/POSTDOC/software/biPangolin/data/gencode.v47.basic.annotation.gtf")
    parser.add_argument("--chroms", default="chr2,chr4,chr6,chr8,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22")
    parser.add_argument("--strands", default="+,-")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048, help="Central ROI length used for labels/metrics after cropping.")
    parser.add_argument("--context-flank", type=int, default=5000, help="Real genomic flank on each side of the ROI for biPangolin scoring.")
    parser.add_argument("--context-len", type=int, default=0, help="Override total scored sequence length. Default: seq_len + 2*context_flank.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--max-sites", type=int, default=300_000)
    parser.add_argument("--min-non-n-frac", type=float, default=0.95)

    parser.add_argument("--score-source", choices=("probe", "routed_prob"), default="probe")
    parser.add_argument("--pangolin-model-dir", default=None)
    parser.add_argument("--probe-dir", default=None)
    parser.add_argument("--tissue", default="all_tissues")
    parser.add_argument("--n-models-per-tissue", type=int, default=None)
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--correction-k", type=float, default=None)
    parser.add_argument("--correction-file", default=None)
    return parser.parse_args()


def main() -> None:
    benchmark(parse_args())


if __name__ == "__main__":
    main()
