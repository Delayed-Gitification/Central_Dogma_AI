from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATPLOTLIB_CACHE = ROOT / ".cache" / "matplotlib"
MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


DEFAULT_TRAIN_SCRIPT = ROOT / "scripts" / "train_dense_transition_phase_helper.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "dense_transition_phase_helper_sparse_cuda" / "best.pt"
STOP_CODONS = {"TAA", "TAG", "TGA"}

REGION_NAMES = ("intron", "5' UTR", "C phase 0", "C phase 1", "C phase 2", "3' UTR")
REGION_COLORS = ("#f0f0f0", "#80cdc1", "#1b9e77", "#d95f02", "#7570b3", "#c2a5cf")
STATE_COLORS = (
    "#8dd3c7",
    "#66c2a5",
    "#41ae76",
    "#1b9e77",
    "#44aa99",
    "#88ccee",
    "#d95f02",
    "#ee8866",
    "#e6ab02",
    "#7570b3",
    "#9970ab",
    "#b2abd2",
    "#d9d9d9",
    "#bdbdbd",
    "#969696",
    "#c7eae5",
    "#80cdc1",
    "#35978f",
    "#f6e8c3",
    "#dfc27d",
    "#bf812d",
    "#525252",
    "#737373",
    "#969696",
)
CODON_SIGNAL_COLORS = ("#ffffff", "#1a9850", "#d73027")


def import_training_script(path: Path):
    spec = importlib.util.spec_from_file_location("dense_transition_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dense_transition_train"] = module
    spec.loader.exec_module(module)
    return module


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def default_model_args() -> argparse.Namespace:
    return argparse.Namespace(
        device="auto",
        seed=20260707,
        checkpoint_dir=ROOT / "checkpoints" / "_plot_unused",
        steps=1,
        examples_per_step=1,
        validation_examples=1,
        eval_batch_size=1,
        print_every=1,
        validate_every=1,
        checkpoint_every=1,
        lr=3e-4,
        weight_decay=0.0,
        grad_clip=5.0,
        hidden_dim=96,
        conv_layers=4,
        use_splice_tracks=True,
        materialize_transitions=True,
        evidence_loss_weight=0.1,
        start_loss_weight=0.25,
        stop_loss_weight=0.25,
        donor_loss_weight=0.1,
        acceptor_loss_weight=0.1,
        min_utr5_length=100,
        max_utr5_length=300,
        min_coding_codons=40,
        max_coding_codons=140,
        min_utr3_length=100,
        max_utr3_length=400,
        min_exons=2,
        max_exons=8,
        min_exon_length=6,
        min_intron_length=50,
        max_intron_length=300,
        intron_mod="any",
    )


def merged_args(checkpoint_args: dict[str, object], cli_args: argparse.Namespace) -> argparse.Namespace:
    defaults = vars(default_model_args())
    defaults.update(checkpoint_args)
    for key in (
        "hidden_dim",
        "conv_layers",
        "use_splice_tracks",
        "materialize_transitions",
        "min_utr5_length",
        "max_utr5_length",
        "min_coding_codons",
        "max_coding_codons",
        "min_utr3_length",
        "max_utr3_length",
        "min_exons",
        "max_exons",
        "min_exon_length",
        "min_intron_length",
        "max_intron_length",
        "intron_mod",
    ):
        value = getattr(cli_args, key, None)
        if value is not None:
            defaults[key] = value
    defaults["checkpoint_dir"] = ROOT / "checkpoints" / "_plot_unused"
    return argparse.Namespace(**defaults)


def state_region_codes(module, target: torch.Tensor) -> np.ndarray:
    target_np = target.cpu().numpy()
    codes = np.zeros_like(target_np, dtype=int)
    for g in range(3):
        codes[target_np == module.idx_U(g)] = 1
        codes[target_np == module.idx_T(g)] = 5
        for p in range(3):
            codes[target_np == module.idx_C(g, p)] = 2 + p
            codes[target_np == module.idx_I(g, p)] = 0
    return codes


def exon_spans(module, target: torch.Tensor) -> list[tuple[int, int]]:
    target_np = target.cpu().numpy()
    is_intron = np.zeros_like(target_np, dtype=bool)
    for g in range(3):
        for p in range(3):
            is_intron |= target_np == module.idx_I(g, p)
    is_exonic = ~is_intron
    spans = []
    start = None
    for index, exonic in enumerate(is_exonic):
        if exonic and start is None:
            start = index
        elif not exonic and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(target_np)))
    return spans


def raw_genome_codon_signal(dna: str, start: int, end: int) -> np.ndarray:
    signal = np.zeros((3, end - start), dtype=int)
    for pos in range(start, end):
        if pos + 3 > len(dna):
            continue
        codon = dna[pos : pos + 3]
        if codon == "ATG":
            signal[pos % 3, pos - start] = 1
        elif codon in STOP_CODONS:
            signal[pos % 3, pos - start] = 2
    return signal


def junction_windows(gene, pad: int, limit: int = 4) -> list[tuple[int, int, str]]:
    windows = []
    for index, donor in enumerate(gene.donor_positions):
        donor_base = max(0, donor - 1)
        windows.append((max(0, donor_base - pad), min(len(gene.dna), donor_base + pad), f"donor {index}"))
        if len(windows) >= limit:
            return windows
    for index, acceptor in enumerate(gene.acceptor_positions):
        windows.append((max(0, acceptor - pad), min(len(gene.dna), acceptor + pad), f"acceptor {index}"))
        if len(windows) >= limit:
            return windows
    return windows


def plot_example(
    *,
    module,
    gene,
    output,
    target: torch.Tensor,
    output_path: Path,
    title: str,
    start: int,
    end: int,
    show_bases: bool = False,
    zoom: bool = False,
) -> dict[str, object]:
    start = max(0, int(start))
    end = min(len(gene.dna), int(end))
    sl = slice(start, end)
    x = np.arange(start, end)

    pred = output.state_log_probs.argmax(dim=-1).detach().cpu()
    probs = output.state_probs.detach().cpu()
    confidence = probs.max(dim=-1).values.numpy()[sl]
    target_cpu = target.detach().cpu()
    target_np = target_cpu.numpy()
    pred_np = pred.numpy()
    correct = (target_np[sl] == pred_np[sl]).astype(float)
    regions = state_region_codes(module, target_cpu)[sl]
    codon_signal = raw_genome_codon_signal(gene.dna, start, end)
    posteriors = {
        "start": output.start_posterior.detach().cpu().numpy()[sl],
        "stop": output.stop_posterior.detach().cpu().numpy()[sl],
        "donor": output.donor_posterior.detach().cpu().numpy()[sl],
        "acceptor": output.acceptor_posterior.detach().cpu().numpy()[sl],
    }

    region_cmap = ListedColormap(REGION_COLORS)
    region_norm = BoundaryNorm(np.arange(-0.5, len(REGION_NAMES) + 0.5), region_cmap.N)
    state_cmap = ListedColormap(STATE_COLORS)
    state_norm = BoundaryNorm(np.arange(-0.5, len(module.STATE_NAMES) + 0.5), state_cmap.N)
    codon_cmap = ListedColormap(CODON_SIGNAL_COLORS)
    codon_norm = BoundaryNorm(np.arange(-0.5, 3.5), codon_cmap.N)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(16 if zoom else 18, 9.5 if zoom else 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 0.8, 0.5, 0.5, 0.75, 0.9, 0.75]},
    )

    axes[0].imshow(regions[None, :], aspect="auto", interpolation="nearest", cmap=region_cmap, norm=region_norm, extent=[start, end, 0, 1])
    for exon_start, exon_end in exon_spans(module, target_cpu):
        if exon_end >= start and exon_start <= end:
            axes[0].axvspan(max(start, exon_start), min(end, exon_end), ymin=0.05, ymax=0.95, fill=False, edgecolor="black", lw=0.7)
    axes[0].set_ylabel("gene\nfeatures", rotation=0, ha="right", va="center")

    axes[1].imshow(codon_signal, aspect="auto", interpolation="nearest", origin="lower", cmap=codon_cmap, norm=codon_norm, extent=[start, end, -0.5, 2.5])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["0", "1", "2"], fontsize=8)
    axes[1].set_ylabel("raw\nframes", rotation=0, ha="right", va="center")

    axes[2].imshow(target_np[sl][None, :], aspect="auto", interpolation="nearest", cmap=state_cmap, norm=state_norm, extent=[start, end, 0, 1])
    axes[2].set_ylabel("true\n24-state", rotation=0, ha="right", va="center")
    axes[3].imshow(pred_np[sl][None, :], aspect="auto", interpolation="nearest", cmap=state_cmap, norm=state_norm, extent=[start, end, 0, 1])
    axes[3].set_ylabel("model\n24-state", rotation=0, ha="right", va="center")

    axes[4].fill_between(x, 0, correct, step="pre", color="#1a9850", alpha=0.75, label="correct")
    axes[4].fill_between(x, correct, 1, step="pre", color="#d73027", alpha=0.75, label="wrong")
    axes[4].plot(x, confidence, color="black", lw=1.0, label="confidence")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("correct\nconf", rotation=0, ha="right", va="center")
    axes[4].legend(loc="upper right", fontsize=8, ncols=3)

    axes[5].plot(x, posteriors["start"], color="#00441b", lw=1.1, label="start")
    axes[5].plot(x, posteriors["stop"], color="#7f0000", lw=1.1, label="stop")
    axes[5].plot(x, posteriors["donor"], color="#762a83", lw=1.1, label="donor")
    axes[5].plot(x, posteriors["acceptor"], color="#1b7837", lw=1.1, label="acceptor")
    ymax = max(0.08, max(float(values.max()) if values.size else 0.0 for values in posteriors.values()) * 1.15)
    axes[5].set_ylim(-0.02, ymax)
    axes[5].set_ylabel("transition\nposteriors", rotation=0, ha="right", va="center")
    axes[5].legend(loc="upper right", fontsize=8, ncols=4)

    donor = np.zeros(len(gene.dna), dtype=float)
    acceptor = np.zeros(len(gene.dna), dtype=float)
    for pos in gene.donor_positions:
        donor[pos] = 1.0
    for pos in gene.acceptor_positions:
        acceptor[pos] = 1.0
    axes[6].plot(x, donor[sl], color="#762a83", lw=1.5, label="donor target")
    axes[6].plot(x, acceptor[sl], color="#1b7837", lw=1.5, label="acceptor target")
    axes[6].set_ylim(-0.05, 1.05)
    axes[6].set_ylabel("splice\nlabels", rotation=0, ha="right", va="center")
    axes[6].legend(loc="upper right", fontsize=8)

    markers = [
        (gene.start_codon_start, "#00441b", "start"),
        (gene.stop_codon_start, "#7f0000", "stop codon"),
        (gene.stop_transition_position, "#b2182b", "stop transition"),
    ]
    for ax in axes:
        ax.set_xlim(start, end)
        for pos, color, _label in markers:
            if start <= pos < end:
                ax.axvline(pos, color=color, lw=0.8, alpha=0.6)
    for ax in (axes[0], axes[2], axes[3]):
        ax.set_yticks([])
    if show_bases and end - start <= 240:
        for pos in range(start, end):
            axes[0].text(pos + 0.5, 0.5, gene.dna[pos], ha="center", va="center", fontsize=5.5, color="black", clip_on=True)

    accuracy = float((pred == target_cpu).float().mean().item())
    exact = bool(torch.equal(pred, target_cpu))
    fig.suptitle(
        f"{title} | genome={len(gene.dna)} bp, exons={len(gene.exon_lengths)}, "
        f"introns={len(gene.intron_lengths)}, UTR5={gene.utr5_length} bp, "
        f"start={gene.start_codon_start}, stop={gene.stop_codon_start}, "
        f"accuracy={accuracy:.3f}, exact={exact}",
        y=1.02,
    )
    axes[-1].set_xlabel("genomic coordinate in synthetic gene")

    region_handles = [Patch(color=color, label=name) for name, color in zip(REGION_NAMES, REGION_COLORS)]
    codon_handles = [Patch(color="#1a9850", label="raw ATG"), Patch(color="#d73027", label="raw stop")]
    fig.legend(handles=region_handles + codon_handles, loc="lower center", ncols=8, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "length": len(gene.dna),
        "exons": len(gene.exon_lengths),
        "introns": len(gene.intron_lengths),
        "accuracy": accuracy,
        "exact": exact,
        "start_codon_start": int(gene.start_codon_start),
        "stop_codon_start": int(gene.stop_codon_start),
        "stop_transition_position": int(gene.stop_transition_position),
        "utr5_length": int(gene.utr5_length),
        "utr3_length": int(gene.utr3_length),
        "plot_start": int(start),
        "plot_end": int(end),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dense transition phase helper predictions.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dense_transition_phase_plots")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--zoom-pad", type=int, default=120)
    parser.add_argument("--window-bases", type=int, default=2200)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--conv-layers", type=int, default=None)
    parser.add_argument("--use-splice-tracks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--materialize-transitions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-utr5-length", type=int, default=None)
    parser.add_argument("--max-utr5-length", type=int, default=None)
    parser.add_argument("--min-coding-codons", type=int, default=None)
    parser.add_argument("--max-coding-codons", type=int, default=None)
    parser.add_argument("--min-utr3-length", type=int, default=None)
    parser.add_argument("--max-utr3-length", type=int, default=None)
    parser.add_argument("--min-exons", type=int, default=None)
    parser.add_argument("--max-exons", type=int, default=None)
    parser.add_argument("--min-exon-length", type=int, default=None)
    parser.add_argument("--min-intron-length", type=int, default=None)
    parser.add_argument("--max-intron-length", type=int, default=None)
    parser.add_argument("--intron-mod", choices=("any", "0", "1", "2", "nonzero"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    module = import_training_script(args.train_script)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    model_args = merged_args(ckpt_args, args)
    model = module.DenseTransitionPhaseModel(
        hidden_dim=model_args.hidden_dim,
        conv_layers=model_args.conv_layers,
        use_splice_tracks=model_args.use_splice_tracks,
        materialize_transitions=model_args.materialize_transitions,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = random.Random(args.seed)
    summary = {"checkpoint": str(args.checkpoint), "output_dir": str(args.output_dir), "examples": []}
    with torch.no_grad():
        for example_index in range(args.examples):
            gene = module.make_gene(model_args, rng)
            dna_one_hot = gene.dna_one_hot.to(device)
            splice_tracks = gene.splice_tracks.to(device)
            target = gene.target_states.to(device)
            output, _evidence_logits = model(dna_one_hot, splice_tracks if model_args.use_splice_tracks else None)

            base_name = f"example_{example_index:02d}"
            overview_end = min(len(gene.dna), args.window_bases)
            info = plot_example(
                module=module,
                gene=gene,
                output=output,
                target=target,
                output_path=args.output_dir / f"{base_name}_overview.png",
                title=f"{base_name} overview",
                start=0,
                end=overview_end,
            )
            plot_example(
                module=module,
                gene=gene,
                output=output,
                target=target,
                output_path=args.output_dir / f"{base_name}_start_zoom.png",
                title=f"{base_name} start zoom",
                start=gene.start_codon_start - args.zoom_pad,
                end=gene.start_codon_start + args.zoom_pad,
                show_bases=True,
                zoom=True,
            )
            plot_example(
                module=module,
                gene=gene,
                output=output,
                target=target,
                output_path=args.output_dir / f"{base_name}_stop_zoom.png",
                title=f"{base_name} stop zoom",
                start=gene.stop_codon_start - args.zoom_pad,
                end=gene.stop_codon_start + args.zoom_pad,
                show_bases=True,
                zoom=True,
            )
            for junction_index, (start, end, label) in enumerate(junction_windows(gene, args.zoom_pad, limit=4)):
                plot_example(
                    module=module,
                    gene=gene,
                    output=output,
                    target=target,
                    output_path=args.output_dir / f"{base_name}_junction_{junction_index:02d}_zoom.png",
                    title=f"{base_name} {label} zoom",
                    start=start,
                    end=end,
                    show_bases=True,
                    zoom=True,
                )
            summary["examples"].append(info)

    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"saved plots to: {args.output_dir}")
    print(f"saved summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
