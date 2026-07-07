from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


DEFAULT_TRAIN_SCRIPT = ROOT / "scripts" / "train_structured_phase_helper.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "structured_phase_mamba_longer_genes_split_mix" / "best.pt"

STATE_NAMES = ("N", "C0", "C1", "C2", "T")
STATE_COLORS = ("#bdbdbd", "#1b9e77", "#d95f02", "#7570b3", "#525252")
REGION_NAMES = ("intron", "5' UTR", "C0", "C1", "C2", "3' UTR")
REGION_COLORS = ("#f0f0f0", "#80cdc1", "#1b9e77", "#d95f02", "#7570b3", "#c2a5cf")


def import_training_script(path: Path):
    spec = importlib.util.spec_from_file_location("structured_phase_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["structured_phase_train"] = module
    spec.loader.exec_module(module)
    return module


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def merged_args(checkpoint_args: dict[str, object], cli_args: argparse.Namespace) -> argparse.Namespace:
    defaults = vars(module_parse_defaults())
    defaults.update(checkpoint_args)
    for key in (
        "mode",
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
        "allow_split_start_stop",
        "require_split_codon",
        "unsplit_codon_fraction",
        "evidence_model",
        "mamba_hidden_dim",
        "mamba_layers",
        "mamba_chunk_size",
        "mamba_headdim",
        "bidirectional",
        "min_orf_codons",
    ):
        value = getattr(cli_args, key, None)
        if value is not None:
            defaults[key] = value
    defaults["mamba_cache_root"] = cli_args.mamba_cache_root
    defaults["checkpoint_dir"] = ROOT / "checkpoints" / "_plot_unused"
    defaults["init_from"] = None
    defaults["resume"] = False
    return argparse.Namespace(**defaults)


def module_parse_defaults() -> argparse.Namespace:
    # Keep this script independent of sys.argv while reusing the trainer's defaults.
    return argparse.Namespace(
        device="auto",
        seed=20260706,
        checkpoint_dir=ROOT / "checkpoints" / "_plot_unused",
        init_from=None,
        resume=False,
        steps=1,
        examples_per_step=1,
        validation_examples=1,
        print_every=1,
        validate_every=1,
        checkpoint_every=1,
        lr=0.0,
        weight_decay=0.0,
        grad_clip=0.0,
        min_orf_codons=2,
        start_loss_weight=0.25,
        stop_loss_weight=0.25,
        mode="multiexon",
        evidence_model="mamba",
        mamba_hidden_dim=64,
        mamba_layers=4,
        mamba_chunk_size=16,
        mamba_headdim=8,
        mamba_cache_root=None,
        bidirectional=True,
        min_utr5_length=20,
        max_utr5_length=300,
        min_coding_codons=20,
        max_coding_codons=200,
        min_utr3_length=20,
        max_utr3_length=500,
        min_exons=2,
        max_exons=12,
        min_exon_length=6,
        min_intron_length=50,
        max_intron_length=300,
        allow_split_start_stop=True,
        require_split_codon="none",
        unsplit_codon_fraction=0.0,
        init_textbook=False,
    )


def region_codes(gene, target: torch.Tensor) -> np.ndarray:
    exonic = np.zeros(len(gene.dna), dtype=bool)
    exonic[gene.paths[0].genomic_indices.cpu().numpy()] = True
    target_np = target.cpu().numpy()
    codes = np.zeros(len(gene.dna), dtype=int)
    codes[exonic & (target_np == 0)] = 1
    codes[exonic & (target_np == 1)] = 2
    codes[exonic & (target_np == 2)] = 3
    codes[exonic & (target_np == 3)] = 4
    codes[exonic & (target_np == 4)] = 5
    return codes


def exon_spans(gene) -> list[tuple[int, int]]:
    path = gene.paths[0].genomic_indices.tolist()
    spans = []
    start = path[0]
    prev = path[0]
    for index in path[1:]:
        if index != prev + 1:
            spans.append((start, prev + 1))
            start = index
        prev = index
    spans.append((start, prev + 1))
    return spans


def split_status(module, gene) -> tuple[bool, bool]:
    return module.split_codon_flags(gene)


def first_split_window(module, gene, pad: int) -> tuple[int, int] | None:
    start_split, stop_split = split_status(module, gene)
    if start_split:
        return max(0, gene.start_codon_start - pad), min(len(gene.dna), gene.start_codon_start + pad)
    if stop_split:
        return max(0, gene.stop_codon_start - pad), min(len(gene.dna), gene.stop_codon_start + pad)
    return None


def junction_windows(gene, pad: int, limit: int = 2) -> list[tuple[int, int, str]]:
    spans = exon_spans(gene)
    windows = []
    for junction_index, (left, right) in enumerate(zip(spans[:-1], spans[1:])):
        donor = left[1] - 1
        acceptor = right[0]
        windows.append((max(0, donor - pad), min(len(gene.dna), donor + pad), f"donor {junction_index}"))
        windows.append((max(0, acceptor - pad), min(len(gene.dna), acceptor + pad), f"acceptor {junction_index}"))
        if len(windows) >= limit:
            break
    return windows


def plot_example(
    *,
    module,
    gene,
    output,
    target: torch.Tensor,
    output_path: Path,
    title: str,
    start: int = 0,
    end: int | None = None,
    show_bases: bool = False,
    zoom: bool = False,
) -> dict[str, float | int | bool]:
    if end is None:
        end = len(gene.dna)
    start = max(0, int(start))
    end = min(len(gene.dna), int(end))
    sl = slice(start, end)
    x = np.arange(start, end)

    pred = output.state_log_probs.argmax(dim=-1).detach().cpu()
    probs = output.state_probs.detach().cpu()
    confidence = probs.max(dim=-1).values.numpy()[sl]
    target_np = target.cpu().numpy()
    pred_np = pred.numpy()
    correct = (target_np[sl] == pred_np[sl]).astype(float)
    init_post = output.initiation_log_probs.exp().detach().cpu().numpy()[sl]
    term_post = output.termination_log_probs.exp().detach().cpu().numpy()[sl]
    regions = region_codes(gene, target)[sl]

    state_cmap = ListedColormap(STATE_COLORS)
    state_norm = BoundaryNorm(np.arange(-0.5, len(STATE_NAMES) + 0.5), state_cmap.N)
    region_cmap = ListedColormap(REGION_COLORS)
    region_norm = BoundaryNorm(np.arange(-0.5, len(REGION_NAMES) + 0.5), region_cmap.N)

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16 if zoom else 18, 8.2 if zoom else 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 0.45, 0.45, 0.75, 0.75, 0.9]},
    )

    axes[0].imshow(
        regions[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=region_cmap,
        norm=region_norm,
        extent=[start, end, 0, 1],
    )
    for exon_start, exon_end in exon_spans(gene):
        if exon_end >= start and exon_start <= end:
            axes[0].axvspan(max(start, exon_start), min(end, exon_end), ymin=0.05, ymax=0.95, fill=False, edgecolor="black", lw=0.7)
            if zoom:
                if exon_start < start:
                    axes[0].text(start, 1.05, "exon continues", ha="left", va="bottom", fontsize=7, color="black")
                if exon_end > end:
                    axes[0].text(end, 1.05, "exon continues", ha="right", va="bottom", fontsize=7, color="black")
    if start <= gene.start_codon_start < end:
        axes[0].axvline(gene.start_codon_start, color="#00441b", lw=1.6, label="start")
    if start <= gene.stop_codon_start < end:
        axes[0].axvline(gene.stop_codon_start, color="#7f0000", lw=1.6, label="stop")
    axes[0].set_ylabel("gene\nfeatures", rotation=0, ha="right", va="center")

    axes[1].imshow(target_np[sl][None, :], aspect="auto", interpolation="nearest", cmap=state_cmap, norm=state_norm, extent=[start, end, 0, 1])
    axes[1].set_ylabel("true\nphase", rotation=0, ha="right", va="center")
    axes[2].imshow(pred_np[sl][None, :], aspect="auto", interpolation="nearest", cmap=state_cmap, norm=state_norm, extent=[start, end, 0, 1])
    axes[2].set_ylabel("model\nphase", rotation=0, ha="right", va="center")

    axes[3].fill_between(x, 0, correct, step="pre", color="#1a9850", alpha=0.75, label="correct")
    axes[3].fill_between(x, correct, 1, step="pre", color="#d73027", alpha=0.75, label="wrong")
    axes[3].plot(x, confidence, color="black", lw=1.0, label="confidence")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel("correct\nconf", rotation=0, ha="right", va="center")
    axes[3].legend(loc="upper right", fontsize=8, ncols=3)

    axes[4].plot(x, init_post, color="#00441b", lw=1.2, label="start posterior")
    axes[4].plot(x, term_post, color="#7f0000", lw=1.2, label="stop posterior")
    axes[4].set_ylim(-0.02, max(0.08, float(max(init_post.max(initial=0), term_post.max(initial=0))) * 1.15))
    axes[4].set_ylabel("ORF\nposts", rotation=0, ha="right", va="center")
    axes[4].legend(loc="upper right", fontsize=8)

    path_set = set(gene.paths[0].genomic_indices.tolist())
    donor = np.zeros(len(gene.dna), dtype=float)
    acceptor = np.zeros(len(gene.dna), dtype=float)
    path = gene.paths[0].genomic_indices.tolist()
    for offset in range(len(path) - 1):
        if path[offset + 1] != path[offset] + 1:
            donor[path[offset]] = 1.0
            acceptor[path[offset + 1]] = 1.0
    axes[5].plot(x, donor[sl], color="#762a83", lw=1.5, label="donor")
    axes[5].plot(x, acceptor[sl], color="#1b7837", lw=1.5, label="acceptor")
    axes[5].set_ylim(-0.05, 1.05)
    axes[5].set_ylabel("splice\nsites", rotation=0, ha="right", va="center")
    axes[5].legend(loc="upper right", fontsize=8)

    for ax in axes[:3]:
        ax.set_yticks([])
    for ax in axes:
        ax.set_xlim(start, end)
        if start <= gene.start_codon_start < end:
            ax.axvline(gene.start_codon_start, color="#00441b", lw=0.8, alpha=0.6)
        if start <= gene.stop_codon_start < end:
            ax.axvline(gene.stop_codon_start, color="#7f0000", lw=0.8, alpha=0.6)
    if show_bases and end - start <= 220:
        for pos in range(start, end):
            color = "black" if pos in path_set else "#9e9e9e"
            axes[0].text(pos + 0.5, 0.5, gene.dna[pos], ha="center", va="center", fontsize=5.5, color=color, clip_on=True)

    start_split, stop_split = split_status(module, gene)
    accuracy = float((pred == target.cpu()).float().mean().item())
    exact = bool(torch.equal(pred, target.cpu()))
    fig.suptitle(
        f"{title} | genome={len(gene.dna)} bp, exons={len(gene.exon_lengths)}, "
        f"introns={len(gene.intron_lengths)}, split start={start_split}, split stop={stop_split}, "
        f"accuracy={accuracy:.3f}, exact={exact}",
        y=1.02,
    )
    axes[-1].set_xlabel("genomic coordinate in synthetic gene")

    region_handles = [Patch(color=color, label=name) for name, color in zip(REGION_NAMES, REGION_COLORS)]
    state_handles = [Patch(color=color, label=name) for name, color in zip(STATE_NAMES, STATE_COLORS)]
    fig.legend(handles=region_handles + state_handles, loc="lower center", ncols=6, bbox_to_anchor=(0.5, -0.09), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "length": len(gene.dna),
        "exons": len(gene.exon_lengths),
        "introns": len(gene.intron_lengths),
        "start_split": start_split,
        "stop_split": stop_split,
        "accuracy": accuracy,
        "exact": exact,
        "start_codon_start": int(gene.start_codon_start),
        "stop_codon_start": int(gene.stop_codon_start),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot structured phase helper predictions for presentation figures.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "structured_phase_presentation_plots")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mamba-cache-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--zoom-pad", type=int, default=120)
    parser.add_argument("--window-bases", type=int, default=700)

    parser.add_argument("--mode", choices=("single_exon", "multiexon"), default=None)
    parser.add_argument("--evidence-model", choices=("motif", "mamba"), default=None)
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
    parser.add_argument("--allow-split-start-stop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-split-codon", choices=("none", "any", "start", "stop"), default=None)
    parser.add_argument("--unsplit-codon-fraction", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    module = import_training_script(args.train_script)
    cache_root = module.configure_mamba_cache(args.mamba_cache_root)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    model_args = merged_args(ckpt_args, args)
    model_args.mamba_cache_root = cache_root
    model = module.build_model(model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = __import__("random").Random(args.seed)
    summary = {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "examples": [],
    }
    with torch.no_grad():
        for example_index in range(args.examples):
            gene = module.make_gene(model_args, rng)
            dna_one_hot = gene.dna_one_hot.to(device)
            target = gene.target_states.to(device)
            output = module.run_model_on_gene(model, model_args, gene, dna_one_hot)

            overview_window = (0, min(len(gene.dna), args.window_bases))
            if len(gene.dna) > args.window_bases:
                centre = max(0, min(len(gene.dna), gene.start_codon_start + args.window_bases // 2))
                overview_window = (
                    max(0, centre - args.window_bases // 2),
                    min(len(gene.dna), centre + args.window_bases // 2),
                )
            base_name = f"example_{example_index:02d}"
            info = plot_example(
                module=module,
                gene=gene,
                output=output,
                target=target,
                output_path=args.output_dir / f"{base_name}_overview.png",
                title=f"{base_name} overview",
                start=overview_window[0],
                end=overview_window[1],
            )
            plot_example(
                module=module,
                gene=gene,
                output=output,
                target=target,
                output_path=args.output_dir / f"{base_name}_start_zoom.png",
                title=f"{base_name} start codon zoom",
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
                title=f"{base_name} stop codon zoom",
                start=gene.stop_codon_start - args.zoom_pad,
                end=gene.stop_codon_start + args.zoom_pad,
                show_bases=True,
                zoom=True,
            )
            split_window = first_split_window(module, gene, args.zoom_pad)
            if split_window is not None:
                plot_example(
                    module=module,
                    gene=gene,
                    output=output,
                    target=target,
                    output_path=args.output_dir / f"{base_name}_split_codon_zoom.png",
                    title=f"{base_name} split codon zoom",
                    start=split_window[0],
                    end=split_window[1],
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
                    title=f"{base_name} splice junction zoom {label}",
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
