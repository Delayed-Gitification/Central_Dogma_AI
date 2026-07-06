from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT = ROOT / "scripts" / "train_synthetic_splice_hybrid_phase_track.py"
DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "synthetic_splice_phase_track_hybrid_carried_bidir_l6_attn2"
    / "best.pt"
)


STATE_COLORS = [
    "#d73027",
    "#fc8d59",
    "#fee08b",
    "#4575b4",
    "#91bfdb",
    "#e0f3f8",
]
COLLAPSED_COLORS = ["#d73027", "#fc8d59", "#fee08b", "#d9d9d9"]


def import_training_script(path: Path):
    spec = importlib.util.spec_from_file_location("phase_track_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase_track_module"] = module
    spec.loader.exec_module(module)
    return module


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_model(module, checkpoint: dict[str, object], device: torch.device) -> torch.nn.Module:
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = {}
    kwargs = {
        "hidden_dim": args.get("hidden_dim", 64),
        "layers": args.get("layers", 3),
        "chunk_size": args.get("chunk_size", 32),
        "headdim": args.get("headdim", 8),
        "local_conv_kernel": args.get("local_conv_kernel", 9),
        "head_conv_kernel": args.get("head_conv_kernel", 7),
        "bidirectional": args.get("bidirectional", False),
    }
    if "attention_layers" in args or hasattr(module, "LocalWindowAttentionBlock"):
        kwargs.update(
            {
                "attention_layers": args.get("attention_layers", 2),
                "attention_heads": args.get("attention_heads", 4),
                "attention_window": args.get("attention_window", 129),
            }
        )
    model = module.MambaPhaseTrackPredictor(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def make_eval_batch(module, checkpoint: dict[str, object], device: torch.device, args: argparse.Namespace):
    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}
    return module.make_batch(
        batch_size=args.batch_size,
        device=device,
        min_protein_codons=args.min_protein_codons or ckpt_args.get("min_protein_codons", 24),
        max_protein_codons=args.max_protein_codons or ckpt_args.get("max_protein_codons", 96),
        min_exon_count=args.min_exon_count or ckpt_args.get("min_exon_count", 1),
        max_exon_count=args.max_exon_count or ckpt_args.get("max_exon_count", 21),
        min_exon_bases=args.min_exon_bases or ckpt_args.get("min_exon_bases", 3),
        max_exon_bases=args.max_exon_bases or ckpt_args.get("max_exon_bases", 300),
        median_exon_bases=args.median_exon_bases or ckpt_args.get("median_exon_bases", 50),
        rare_short_exon_prob=(
            args.rare_short_exon_prob
            if args.rare_short_exon_prob is not None
            else ckpt_args.get("rare_short_exon_prob", 0.05)
        ),
        exon_length_mode=args.exon_length_mode or ckpt_args.get("exon_length_mode", "sampled"),
        min_intron_length=args.min_intron_length or ckpt_args.get("min_intron_length", 50),
        max_intron_length=args.max_intron_length or ckpt_args.get("max_intron_length", 300),
        length_bucket_size=args.length_bucket_size or ckpt_args.get("length_bucket_size", 4096),
        seed=args.seed,
    )


def collapse_to_4(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x < 3, x, torch.full_like(x, 3))


def first_error_window(
    *,
    example_index: int,
    pred: torch.Tensor,
    targets: torch.Tensor,
    genome_mask: torch.Tensor,
    genome_length: int,
    pad: int,
    collapsed: bool,
) -> tuple[int, int]:
    if collapsed:
        bad = (collapse_to_4(pred[example_index]) != collapse_to_4(targets[example_index])) & genome_mask[example_index]
    else:
        bad = (pred[example_index] != targets[example_index]) & genome_mask[example_index]
    positions = torch.nonzero(bad, as_tuple=False).flatten().detach().cpu().numpy()
    if len(positions) == 0:
        return 0, min(genome_length, 600)
    centre = int(positions[0])
    return max(0, centre - pad), min(genome_length, centre + pad)


def plot_phase(
    *,
    module,
    example_index: int,
    start: int,
    end: int,
    examples: list[dict[str, object]],
    splice_tracks: torch.Tensor,
    targets: torch.Tensor,
    pred: torch.Tensor,
    probs: torch.Tensor,
    output_path: Path,
) -> None:
    example = examples[example_index]
    genome_length = len(str(example["genome"]))
    start = max(0, int(start))
    end = min(genome_length, int(end))
    sl = slice(start, end)
    x = np.arange(start, end)

    state_cmap = ListedColormap(STATE_COLORS)
    state_norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), state_cmap.N)
    collapsed_cmap = ListedColormap(COLLAPSED_COLORS)
    collapsed_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), collapsed_cmap.N)

    true_6 = targets[example_index, sl].detach().cpu().numpy()
    pred_6 = pred[example_index, sl].detach().cpu().numpy()
    true_4 = collapse_to_4(targets[example_index, sl]).detach().cpu().numpy()
    pred_4 = collapse_to_4(pred[example_index, sl]).detach().cpu().numpy()
    confidence = probs[example_index, sl].max(dim=-1).values.detach().cpu().numpy()
    donor = splice_tracks[example_index, sl, 0].detach().cpu().numpy()
    acceptor = splice_tracks[example_index, sl, 1].detach().cpu().numpy()
    correct_6 = (true_6 == pred_6).astype(float)

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(18, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [0.45, 0.45, 0.45, 0.45, 1.0, 1.0]},
    )
    axes[0].imshow(true_6[None, :], aspect="auto", cmap=state_cmap, norm=state_norm, extent=[start, end, 0, 1])
    axes[0].set_ylabel("true\n6-state", rotation=0, ha="right", va="center")
    axes[1].imshow(pred_6[None, :], aspect="auto", cmap=state_cmap, norm=state_norm, extent=[start, end, 0, 1])
    axes[1].set_ylabel("pred\n6-state", rotation=0, ha="right", va="center")
    axes[2].imshow(true_4[None, :], aspect="auto", cmap=collapsed_cmap, norm=collapsed_norm, extent=[start, end, 0, 1])
    axes[2].set_ylabel("true\n4-track", rotation=0, ha="right", va="center")
    axes[3].imshow(pred_4[None, :], aspect="auto", cmap=collapsed_cmap, norm=collapsed_norm, extent=[start, end, 0, 1])
    axes[3].set_ylabel("pred\n4-track", rotation=0, ha="right", va="center")

    axes[4].fill_between(x, 0, correct_6, step="pre", color="#1a9850", alpha=0.75, label="6-state correct")
    axes[4].fill_between(x, correct_6, 1, step="pre", color="#d73027", alpha=0.75, label="6-state wrong")
    axes[4].plot(x, confidence, color="black", lw=1.0, label="confidence")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("correct\nconf", rotation=0, ha="right", va="center")
    axes[4].legend(loc="upper right", ncols=3, fontsize=8)

    axes[5].plot(x, donor, color="#762a83", lw=1.5, label="donor track")
    axes[5].plot(x, acceptor, color="#1b7837", lw=1.5, label="acceptor track")
    axes[5].set_ylim(-0.05, 1.05)
    axes[5].set_ylabel("splice\ntracks", rotation=0, ha="right", va="center")
    axes[5].legend(loc="upper right", fontsize=8)

    for ax in axes[:4]:
        ax.set_yticks([])
    axes[-1].set_xlabel("genomic position in synthetic example")
    fig.suptitle(
        f"Example {example_index}: genome={genome_length} bp, exons={example['exon_count']}, "
        f"protein={example['protein_codons']} aa, window={start}:{end}",
        y=1.02,
    )

    state_handles = [Patch(color=color, label=label) for color, label in zip(STATE_COLORS, module.PHASE_LABELS)]
    collapsed_handles = [
        Patch(color=color, label=label)
        for color, label in zip(COLLAPSED_COLORS, ["phase0", "phase1", "phase2", "not coding"])
    ]
    fig.legend(handles=state_handles + collapsed_handles, loc="lower center", ncols=5, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot carried-phase predictions from a synthetic splice checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "phase_track_plots")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--zoom-pad", type=int, default=140)
    parser.add_argument("--min-protein-codons", type=int, default=0)
    parser.add_argument("--max-protein-codons", type=int, default=0)
    parser.add_argument("--min-exon-count", type=int, default=0)
    parser.add_argument("--max-exon-count", type=int, default=0)
    parser.add_argument("--min-exon-bases", type=int, default=0)
    parser.add_argument("--max-exon-bases", type=int, default=0)
    parser.add_argument("--median-exon-bases", type=int, default=0)
    parser.add_argument("--rare-short-exon-prob", type=float, default=None)
    parser.add_argument("--exon-length-mode", choices=("split", "sampled"), default="")
    parser.add_argument("--min-intron-length", type=int, default=0)
    parser.add_argument("--max-intron-length", type=int, default=0)
    parser.add_argument("--length-bucket-size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    module = import_training_script(args.script)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(module, checkpoint, device)
    batch = make_eval_batch(module, checkpoint, device, args)
    dna, splice_tracks, *_unused, examples = batch
    targets, genome_mask = module.phase_targets_from_examples(
        examples,
        max_length=dna.shape[1],
        device=device,
    )

    with torch.no_grad():
        logits, encoded = model(dna, splice_tracks)
        probs = logits.softmax(dim=-1)
        pred = logits.argmax(dim=-1)

    metrics = module.phase_metrics(logits, targets, genome_mask)
    summary = {
        "checkpoint": str(args.checkpoint),
        "script": str(args.script),
        "device": str(device),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_metrics": checkpoint.get("final_metrics"),
        "eval_metrics": metrics,
        "encoded_shape": list(encoded.shape),
        "logits_shape": list(logits.shape),
        "examples": [],
    }

    max_examples = min(args.examples, len(examples))
    for example_index in range(max_examples):
        example = examples[example_index]
        genome_length = len(str(example["genome"]))
        real_mask = genome_mask[example_index]
        wrong_6 = int(((pred[example_index] != targets[example_index]) & real_mask).sum().item())
        wrong_4 = int(
            (
                (collapse_to_4(pred[example_index]) != collapse_to_4(targets[example_index]))
                & real_mask
            )
            .sum()
            .item()
        )
        example_summary = {
            "example_index": example_index,
            "genome_length": genome_length,
            "protein_codons": int(example["protein_codons"]),
            "exon_count": int(example["exon_count"]),
            "exon_lengths": list(map(int, example["exon_lengths"])),
            "intron_lengths": list(map(int, example["intron_lengths"])),
            "wrong_6_state_bases": wrong_6,
            "wrong_collapsed_4_track_bases": wrong_4,
        }
        summary["examples"].append(example_summary)

        full_end = min(genome_length, 5000)
        plot_phase(
            module=module,
            example_index=example_index,
            start=0,
            end=full_end,
            examples=examples,
            splice_tracks=splice_tracks,
            targets=targets,
            pred=pred,
            probs=probs,
            output_path=args.output_dir / f"example_{example_index:02d}_full.png",
        )
        zoom_start, zoom_end = first_error_window(
            example_index=example_index,
            pred=pred,
            targets=targets,
            genome_mask=genome_mask,
            genome_length=genome_length,
            pad=args.zoom_pad,
            collapsed=False,
        )
        plot_phase(
            module=module,
            example_index=example_index,
            start=zoom_start,
            end=zoom_end,
            examples=examples,
            splice_tracks=splice_tracks,
            targets=targets,
            pred=pred,
            probs=probs,
            output_path=args.output_dir / f"example_{example_index:02d}_first_6_state_error.png",
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"output_dir": str(args.output_dir), "summary": str(summary_path), "eval_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
