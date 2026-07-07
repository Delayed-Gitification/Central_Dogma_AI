import argparse
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Ensure matplotlib runs headlessly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_dense_transition_phase_helper_forward_backward import (
    StackedDenseTransitionPhaseModel, make_gene, batch_to_device, 
    S, K, STATE_NAMES, TYPE_TO_INDEX
)

def main():
    parser = argparse.ArgumentParser(description="Headless prediction plotter.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--output", type=str, default="prediction_plot.png", help="Output plot filename")
    parser.add_argument("--num-examples", type=int, default=4, help="Number of random examples to plot")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint from {args.checkpoint} on {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Reconstruct model parameters
    ckpt_args = checkpoint["args"]
    model = StackedDenseTransitionPhaseModel(
        hidden_dim=ckpt_args["hidden_dim"],
        conv_layers=ckpt_args["conv_layers"],
        use_splice_tracks=ckpt_args["use_splice_tracks"],
        materialize_transitions=ckpt_args["materialize_transitions"]
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy CLI args block for make_gene to match checkpoint settings
    class ConfigArgs:
        pass
    cfg = ConfigArgs()
    for k, v in ckpt_args.items():
        setattr(cfg, k, v)

    import random
    rng = random.Random(42)
    output_path = Path(args.output)

    for i in range(args.num_examples):
        print(f"Generating synthetic gene {i}...")
        gene = make_gene(cfg, rng)
        batch = batch_to_device([gene], device)

        # Run forward pass
        with torch.no_grad():
            _, (output, _) = model(batch.dna_one_hot, batch.splice_tracks if cfg.use_splice_tracks else None)

        # Extract sequences to CPU
        true_states = batch.target_states[0].cpu().numpy()
        pred_states = output.state_log_probs[0].argmax(dim=-1).cpu().numpy()
        
        # Map 24 HMM states to 4 biological regions: UTR5=0, Coding Exon=1, Intron=2, UTR3=3
        # to prevent Mod-3 saw-tooth wave jitter in line plot
        def states_to_regions(states):
            regions = np.zeros_like(states)
            regions[(states >= 3) & (states <= 11)] = 1   # Exon (CDS)
            regions[(states >= 12) & (states <= 20)] = 2  # Intron
            regions[(states >= 21) & (states <= 23)] = 3  # UTR3
            return regions
            
        true_regions = states_to_regions(true_states)
        pred_regions = states_to_regions(pred_states)
        
        start_pos = output.start_posterior[0].cpu().numpy()
        stop_pos = output.stop_posterior[0].cpu().numpy()
        donor_pos = output.donor_posterior[0].cpu().numpy()
        acceptor_pos = output.acceptor_posterior[0].cpu().numpy()

        length = len(true_states)
        x = np.arange(length)

        print(f"Plotting results for example {i}...")
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 2]})

        # Subplot 1: True vs Predicted Regions (step plot)
        axes[0].step(x, true_regions, where="post", label="True Region", color="black", alpha=0.7)
        axes[0].step(x, pred_regions, where="post", label="Predicted Region", color="crimson", linestyle="--", alpha=0.8)
        axes[0].set_yticks([0, 1, 2, 3])
        axes[0].set_yticklabels(["5' UTR", "Coding Exon", "Intron", "3' UTR"])
        axes[0].set_ylabel("Transcript Region")
        axes[0].set_title("Gene Structural Annotation")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Subplot 2: Transition Signal Posteriors
        axes[1].plot(x, start_pos, label="Start Codon Posterior", color="forestgreen", alpha=0.8)
        axes[1].plot(x, stop_pos, label="Stop Codon Posterior", color="darkorange", alpha=0.8)
        axes[1].plot(x, donor_pos, label="Donor Splice Posterior", color="royalblue", alpha=0.8)
        axes[1].plot(x, acceptor_pos, label="Acceptor Splice Posterior", color="purple", alpha=0.8)

        # Plot Ground Truth positions as vertical lines on ALL subplots for maximum visibility
        for ax in axes:
            ax.axvline(gene.start_codon_start, color="forestgreen", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.axvline(gene.stop_transition_position, color="darkorange", linestyle="--", linewidth=1.5, alpha=0.7)
            for d_pos in gene.donor_positions:
                ax.axvline(d_pos, color="royalblue", linestyle="--", linewidth=1.5, alpha=0.7)
            for a_pos in gene.acceptor_positions:
                ax.axvline(a_pos, color="purple", linestyle="--", linewidth=1.5, alpha=0.7)

        # Add legend handles for vertical lines on subplot 2
        axes[1].plot([], [], color="forestgreen", linestyle="--", label="True Start")
        axes[1].plot([], [], color="darkorange", linestyle="--", label="True Stop")
        axes[1].plot([], [], color="royalblue", linestyle="--", label="True Donor")
        axes[1].plot([], [], color="purple", linestyle="--", label="True Acceptor")

        axes[1].set_ylabel("Probability")
        axes[1].set_title("Boundary Posteriors & True Coordinates (dashed lines)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # Subplot 3: State-by-state probabilities (Heatmap)
        state_probs = output.state_probs[0].cpu().numpy() # (L, S)
        im = axes[2].imshow(state_probs.T, aspect='auto', cmap='magma', interpolation='nearest')
        axes[2].set_ylabel("State Index")
        axes[2].set_xlabel("Nucleotide Position (bp)")
        axes[2].set_title("Full Posterior State Probabilities Heatmap")
        fig.colorbar(im, ax=axes[2], orientation='horizontal', pad=0.15, label="Probability")

        plt.tight_layout()
        this_output = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
        plt.savefig(this_output, dpi=150)
        plt.close(fig)
        print(f"Saved plot successfully to {this_output}")

if __name__ == "__main__":
    main()
