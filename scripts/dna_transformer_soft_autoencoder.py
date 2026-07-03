#!/usr/bin/env python3
"""
DNA Transformer Soft Autoencoder (v1)
=====================================

Architecture:
  DNA → CNN Encoder → Latent Memory [B, M, D]
  Latent Memory → Causal Transformer Decoder → A/C/G/T logits

Training mode:  Teacher forcing (standard autoregressive training)
Design mode:    Soft autoregressive decoding (fully differentiable, no hard sampling)

Key design:
  - Each position outputs P(A), P(C), P(G), P(T)
  - In soft mode, previous soft distribution is embedded via: probs @ embedding_matrix
  - This keeps the entire path differentiable for downstream gradient-based design
  - Fixed max_len for v1 (no EOS complexity)
"""

import argparse
import difflib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

BASES = "ACGT"


# ─── Encoder ────────────────────────────────────────────────────────────────

class DNAEncoder(nn.Module):
    """CNN encoder: (B, 4, L) one-hot DNA → (B, M, D) latent memory tokens."""

    def __init__(self, d_model: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=stride * 2, stride=stride, padding=stride // 2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).transpose(1, 2)  # (B, M, D)


# ─── Transformer Decoder ────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block: causal self-attn → cross-attn → FFN."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        h = self.self_attn(h, h, h, attn_mask=causal_mask)[0]
        x = x + h

        h = self.ln2(x)
        h = self.cross_attn(h, memory, memory)[0]
        x = x + h

        h = self.ln3(x)
        x = x + self.ffn(h)
        return x


class DNATransformerDecoder(nn.Module):
    """
    Causal transformer decoder conditioned on latent memory.

    Vocab: A(0), C(1), G(2), T(3), BOS(4).
    Output head projects to 4 classes (DNA bases only, no EOS for v1).
    """

    VOCAB_SIZE = 5   # A, C, G, T, BOS
    BOS_ID = 4

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(self.VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 4)  # A/C/G/T only

    # ── helpers ──────────────────────────────────────────────────────────────

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Float causal mask: 0 = attend, -inf = block. MPS-safe."""
        mask = torch.zeros(length, length, device=device)
        mask.masked_fill_(torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1),
                          float("-inf"))
        return mask

    def _embed_and_position(self, seq: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to a (B, T, D) sequence of token embeddings."""
        T = seq.shape[1]
        return seq + self.pos_emb(torch.arange(T, device=seq.device))

    # ── forward modes ────────────────────────────────────────────────────────

    def forward_teacher_forced(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Standard teacher-forced training pass.

        target: (B, L) int64 ground-truth DNA indices 0-3
        memory: (B, M, D) latent memory from encoder
        returns: (B, L, 4) logits
        """
        B, L = target.shape

        # Shift right: input = [BOS, x_0, x_1, ..., x_{L-2}]
        bos = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=target.device)
        input_tokens = torch.cat([bos, target[:, :-1]], dim=1)   # (B, L)

        x = self._embed_and_position(self.token_emb(input_tokens))
        mask = self._causal_mask(L, target.device)

        for block in self.blocks:
            x = block(x, memory, causal_mask=mask)

        return self.out_proj(self.ln_final(x))  # (B, L, 4)

    def forward_soft_autoregressive(self, memory: torch.Tensor, max_len: int,
                                    temperature: float = 1.0) -> torch.Tensor:
        """
        Soft autoregressive decoding — fully differentiable.

        At each step: logits → softmax → probs @ embedding_matrix → next input.
        No hard sampling anywhere.

        memory: (B, M, D)
        returns: (B, max_len, 4) soft DNA probabilities
        """
        B, device = memory.shape[0], memory.device
        dna_emb_matrix = self.token_emb.weight[:4]  # (4, D) — only DNA bases

        # Start with BOS
        bos_emb = self.token_emb(torch.full((B,), self.BOS_ID, dtype=torch.long, device=device))
        raw_embeddings = [bos_emb.unsqueeze(1)]  # [(B, 1, D)]
        all_probs = []

        for t in range(max_len):
            seq = self._embed_and_position(torch.cat(raw_embeddings, dim=1))  # (B, t+1, D)
            mask = self._causal_mask(seq.shape[1], device)

            h = seq
            for block in self.blocks:
                h = block(h, memory, causal_mask=mask)
            h = self.ln_final(h)

            logits_t = self.out_proj(h[:, -1, :])            # (B, 4)
            probs_t = F.softmax(logits_t / temperature, dim=-1)  # (B, 4)
            all_probs.append(probs_t)

            # Soft embedding for the next step (no hard sampling)
            soft_emb = probs_t @ dna_emb_matrix              # (B, D)
            raw_embeddings.append(soft_emb.unsqueeze(1))

        return torch.stack(all_probs, dim=1)  # (B, max_len, 4)


# ─── Full Autoencoder ───────────────────────────────────────────────────────

class DNASoftAutoencoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 4,
                 dim_ff: int = 256, max_seq_len: int = 200,
                 target_bases_per_latent: int = 20, dropout: float = 0.1):
        super().__init__()
        self.encoder = DNAEncoder(d_model, stride=target_bases_per_latent)
        self.decoder = DNATransformerDecoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_ff=dim_ff, max_seq_len=max_seq_len, dropout=dropout,
        )
        self.max_seq_len = max_seq_len

    def forward(self, x_onehot: torch.Tensor, target_indices: torch.Tensor):
        """Teacher-forced pass. Returns (logits, memory)."""
        memory = self.encoder(x_onehot)
        logits = self.decoder.forward_teacher_forced(target_indices, memory)
        return logits, memory

    def soft_decode(self, x_onehot: torch.Tensor, max_len: int | None = None,
                    temperature: float = 1.0):
        """Encode then soft-decode. Returns (soft_probs, memory)."""
        memory = self.encoder(x_onehot)
        L = max_len or self.max_seq_len
        return self.decoder.forward_soft_autoregressive(memory, L, temperature), memory

    def soft_decode_from_memory(self, memory: torch.Tensor, max_len: int | None = None,
                                temperature: float = 1.0):
        """Soft-decode from pre-computed (possibly perturbed) memory."""
        L = max_len or self.max_seq_len
        return self.decoder.forward_soft_autoregressive(memory, L, temperature)


# ─── Synthetic Data ──────────────────────────────────────────────────────────

def generate_synthetic_data(B: int, L: int, device: torch.device) -> torch.Tensor:
    """
    Every sequence is a mixture of high-entropy random DNA
    and 2-5 low-entropy blocks (homopolymers, dinuc repeats, pyrimidine tracts).
    """
    seq = torch.randint(0, 4, (B, L), device=device)
    for b in range(B):
        n_blocks = torch.randint(2, 6, (1,)).item()
        for _ in range(n_blocks):
            tract_len = min(torch.randint(15, 60, (1,)).item(), L)
            start = torch.randint(0, L - tract_len + 1, (1,)).item()
            kind = torch.randint(0, 3, (1,)).item()
            if kind == 0:
                # homopolymer
                seq[b, start:start + tract_len] = torch.randint(0, 4, (1,)).item()
            elif kind == 1:
                # dinucleotide repeat
                a, c = torch.randint(0, 4, (1,)).item(), torch.randint(0, 4, (1,)).item()
                motif = torch.tensor([a, c] * (tract_len // 2 + 1), device=device)[:tract_len]
                seq[b, start:start + tract_len] = motif
            else:
                # pyrimidine-rich
                pyr = torch.tensor([1, 3], device=device)
                seq[b, start:start + tract_len] = pyr[torch.randint(0, 2, (tract_len,), device=device)]
    return seq


# ─── Perturbation Diagnostics ───────────────────────────────────────────────

def decode_to_str(indices: torch.Tensor) -> str:
    return "".join(BASES[i] for i in indices.tolist())


def test_perturbation(model: DNASoftAutoencoder, args: argparse.Namespace,
                      device: torch.device) -> None:
    print("\n" + "=" * 60)
    print("Latent Perturbation Test  (soft autoregressive)")
    print("=" * 60)

    seq_indices = generate_synthetic_data(1, args.max_seq_len, device)
    x = F.one_hot(seq_indices, num_classes=4).float().transpose(1, 2)

    model.eval()
    with torch.no_grad():
        memory = model.encoder(x)                          # (1, M, D)
        M, D = memory.shape[1], memory.shape[2]

        # ── original soft decode ────────────────────────────────────────
        probs_orig = model.soft_decode_from_memory(memory, args.max_seq_len, temperature=args.temperature)
        orig_hard = probs_orig.argmax(dim=-1)[0]           # (L,)
        orig_seq = decode_to_str(orig_hard)

        # ── perturb one memory column ───────────────────────────────────
        col = M // 2
        perturbation = torch.randn(1, 1, D, device=device) * args.perturb_magnitude
        memory_pert = memory.clone()
        memory_pert[:, col, :] += perturbation.squeeze(1)

        probs_pert = model.soft_decode_from_memory(memory_pert, args.max_seq_len, temperature=args.temperature)
        pert_hard = probs_pert.argmax(dim=-1)[0]
        pert_seq = decode_to_str(pert_hard)

    # ── alignment analysis ──────────────────────────────────────────────
    input_seq = decode_to_str(seq_indices[0])
    print(f"Input sequence   (first 80): {input_seq[:80]}...")
    print(f"Orig  decoded    (first 80): {orig_seq[:80]}...")
    print(f"Pert  decoded    (first 80): {pert_seq[:80]}...")
    print(f"\nPerturbed memory column: {col} / {M}  (magnitude {args.perturb_magnitude:.1f})")

    # edit distance
    matcher = difflib.SequenceMatcher(None, orig_seq, pert_seq)
    ops = matcher.get_opcodes()
    n_equal = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag == "equal")
    n_changed = len(orig_seq) - n_equal

    print(f"Edit distance: {n_changed} / {len(orig_seq)} bases changed  ({100 * n_changed / len(orig_seq):.1f}%)")
    print("\nAlignment diffs:")
    for tag, i1, i2, j1, j2 in ops:
        if tag != "equal":
            print(f"  {tag:7s}  orig[{i1:3d}:{i2:3d}] → pert[{j1:3d}:{j2:3d}]")
            if (i2 - i1) + (j2 - j1) < 120:
                print(f"    orig: {orig_seq[i1:i2]}")
                print(f"    pert: {pert_seq[j1:j2]}")

    # entropy of soft probabilities
    entropy_orig = -(probs_orig.clamp_min(1e-8) * probs_orig.clamp_min(1e-8).log()).sum(-1).mean().item()
    entropy_pert = -(probs_pert.clamp_min(1e-8) * probs_pert.clamp_min(1e-8).log()).sum(-1).mean().item()
    print(f"\nMean soft-decode entropy: orig={entropy_orig:.3f}  pert={entropy_pert:.3f}  (max=ln4≈{math.log(4):.3f})")
    print("=" * 60 + "\n")


# ─── Training ───────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "mps" if args.mps and torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model = DNASoftAutoencoder(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        max_seq_len=args.max_seq_len,
        target_bases_per_latent=args.target_bases_per_latent,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    B, L = args.batch_size, args.max_seq_len

    # Fixed validation batch
    val_seq = generate_synthetic_data(B, L, device)
    val_x = F.one_hot(val_seq, num_classes=4).float().transpose(1, 2)

    print(f"\nTraining for {args.epochs} steps  (batch_size={B}, seq_len={L})\n")

    best_val_loss = float("inf")

    for step in range(1, args.epochs + 1):
        model.train()
        seq_indices = generate_synthetic_data(B, L, device)
        x = F.one_hot(seq_indices, num_classes=4).float().transpose(1, 2)

        logits, memory = model(x, seq_indices)
        loss = F.cross_entropy(logits.reshape(-1, 4), seq_indices.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ── diagnostics ─────────────────────────────────────────────────
        if step % args.print_every == 0 or step == args.epochs:
            model.eval()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == seq_indices).float().mean().item()
                exact = (preds == seq_indices).all(dim=-1).float().mean().item()

                # pred entropy (are outputs sharp?)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(-1).mean().item()

                # validation
                val_logits, _ = model(val_x, val_seq)
                val_loss = F.cross_entropy(val_logits.reshape(-1, 4), val_seq.reshape(-1)).item()
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == val_seq).float().mean().item()
                val_exact = (val_preds == val_seq).all(dim=-1).float().mean().item()

            print(
                f"step {step:05d}/{args.epochs} | "
                f"CE {loss.item():.4f}  val_CE {val_loss:.4f} | "
                f"acc {acc:.4f} (exact {exact:.3f})  val_acc {val_acc:.4f} (exact {val_exact:.3f}) | "
                f"entropy {entropy:.3f}"
            )

            # checkpoint
            if args.checkpoint_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                path = os.path.join(args.checkpoint_dir, "best.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "args": vars(args),
                }, path)
                print(f"  ↳ saved best checkpoint ({val_loss:.4f}): {path}")

    # ── end-of-training diagnostics ─────────────────────────────────────
    print("\n── Reconstruction sample (teacher-forced) ──")
    model.eval()
    with torch.no_grad():
        sample_seq = generate_synthetic_data(1, L, device)
        sample_x = F.one_hot(sample_seq, num_classes=4).float().transpose(1, 2)
        sample_logits, _ = model(sample_x, sample_seq)
        sample_pred = sample_logits.argmax(dim=-1)[0]

    inp = decode_to_str(sample_seq[0])
    out = decode_to_str(sample_pred)
    match = sum(a == b for a, b in zip(inp, out))
    print(f"  input:     {inp[:80]}...")
    print(f"  predicted: {out[:80]}...")
    print(f"  match: {match}/{L} ({100 * match / L:.1f}%)")

    # perturbation test
    test_perturbation(model, args, device)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DNA Transformer Soft Autoencoder")

    # architecture
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--max-seq-len", type=int, default=200)
    p.add_argument("--target-bases-per-latent", type=int, default=20)
    p.add_argument("--dropout", type=float, default=0.1)

    # training
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--print-every", type=int, default=500)

    # soft decode / perturbation
    p.add_argument("--temperature", type=float, default=0.5,
                   help="Softmax temperature for soft autoregressive decoding")
    p.add_argument("--perturb-magnitude", type=float, default=2.0,
                   help="L2 magnitude of random perturbation to one memory column")

    # infra
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/dna_transformer_soft_ae")
    p.add_argument("--mps", action="store_true", default=True)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
