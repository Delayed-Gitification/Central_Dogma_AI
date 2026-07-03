#!/usr/bin/env python3
"""
DNA Causal Conv Autoencoder (v4)
================================

Architecture:
  DNA → CNN Encoder → Spatial Latent Tokens [B, D, M]
                          ↓ nearest-neighbour upsample
                    FiLM conditioning [B, D, L]
                          ↓
  BOS + corrupted DNA → Causal Dilated ConvNet decoder → A/C/G/T logits
                        (trained to reconstruct clean DNA)

Anti-latent-collapse:
  Input token corruption — random fraction of teacher-forced tokens replaced
  with [MASK], forcing the decoder to read z for those positions.

v4 improvements over v3:
  - Cosine LR schedule with linear warmup
  - Corruption rate annealing (ramps up over first N steps)
  - Print per-position entropy profile (low-complexity vs high-complexity)
  - Print encoder/decoder param counts separately
  - Smarter defaults (corruption=0.20, print_every=200)
"""

import argparse
import contextlib
import difflib
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

BASES = "ACGT"

# Vocabulary
BASE_A, BASE_C, BASE_G, BASE_T = 0, 1, 2, 3
BOS_ID = 4
MASK_ID = 5
VOCAB_SIZE = 6   # A, C, G, T, BOS, MASK


# ─── Causal Conv Building Blocks ────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Left-padded causal 1D convolution: output[t] depends only on input[0..t]."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation: h → (1 + γ(z)) · h + β(z).

    Zero-initialised so the model starts unconditionally (identity transform).
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond.transpose(1, 2)).transpose(1, 2)  # (B, 2D, L)
        gamma, beta = params.chunk(2, dim=1)
        return (1.0 + gamma) * h + beta


class CausalResBlock(nn.Module):
    """GroupNorm → CausalConv → GELU → FiLM → Pointwise → residual."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.causal_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.act = nn.GELU()
        self.film = FiLMConditioning(cond_dim, channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.causal_conv(h)
        h = self.act(h)
        h = self.film(h, cond)
        h = self.pointwise(h)
        return x + h


# ─── Encoder ────────────────────────────────────────────────────────────────

class DNAEncoder(nn.Module):
    """CNN encoder: (B, 4, L) one-hot DNA → (B, D, M) spatial latent tokens."""

    def __init__(self, d_model: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv1d(4, d_model, kernel_size=stride * 2, stride=stride,
                      padding=stride // 2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B, D, M) channels-first


# ─── Decoder ────────────────────────────────────────────────────────────────

class CausalConvDecoder(nn.Module):
    """
    Causal dilated ConvNet decoder conditioned on upsampled latent tokens via FiLM.

    Input vocab: A(0), C(1), G(2), T(3), BOS(4), MASK(5).
    Output: 4-class DNA logits (fixed max_len, no EOS).
    """

    def __init__(self, d_model: int, cond_dim: int, kernel_size: int,
                 dilations: list[int]):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)

        self.blocks = nn.ModuleList([
            CausalResBlock(d_model, kernel_size, d, cond_dim) for d in dilations
        ])
        self.norm_final = nn.GroupNorm(1, d_model)
        self.out_proj = nn.Conv1d(d_model, 4, 1)

    def _run_blocks(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.norm_final(h)).transpose(1, 2)  # (B, L, 4)

    # ── teacher forcing ──────────────────────────────────────────────────

    def forward_teacher_forced(self, target: torch.Tensor,
                               conditioning: torch.Tensor,
                               corruption_rate: float = 0.0) -> torch.Tensor:
        """
        target:          (B, L) int64 clean DNA indices 0-3
        conditioning:    (B, D, L) upsampled latent
        corruption_rate: fraction of non-BOS input tokens replaced with MASK
        returns:         (B, L, 4) logits predicting clean DNA at every position
        """
        B, L = target.shape

        bos = torch.full((B, 1), BOS_ID, dtype=torch.long, device=target.device)
        input_tokens = torch.cat([bos, target[:, :-1]], dim=1)   # (B, L)

        if corruption_rate > 0 and self.training:
            mask = (torch.rand(B, L, device=target.device) < corruption_rate)
            mask[:, 0] = False   # never corrupt BOS
            input_tokens = input_tokens.masked_fill(mask, MASK_ID)

        h = self.token_emb(input_tokens).transpose(1, 2)  # (B, D, L)
        return self._run_blocks(h, conditioning)

    # ── soft autoregressive (differentiable) ─────────────────────────────

    def forward_soft_autoregressive(self, conditioning: torch.Tensor,
                                    max_len: int,
                                    temperature: float = 1.0,
                                    no_grad: bool = False) -> torch.Tensor:
        """
        Soft autoregressive decoding — differentiable by default.
        Set no_grad=True for diagnostics to save memory.
        """
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            B, device = conditioning.shape[0], conditioning.device
            dna_emb = self.token_emb.weight[:4]  # (4, D)

            buf = torch.zeros(B, self.d_model, max_len, device=device)
            buf[:, :, 0] = self.token_emb(
                torch.full((B,), BOS_ID, dtype=torch.long, device=device)
            )

            all_probs = []
            for t in range(max_len):
                h = buf
                for block in self.blocks:
                    h = block(h, conditioning)
                logits = self.out_proj(self.norm_final(h))  # (B, 4, max_len)

                probs_t = F.softmax(logits[:, :, t] / temperature, dim=-1)
                all_probs.append(probs_t)

                if t < max_len - 1:
                    buf = buf.clone()
                    buf[:, :, t + 1] = probs_t @ dna_emb

            return torch.stack(all_probs, dim=1)  # (B, max_len, 4)


# ─── Full Model ──────────────────────────────────────────────────────────────

class DNACausalConvAutoencoder(nn.Module):
    def __init__(self, d_model: int = 48, kernel_size: int = 3,
                 dilation_cycles: int = 1, max_seq_len: int = 200,
                 target_bases_per_latent: int = 20):
        super().__init__()
        dilations = [1, 2, 4, 8] * dilation_cycles
        self.dilations = dilations
        self.target_bases_per_latent = target_bases_per_latent
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.receptive_field = 1 + sum((kernel_size - 1) * d for d in dilations)

        self.encoder = DNAEncoder(d_model, stride=target_bases_per_latent)
        self.decoder = CausalConvDecoder(
            d_model=d_model, cond_dim=d_model,
            kernel_size=kernel_size, dilations=dilations,
        )

    def _upsample(self, z: torch.Tensor, length: int) -> torch.Tensor:
        return F.interpolate(z, size=length, mode="nearest")

    def encode(self, x_onehot: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_onehot)

    def forward(self, x_onehot: torch.Tensor, target: torch.Tensor,
                corruption_rate: float = 0.0):
        """Teacher-forced pass. Returns (logits, z)."""
        L = x_onehot.shape[2]
        z = self.encoder(x_onehot)
        cond = self._upsample(z, L)
        logits = self.decoder.forward_teacher_forced(target, cond, corruption_rate)
        return logits, z

    def soft_decode_from_latent(self, z: torch.Tensor,
                                max_len: int | None = None,
                                temperature: float = 1.0,
                                no_grad: bool = False) -> torch.Tensor:
        L = max_len or self.max_seq_len
        cond = self._upsample(z, L)
        return self.decoder.forward_soft_autoregressive(cond, L, temperature, no_grad=no_grad)


# ─── Synthetic Data ──────────────────────────────────────────────────────────

def generate_synthetic_data(B: int, L: int, device: torch.device) -> torch.Tensor:
    """
    Uniform-random background + 2-5 low-entropy blocks per sequence.
    Block types: homopolymer, dinucleotide repeat, pyrimidine-rich tract.
    """
    seq = torch.randint(0, 4, (B, L), device=device)
    for b in range(B):
        n_blocks = torch.randint(2, 6, (1,)).item()
        for _ in range(n_blocks):
            tl = min(torch.randint(15, 60, (1,)).item(), L)
            st = torch.randint(0, L - tl + 1, (1,)).item()
            kind = torch.randint(0, 3, (1,)).item()
            if kind == 0:
                seq[b, st:st + tl] = torch.randint(0, 4, (1,)).item()
            elif kind == 1:
                a, c = torch.randint(0, 4, (1,)).item(), torch.randint(0, 4, (1,)).item()
                seq[b, st:st + tl] = torch.tensor(
                    [a, c] * (tl // 2 + 1), device=device
                )[:tl]
            else:
                pyr = torch.tensor([BASE_C, BASE_T], device=device)
                seq[b, st:st + tl] = pyr[torch.randint(0, 2, (tl,), device=device)]
    return seq


# ─── LR Schedule ────────────────────────────────────────────────────────────

def get_lr(step: int, total_steps: int, base_lr: float, min_lr: float,
           warmup_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / float(decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# ─── Diagnostics ────────────────────────────────────────────────────────────

def decode_str(idx: torch.Tensor) -> str:
    return "".join(BASES[i] for i in idx.tolist())


@torch.no_grad()
def latent_ablation_ce(model: DNACausalConvAutoencoder,
                       seq: torch.Tensor) -> float:
    """CE when conditioning is zeroed. Large Δ = latent is being used."""
    L, B = seq.shape[1], seq.shape[0]
    zero_cond = torch.zeros(B, model.d_model, L, device=seq.device)
    logits = model.decoder.forward_teacher_forced(seq, zero_cond, corruption_rate=0.0)
    return F.cross_entropy(logits.reshape(-1, 4), seq.reshape(-1)).item()


@torch.no_grad()
def free_running_metrics(model: DNACausalConvAutoencoder,
                         seq: torch.Tensor,
                         temperature: float) -> tuple[float, float]:
    """Free-running soft AR accuracy — the metric that actually matters."""
    x = F.one_hot(seq, num_classes=4).float().transpose(1, 2)
    z = model.encode(x)
    probs = model.soft_decode_from_latent(z, seq.shape[1], temperature, no_grad=True)
    preds = probs.argmax(-1)
    acc = (preds == seq).float().mean().item()
    exact = (preds == seq).all(-1).float().mean().item()
    return acc, exact


def test_perturbation(model: DNACausalConvAutoencoder,
                      args: argparse.Namespace,
                      device: torch.device) -> None:
    hbar = "=" * 60
    print(f"\n{hbar}")
    print("Latent Perturbation Test  (soft autoregressive)")
    print(hbar)

    seq = generate_synthetic_data(1, args.max_seq_len, device)
    x = F.one_hot(seq, num_classes=4).float().transpose(1, 2)

    model.eval()
    with torch.no_grad():
        z = model.encode(x)
        M = z.shape[2]
        col = M // 2
        expected_start = col * model.target_bases_per_latent
        expected_end = (col + 1) * model.target_bases_per_latent

        probs_orig = model.soft_decode_from_latent(z, args.max_seq_len,
                                                    args.temperature, no_grad=True)
        orig_seq = decode_str(probs_orig.argmax(-1)[0])

        z_pert = z.clone()
        z_pert[:, :, col] += torch.randn_like(z_pert[:, :, col]) * args.perturb_magnitude

        probs_pert = model.soft_decode_from_latent(z_pert, args.max_seq_len,
                                                    args.temperature, no_grad=True)
        pert_seq = decode_str(probs_pert.argmax(-1)[0])

    input_seq = decode_str(seq[0])
    print(f"Input  (first 80): {input_seq[:80]}…")
    print(f"Orig   (first 80): {orig_seq[:80]}…")
    print(f"Pert   (first 80): {pert_seq[:80]}…")
    print(f"\nPerturbed column {col}/{M} → expected window "
          f"[{expected_start}, {expected_end})  magnitude={args.perturb_magnitude:.1f}")

    matcher = difflib.SequenceMatcher(None, orig_seq, pert_seq)
    ops = matcher.get_opcodes()
    change_positions = []
    n_equal = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag == "equal")
    n_changed = len(orig_seq) - n_equal

    print(f"\nEdit distance: {n_changed}/{len(orig_seq)} bases changed "
          f"({100 * n_changed / max(1, len(orig_seq)):.1f}%)")

    print("\nAlignment diffs:")
    for tag, i1, i2, j1, j2 in ops:
        if tag != "equal":
            change_positions.extend(range(i1, i2))
            print(f"  {tag:7s}  orig[{i1:3d}:{i2:3d}] → pert[{j1:3d}:{j2:3d}]")
            if (i2 - i1) + (j2 - j1) < 120:
                print(f"    orig: {orig_seq[i1:i2]}")
                print(f"    pert: {pert_seq[j1:j2]}")

    if change_positions:
        mean_pos = sum(change_positions) / len(change_positions)
        upstream = sum(p < expected_start for p in change_positions)
        in_win = sum(expected_start <= p < expected_end for p in change_positions)
        downstream = sum(p >= expected_end for p in change_positions)
        N = len(change_positions)
        print(f"\nMean change position : {mean_pos:.0f}  "
              f"(expected centre: {(expected_start + expected_end) / 2:.0f})")
        print(f"Locality:  upstream={upstream/N:.2f}  "
              f"in-window={in_win/N:.2f}  downstream={downstream/N:.2f}")

    ent_orig = -(probs_orig.clamp_min(1e-8) * probs_orig.clamp_min(1e-8).log()
                 ).sum(-1).mean().item()
    ent_pert = -(probs_pert.clamp_min(1e-8) * probs_pert.clamp_min(1e-8).log()
                 ).sum(-1).mean().item()
    print(f"\nSoft-decode entropy: orig={ent_orig:.3f}  pert={ent_pert:.3f}  "
          f"(max=ln4≈{math.log(4):.3f})")
    print(hbar + "\n")


# ─── Training ───────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "mps" if args.mps and torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model = DNACausalConvAutoencoder(
        d_model=args.d_model,
        kernel_size=args.kernel_size,
        dilation_cycles=args.dilation_cycles,
        max_seq_len=args.max_seq_len,
        target_bases_per_latent=args.target_bases_per_latent,
    ).to(device)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    n_params = enc_params + dec_params
    n_latent = args.max_seq_len // args.target_bases_per_latent
    bottleneck_bits = n_latent * args.d_model  # total floats through bottleneck

    print(f"Parameters       : {n_params:,}  (encoder {enc_params:,} / decoder {dec_params:,})")
    print(f"Receptive field  : {model.receptive_field} bases  "
          f"(kernel={args.kernel_size}, dilations={model.dilations})")
    print(f"Latent bottleneck: {n_latent} tokens × {args.d_model} dims = "
          f"{bottleneck_bits} floats")
    print(f"Sequence info    : {args.max_seq_len} bases × 2 bits ≈ "
          f"{args.max_seq_len * 2} bits to encode")
    print(f"Token corruption : {args.corruption_rate:.0%}")
    print(f"LR schedule      : warmup {args.warmup_steps} steps → "
          f"cosine decay {args.lr:.1e} → {args.min_lr:.1e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    B, L = args.batch_size, args.max_seq_len

    # Fixed validation batch
    val_seq = generate_synthetic_data(B, L, device)
    val_x = F.one_hot(val_seq, num_classes=4).float().transpose(1, 2)

    print(f"\nTraining for {args.epochs} steps  (batch={B}, seq_len={L})\n")
    best_val = float("inf")

    for step in range(1, args.epochs + 1):
        # LR schedule
        lr = get_lr(step - 1, args.epochs, args.lr, args.min_lr, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Corruption annealing: ramp from 0 to target over warmup
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            corruption = args.corruption_rate * float(step) / float(args.warmup_steps)
        else:
            corruption = args.corruption_rate

        model.train()
        seq = generate_synthetic_data(B, L, device)
        x = F.one_hot(seq, num_classes=4).float().transpose(1, 2)

        logits, _ = model(x, seq, corruption_rate=corruption)
        loss = F.cross_entropy(logits.reshape(-1, 4), seq.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.epochs:
            model.eval()
            with torch.no_grad():
                tf_acc = (logits.argmax(-1) == seq).float().mean().item()
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()
                            ).sum(-1).mean().item()

                val_logits, _ = model(val_x, val_seq, corruption_rate=0.0)
                val_ce = F.cross_entropy(
                    val_logits.reshape(-1, 4), val_seq.reshape(-1)
                ).item()
                val_tf_acc = (val_logits.argmax(-1) == val_seq).float().mean().item()

                abl_ce = latent_ablation_ce(model, val_seq)

            fr_acc, fr_exact = free_running_metrics(model, val_seq, args.temperature)

            latent_delta = abl_ce - val_ce
            collapse_flag = "  ⚠ COLLAPSE" if latent_delta < 0.05 else ""

            print(
                f"step {step:05d}/{args.epochs} │ "
                f"lr {lr:.1e}  CE {loss.item():.4f}  val {val_ce:.4f}  "
                f"abl {abl_ce:.4f} (Δ{latent_delta:+.3f}){collapse_flag} │ "
                f"tf {tf_acc:.3f}/{val_tf_acc:.3f}  "
                f"free {fr_acc:.3f} (ex {fr_exact:.3f}) │ "
                f"H {entropy:.3f}  corr {corruption:.0%}"
            )

            if args.checkpoint_dir and val_ce < best_val:
                best_val = val_ce
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                path = os.path.join(args.checkpoint_dir, "best.pt")
                torch.save({
                    "step": step, "model_state_dict": model.state_dict(),
                    "val_ce": val_ce, "args": vars(args),
                }, path)
                print(f"  ↳ saved best ({val_ce:.4f}): {path}")

    # ── final reconstruction ────────────────────────────────────────────────
    print("\n── Reconstruction (teacher-forced vs free-running) ──")
    model.eval()
    s = generate_synthetic_data(1, L, device)
    sx = F.one_hot(s, num_classes=4).float().transpose(1, 2)

    with torch.no_grad():
        sl, _ = model(sx, s, corruption_rate=0.0)
        sp_tf = sl.argmax(-1)[0]

        z = model.encode(sx)
        probs_free = model.soft_decode_from_latent(z, L, args.temperature, no_grad=True)
        sp_fr = probs_free.argmax(-1)[0]

    inp = decode_str(s[0])
    out_tf = decode_str(sp_tf)
    out_fr = decode_str(sp_fr)
    m_tf = sum(a == b for a, b in zip(inp, out_tf))
    m_fr = sum(a == b for a, b in zip(inp, out_fr))

    print(f"  input:          {inp[:80]}…")
    print(f"  teacher-forced: {out_tf[:80]}…  match {m_tf}/{L} ({100*m_tf/L:.1f}%)")
    print(f"  free-running:   {out_fr[:80]}…  match {m_fr}/{L} ({100*m_fr/L:.1f}%)")

    test_perturbation(model, args, device)


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DNA Causal Conv Autoencoder v4"
    )
    # architecture
    p.add_argument("--d-model", type=int, default=48)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dilation-cycles", type=int, default=1,
                   help="Dilation cycles [1,2,4,8]. 1→RF≈31bp, 2→RF≈61bp.")
    p.add_argument("--max-seq-len", type=int, default=200)
    p.add_argument("--target-bases-per-latent", type=int, default=20)

    # training
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--corruption-rate", type=float, default=0.20,
                   help="Fraction of input tokens replaced with [MASK]. Default 0.20.")

    # soft decode / perturbation
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--perturb-magnitude", type=float, default=2.0)

    # infra
    p.add_argument("--checkpoint-dir", default="checkpoints/dna_causal_conv_ae")
    p.add_argument("--mps", action="store_true", default=True)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
