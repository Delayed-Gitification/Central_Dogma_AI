from __future__ import annotations

import argparse
import contextlib
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DNA_BASES = "ACGT"
BASE_TO_INDEX = {base: index for index, base in enumerate(DNA_BASES)}
BOS_ID = 4
MASK_ID = 5
VOCAB_SIZE = 6


def clean_dna(sequence: str) -> str:
    cleaned = "".join(base for base in sequence.upper() if base in BASE_TO_INDEX)
    if not cleaned:
        raise ValueError("Sequence is empty after filtering to A/C/G/T.")
    return cleaned


def sequence_to_tensor(sequence: str) -> torch.Tensor:
    return torch.tensor([BASE_TO_INDEX[base] for base in clean_dna(sequence)], dtype=torch.long)


def tensor_to_sequence(tensor: torch.Tensor) -> str:
    return "".join(DNA_BASES[int(index)] for index in tensor.detach().cpu().tolist())


def random_dna(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def homopolymer_block(rng: random.Random, length: int) -> str:
    return rng.choice(DNA_BASES) * length


def dinucleotide_block(rng: random.Random, length: int) -> str:
    motif = rng.choice(DNA_BASES) + rng.choice(DNA_BASES)
    return (motif * ((length + 1) // 2))[:length]


def motif_block(rng: random.Random, length: int) -> str:
    motif_length = rng.randint(3, 8)
    motif = random_dna(rng, motif_length)
    return (motif * ((length + motif_length - 1) // motif_length))[:length]


def mixed_dna(rng: random.Random, length: int) -> str:
    builders = [
        ("random", random_dna, 4, 16),
        ("homopolymer", homopolymer_block, 12, 64),
        ("dinucleotide", dinucleotide_block, 12, 64),
        ("motif", motif_block, 12, 64),
    ]
    parts = []
    total = 0
    while total < length:
        _name, builder, min_len, max_len = rng.choice(builders)
        block_len = min(length - total, rng.randint(min_len, max_len))
        parts.append(builder(rng, block_len))
        total += block_len
    return "".join(parts)


def make_batch(
    *,
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    sequences = [mixed_dna(rng, seq_len) for _ in range(batch_size)]
    target = torch.stack([sequence_to_tensor(sequence) for sequence in sequences], dim=0).to(device)
    return sequences, target


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false.")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false.")
    return torch.device(requested)


class TinyFrozenDNAEncoder(nn.Module):
    """Small local fallback encoder, useful for smoke tests without HF downloads."""

    def __init__(self, hidden_dim: int, layers: int):
        super().__init__()
        self.embedding = nn.Embedding(4, hidden_dim)
        blocks = []
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.GELU(),
                ]
            )
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        x = self.embedding(target).transpose(1, 2)
        x = self.blocks(x).transpose(1, 2)
        return self.norm(x)


class HFDNAEncoder(nn.Module):
    def __init__(self, model_name: str, device: torch.device, trust_remote_code: bool):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers to use --encoder hf.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            raise RuntimeError("Could not infer hidden size from Hugging Face model config.")
        self.hidden_dim = int(hidden_size)

    @torch.no_grad()
    def forward_sequences(self, sequences: list[str], *, seq_len: int, device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.model(**tokens)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        return hidden


class LatentBottleneck(nn.Module):
    def __init__(self, encoder_dim: int, latent_dim: int, latent_tokens: int, embedding_mode: str):
        super().__init__()
        self.latent_tokens = latent_tokens
        self.embedding_mode = embedding_mode
        if embedding_mode == "per_token":
            self.proj = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        elif embedding_mode == "mean":
            self.proj = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, latent_dim * latent_tokens),
                nn.GELU(),
                nn.Linear(latent_dim * latent_tokens, latent_dim * latent_tokens),
            )
        else:
            raise ValueError(f"Unknown embedding_mode: {embedding_mode}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.embedding_mode == "per_token":
            pooled = F.adaptive_avg_pool1d(hidden.transpose(1, 2), self.latent_tokens).transpose(1, 2)
            return self.proj(pooled)
        pooled = hidden.mean(dim=1)
        return self.proj(pooled).reshape(hidden.shape[0], self.latent_tokens, -1)


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond.transpose(1, 2)).transpose(1, 2)
        gamma, beta = params.chunk(2, dim=1)
        return (1.0 + gamma) * x + beta


class CausalDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.conv = CausalConv1d(hidden_dim, kernel_size=kernel_size, dilation=dilation)
        self.film = FiLM(cond_dim, hidden_dim)
        self.out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.conv(h))
        h = self.film(h, cond)
        return x + self.out(h)


class LatentConditionedDNADecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dilation_cycles: int,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, hidden_dim)
        dilations = [1, 2, 4, 8] * dilation_cycles
        self.blocks = nn.ModuleList(
            [CausalDecoderBlock(hidden_dim, latent_dim, kernel_size, dilation) for dilation in dilations]
        )
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.out = nn.Conv1d(hidden_dim, 4, kernel_size=1)

    def latent_to_conditioning(self, latents: torch.Tensor, seq_len: int) -> torch.Tensor:
        return F.interpolate(latents.transpose(1, 2), size=seq_len, mode="linear", align_corners=False)

    def forward_teacher_forced(
        self,
        target: torch.Tensor,
        latents: torch.Tensor,
        *,
        corruption_rate: float,
    ) -> torch.Tensor:
        batch_size, seq_len = target.shape
        bos = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=target.device)
        input_tokens = torch.cat([bos, target[:, :-1]], dim=1)
        if corruption_rate > 0 and self.training:
            mask = torch.rand(batch_size, seq_len, device=target.device) < corruption_rate
            mask[:, 0] = False
            input_tokens = input_tokens.masked_fill(mask, MASK_ID)
        x = self.token_emb(input_tokens).transpose(1, 2)
        cond = self.latent_to_conditioning(latents, seq_len)
        for block in self.blocks:
            x = block(x, cond)
        return self.out(self.norm(x)).transpose(1, 2)

    def free_run(self, latents: torch.Tensor, seq_len: int, temperature: float) -> torch.Tensor:
        batch_size = latents.shape[0]
        cond = self.latent_to_conditioning(latents, seq_len)
        dna_embedding = self.token_emb.weight[:4]
        buffer = torch.zeros(batch_size, self.token_emb.embedding_dim, seq_len, device=latents.device)
        buffer[:, :, 0] = self.token_emb(torch.full((batch_size,), BOS_ID, dtype=torch.long, device=latents.device))
        probs = []
        for position in range(seq_len):
            x = buffer
            for block in self.blocks:
                x = block(x, cond)
            logits = self.out(self.norm(x))[:, :, position]
            prob = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            probs.append(prob)
            if position < seq_len - 1:
                buffer = buffer.clone()
                buffer[:, :, position + 1] = prob @ dna_embedding
        return torch.stack(probs, dim=1)


class FoundationLatentDecoder(nn.Module):
    def __init__(self, bottleneck: LatentBottleneck, decoder: LatentConditionedDNADecoder):
        super().__init__()
        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, hidden: torch.Tensor, target: torch.Tensor, corruption_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.bottleneck(hidden)
        logits = self.decoder.forward_teacher_forced(target, latents, corruption_rate=corruption_rate)
        return logits, latents


def reconstruction_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    predicted = logits.argmax(dim=-1)
    correct = predicted == target
    lev_scores = []
    for row in range(target.shape[0]):
        truth = tensor_to_sequence(target[row])
        pred = tensor_to_sequence(predicted[row])
        lev_scores.append(levenshtein_similarity(truth, pred))
    return {
        "acc": float(correct.float().mean().item()),
        "exact": float(correct.all(dim=1).float().mean().item()),
        "lev": float(sum(lev_scores) / max(1, len(lev_scores))),
    }


def levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    distance = previous[-1]
    return 1.0 - float(distance) / float(max(len(a), len(b), 1))


def current_corruption(step: int, steps: int, max_corruption: float, warmup_frac: float) -> float:
    if max_corruption <= 0:
        return 0.0
    warmup_steps = max(1, int(steps * warmup_frac))
    return max_corruption * min(1.0, float(step + 1) / float(warmup_steps))


def load_encoder(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module | HFDNAEncoder, int]:
    if args.encoder == "hf":
        encoder = HFDNAEncoder(args.hf_model, device=device, trust_remote_code=args.trust_remote_code)
        return encoder, encoder.hidden_dim
    encoder = TinyFrozenDNAEncoder(args.encoder_hidden_dim, args.encoder_layers).to(device)
    if args.freeze_encoder:
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    return encoder, args.encoder_hidden_dim


def encode_hidden(
    encoder: nn.Module | HFDNAEncoder,
    *,
    encoder_kind: str,
    sequences: list[str],
    target: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if encoder_kind == "hf":
        assert isinstance(encoder, HFDNAEncoder)
        return encoder.forward_sequences(sequences, seq_len=seq_len, device=device)
    return encoder(target)


def train(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    encoder, encoder_dim = load_encoder(args, device)
    bottleneck = LatentBottleneck(encoder_dim, args.latent_dim, args.latent_tokens, args.embedding_mode)
    decoder = LatentConditionedDNADecoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.decoder_hidden_dim,
        kernel_size=args.kernel_size,
        dilation_cycles=args.dilation_cycles,
    )
    model = FoundationLatentDecoder(bottleneck, decoder).to(device)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.encoder == "tiny" and not args.freeze_encoder:
        trainable_params += [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print("DNA foundation-latent decoder")
    print(f"device: {device}")
    print(f"encoder={args.encoder}; hf_model={args.hf_model if args.encoder == 'hf' else 'n/a'}; encoder_dim={encoder_dim}")
    print(
        f"seq_len={args.seq_len}; latent_tokens={args.latent_tokens}; latent_dim={args.latent_dim}; "
        f"embedding_mode={args.embedding_mode}; decoder_params={sum(parameter.numel() for parameter in model.parameters())}"
    )

    for step in range(args.steps):
        sequences, target = make_batch(batch_size=args.batch_size, seq_len=args.seq_len, rng=rng, device=device)
        with torch.no_grad() if args.freeze_encoder or args.encoder == "hf" else contextlib.nullcontext():
            hidden = encode_hidden(
                encoder,
                encoder_kind=args.encoder,
                sequences=sequences,
                target=target,
                seq_len=args.seq_len,
                device=device,
            )
        corruption_rate = current_corruption(step, args.steps, args.corruption_rate, args.corruption_warmup_frac)
        logits, latents = model(hidden, target, corruption_rate)
        ce_loss = F.cross_entropy(logits.reshape(-1, 4), target.reshape(-1))
        latent_l2 = latents.pow(2).mean()
        loss = ce_loss + args.latent_l2_weight * latent_l2

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            model.eval()
            if isinstance(encoder, nn.Module):
                encoder.eval()
            with torch.no_grad():
                val_sequences, val_target = make_batch(
                    batch_size=args.val_batch_size,
                    seq_len=args.seq_len,
                    rng=rng,
                    device=device,
                )
                val_hidden = encode_hidden(
                    encoder,
                    encoder_kind=args.encoder,
                    sequences=val_sequences,
                    target=val_target,
                    seq_len=args.seq_len,
                    device=device,
                )
                val_logits, val_latents = model(val_hidden, val_target, corruption_rate=0.0)
                val_loss = F.cross_entropy(val_logits.reshape(-1, 4), val_target.reshape(-1))
                train_metrics = reconstruction_metrics(logits, target)
                val_metrics = reconstruction_metrics(val_logits, val_target)
                free_probs = model.decoder.free_run(val_latents[:1], args.seq_len, temperature=args.free_run_temperature)
                free_sequence = tensor_to_sequence(free_probs.argmax(dim=-1)[0])
                teacher_sequence = tensor_to_sequence(val_logits.argmax(dim=-1)[0])
            if val_loss.item() < best_val_loss:
                best_val_loss = float(val_loss.item())
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "step": step,
                        "validation_loss": best_val_loss,
                    },
                    checkpoint_dir / "best.pt",
                )
            print(
                f"\nstep {step:06d} loss {loss.item():.4f} ce {ce_loss.item():.4f} "
                f"val {val_loss.item():.4f} best {best_val_loss:.4f} corruption {corruption_rate:.3f}"
            )
            print(
                f"train acc {train_metrics['acc']:.3f} exact {train_metrics['exact']:.3f} | "
                f"val acc {val_metrics['acc']:.3f} exact {val_metrics['exact']:.3f} lev {val_metrics['lev']:.3f} | "
                f"latent_norm {val_latents.norm(dim=-1).mean().item():.3f}"
            )
            print("target:  ", val_sequences[0])
            print("teacher: ", teacher_sequence)
            print("free:    ", free_sequence)
            model.train()
            if args.encoder == "tiny" and not args.freeze_encoder:
                encoder.train()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": args.steps - 1,
            "validation_loss": best_val_loss,
        },
        checkpoint_dir / "latest.pt",
    )
    print(f"\nsaved latest checkpoint: {checkpoint_dir / 'latest.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DNA decoder from frozen foundation-model latent states.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--encoder", choices=("tiny", "hf"), default="tiny")
    parser.add_argument("--hf-model", default="kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--embedding-mode", choices=("per_token", "mean"), default="per_token")
    parser.add_argument("--encoder-hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--latent-tokens", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilation-cycles", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--corruption-rate", type=float, default=0.25)
    parser.add_argument("--corruption-warmup-frac", type=float, default=0.1)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-5)
    parser.add_argument("--free-run-temperature", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="checkpoints/dna_foundation_latent_decoder")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
