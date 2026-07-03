from __future__ import annotations

import argparse
import difflib
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:
    rapidfuzz_fuzz = None


DNA_BASES = "ACGT"
BASE_TO_INDEX = {base: index for index, base in enumerate(DNA_BASES)}


def sequence_to_tensor(sequence: str) -> torch.Tensor:
    return torch.tensor([BASE_TO_INDEX[base] for base in sequence], dtype=torch.long)


def tensor_to_sequence(tensor: torch.Tensor) -> str:
    return "".join(DNA_BASES[index] for index in tensor.detach().cpu().tolist())


def random_dna_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def gc_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("GGGCCAT") for _ in range(length))


def at_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("AAATTGC") for _ in range(length))


def pyrimidine_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("CTTCC") for _ in range(length))


def purine_rich_block(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("AAGGA") for _ in range(length))


def homopolymer_block(rng: random.Random, length: int) -> str:
    return rng.choice(DNA_BASES) * length


def dinucleotide_repeat_block(rng: random.Random, length: int) -> str:
    motif = rng.choice(DNA_BASES) + rng.choice(DNA_BASES)
    return (motif * ((length + 1) // 2))[:length]


def motif_repeat_block(rng: random.Random, length: int) -> str:
    motif_length = rng.randint(3, 6)
    motif = random_dna_block(rng, motif_length)
    return (motif * ((length + motif_length - 1) // motif_length))[:length]


BLOCK_BUILDERS = {
    "random": random_dna_block,
    "gc": gc_rich_block,
    "at": at_rich_block,
    "pyrimidine": pyrimidine_rich_block,
    "purine": purine_rich_block,
    "homopolymer": homopolymer_block,
    "dinucleotide": dinucleotide_repeat_block,
    "motif": motif_repeat_block,
}


TRAIN_FAMILIES = (
    "random",
    "gc",
    "at",
    "pyrimidine",
    "purine",
    "homopolymer",
    "dinucleotide",
    "motif",
    "hybrid",
)


def block_length_for_family(family: str, rng: random.Random, remaining: int) -> int:
    if family == "random":
        length = rng.randint(3, 12)
    elif family in {"gc", "at", "pyrimidine", "purine"}:
        length = rng.randint(6, 28)
    elif family == "homopolymer":
        length = rng.randint(8, 72)
    elif family == "dinucleotide":
        length = rng.randint(6, 72)
    elif family == "motif":
        length = rng.randint(8, 72)
    else:
        length = rng.randint(3, 72)
    return min(remaining, length)


def generate_controlled_sequence(
    *,
    seq_len: int,
    block_type: str,
    block_len: int,
    rng: random.Random,
) -> str:
    builder = BLOCK_BUILDERS[block_type]
    parts = []
    total = 0
    while total < seq_len:
        length = min(block_len, seq_len - total)
        parts.append(builder(rng, length))
        total += length
    return "".join(parts)


def generate_family_sequence(seq_len: int, family: str, rng: random.Random) -> tuple[str, str]:
    parts = []
    total = 0
    while total < seq_len:
        remaining = seq_len - total
        block_type = rng.choice(tuple(BLOCK_BUILDERS)) if family == "hybrid" else family
        length = block_length_for_family(block_type, rng, remaining)
        parts.append(BLOCK_BUILDERS[block_type](rng, length))
        total += length
    return "".join(parts), family


def generate_variable_sequence(
    *,
    max_seq_len: int,
    min_seq_len: int,
    short_max_len_frac: float,
    long_min_len_frac: float,
    rng: random.Random,
) -> tuple[str, str]:
    short_max_len = max(min_seq_len, int(round(max_seq_len * short_max_len_frac)))
    long_min_len = min(max_seq_len, max(min_seq_len, int(round(max_seq_len * long_min_len_frac))))
    length_mode = rng.random()
    if length_mode < 0.4:
        seq_len = rng.randint(min_seq_len, short_max_len)
    elif length_mode < 0.8:
        seq_len = rng.randint(long_min_len, max_seq_len)
    else:
        seq_len = rng.randint(min_seq_len, max_seq_len)
    return generate_family_sequence(seq_len, rng.choice(TRAIN_FAMILIES), rng)


def shannon_entropy_per_base(sequence: str) -> float:
    if not sequence:
        return 0.0
    entropy = 0.0
    for base in DNA_BASES:
        count = sequence.count(base)
        if count > 0:
            probability = count / len(sequence)
            entropy -= probability * math.log2(probability)
    return entropy


def kmer_entropy(sequence: str, k: int) -> float:
    if len(sequence) < k or k <= 1:
        return shannon_entropy_per_base(sequence)
    counts: dict[str, int] = {}
    for index in range(len(sequence) - k + 1):
        kmer = sequence[index : index + k]
        counts[kmer] = counts.get(kmer, 0) + 1
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def kmer_complexity_per_base(sequence: str, k: int) -> float:
    if not sequence:
        return 0.0
    if len(sequence) < k or k <= 1:
        return shannon_entropy_per_base(sequence)
    window_count = len(sequence) - k + 1
    max_observable_entropy = min(2.0 * float(k), math.log2(float(window_count)))
    if max_observable_entropy <= 0:
        return shannon_entropy_per_base(sequence)
    entropy = kmer_entropy(sequence, k)
    # Rescale by the empirical maximum for this sequence length. Without this,
    # large k makes random DNA look artificially low-entropy because every k-mer
    # is unique and H ~= log2(number_of_windows), not 2k.
    return max(0.0, min(2.0, 2.0 * entropy / max_observable_entropy))


def multiscale_kmer_complexity_per_base(
    sequence: str,
    k_values: list[int],
    percentile: float,
) -> float:
    valid_k = sorted({k for k in k_values if k > 0 and k <= len(sequence)})
    if not valid_k:
        return shannon_entropy_per_base(sequence)
    values = sorted(kmer_complexity_per_base(sequence, k) for k in valid_k)
    if len(values) == 1:
        return values[0]
    percentile = max(0.0, min(1.0, percentile))
    position = percentile * float(len(values) - 1)
    low_index = int(math.floor(position))
    high_index = int(math.ceil(position))
    if low_index == high_index:
        return values[low_index]
    fraction = position - low_index
    return values[low_index] * (1.0 - fraction) + values[high_index] * fraction


def estimated_sequence_bits(sequence: str, k_values: list[int], percentile: float) -> float:
    return len(sequence) * multiscale_kmer_complexity_per_base(
        sequence,
        k_values=k_values,
        percentile=percentile,
    )


def sample_poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    threshold = math.exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def sample_program_repeat_count(
    *,
    rng: random.Random,
    max_repeat: int,
    single_repeat_prob: float,
    long_repeat_prob: float,
    repeat_lambda: float,
    long_repeat_lambda: float,
) -> int:
    if max_repeat <= 1:
        return 1
    draw = rng.random()
    if draw < single_repeat_prob:
        repeat = 1
    elif draw < single_repeat_prob + long_repeat_prob:
        repeat = 1 + sample_poisson(rng, long_repeat_lambda)
    else:
        repeat = 1 + sample_poisson(rng, repeat_lambda)
    return max(1, min(max_repeat, repeat))


def generate_motif_program_sequence(
    *,
    max_seq_len: int,
    min_seq_len: int,
    max_motifs: int,
    max_motif_len: int,
    repeat_lambda: float,
    long_repeat_lambda: float,
    long_repeat_prob: float,
    single_repeat_prob: float,
    rng: random.Random,
    max_tries: int,
) -> tuple[str, str, float, float]:
    best_sequence = ""
    best_bits = float("inf")
    max_motifs = max(1, max_motifs)
    max_motif_len = max(1, max_motif_len)

    for _ in range(max_tries):
        motif_count = rng.choices(
            population=list(range(1, max_motifs + 1)),
            weights=[1.0 / float(index) for index in range(1, max_motifs + 1)],
            k=1,
        )[0]
        parts = []
        program_bits = math.log2(float(motif_count + 1))
        for motif_index in range(motif_count):
            remaining = max_seq_len - sum(len(part) for part in parts)
            remaining_motifs = motif_count - motif_index
            if remaining <= 0:
                break
            motif_len = rng.randint(1, min(max_motif_len, remaining))
            motif = random_dna_block(rng, motif_len)
            max_repeat = max(1, (remaining - (remaining_motifs - 1)) // motif_len)
            repeat = sample_program_repeat_count(
                rng=rng,
                max_repeat=max_repeat,
                single_repeat_prob=single_repeat_prob,
                long_repeat_prob=long_repeat_prob,
                repeat_lambda=repeat_lambda,
                long_repeat_lambda=long_repeat_lambda,
            )
            parts.append(motif * repeat)
            program_bits += 2.0 * motif_len + math.log2(float(repeat + 1))

        sequence = "".join(parts)[:max_seq_len]
        if not sequence:
            continue
        if len(sequence) >= min_seq_len:
            return sequence, f"program{motif_count}", program_bits, program_bits / float(len(sequence))
        if not best_sequence or len(sequence) > len(best_sequence):
            best_sequence = sequence
            best_bits = program_bits

    if not best_sequence:
        best_sequence = rng.choice(DNA_BASES)
        best_bits = 2.0
    return best_sequence, "program_fallback", best_bits, best_bits / float(len(best_sequence))


def generate_entropy_budget_sequence(
    *,
    max_seq_len: int,
    min_seq_len: int,
    bit_budget: float,
    kmer_sizes: list[int],
    entropy_percentile: float,
    families: list[str],
    rng: random.Random,
    max_tries: int,
) -> tuple[str, str, float, float]:
    if not families:
        raise ValueError("At least one single-token family is required.")
    best_sequence = ""
    best_kind = families[0]
    best_bits = float("inf")
    best_entropy = float("inf")
    for _ in range(max_tries):
        seq_len = rng.randint(min_seq_len, max_seq_len)
        kind = rng.choice(families)
        sequence, _ = generate_family_sequence(seq_len, kind, rng)
        entropy = multiscale_kmer_complexity_per_base(
            sequence,
            k_values=kmer_sizes,
            percentile=entropy_percentile,
        )
        bits = len(sequence) * entropy
        if bits < best_bits:
            best_sequence = sequence
            best_kind = kind
            best_bits = bits
            best_entropy = entropy
        if bits <= bit_budget:
            return sequence, kind, bits, entropy

    # If a tight budget rejects everything by chance, fall back to the lowest
    # complexity candidate observed rather than hanging the training loop.
    return best_sequence, best_kind, best_bits, best_entropy


def generate_single_token_curriculum_sequence(
    *,
    curriculum_mode: str,
    max_seq_len: int,
    min_seq_len: int,
    bit_budget: float,
    kmer_sizes: list[int],
    entropy_percentile: float,
    families: list[str],
    rng: random.Random,
    max_tries: int,
    program_max_motifs: int,
    program_max_motif_len: int,
    program_repeat_lambda: float,
    program_long_repeat_lambda: float,
    program_long_repeat_prob: float,
    program_single_repeat_prob: float,
) -> tuple[str, str, float, float]:
    if curriculum_mode == "program":
        return generate_motif_program_sequence(
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            max_motifs=program_max_motifs,
            max_motif_len=program_max_motif_len,
            repeat_lambda=program_repeat_lambda,
            long_repeat_lambda=program_long_repeat_lambda,
            long_repeat_prob=program_long_repeat_prob,
            single_repeat_prob=program_single_repeat_prob,
            rng=rng,
            max_tries=max_tries,
        )
    if curriculum_mode == "entropy":
        return generate_entropy_budget_sequence(
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            bit_budget=bit_budget,
            kmer_sizes=kmer_sizes,
            entropy_percentile=entropy_percentile,
            families=families,
            rng=rng,
            max_tries=max_tries,
        )
    raise ValueError(f"Unknown curriculum mode: {curriculum_mode}")


def make_single_token_entropy_batch(
    *,
    batch_size: int,
    curriculum_mode: str,
    max_seq_len: int,
    min_seq_len: int,
    bit_budget: float,
    kmer_sizes: list[int],
    entropy_percentile: float,
    families: list[str],
    rng: random.Random,
    device: torch.device,
    max_tries: int,
    program_max_motifs: int,
    program_max_motif_len: int,
    program_repeat_lambda: float,
    program_long_repeat_lambda: float,
    program_long_repeat_prob: float,
    program_single_repeat_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    target = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.float32)
    bits = torch.zeros(batch_size, dtype=torch.float32)
    entropies = torch.zeros(batch_size, dtype=torch.float32)
    kinds = []
    for batch_index in range(batch_size):
        sequence, kind, sequence_bits, entropy = generate_single_token_curriculum_sequence(
            curriculum_mode=curriculum_mode,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            bit_budget=bit_budget,
            kmer_sizes=kmer_sizes,
            entropy_percentile=entropy_percentile,
            families=families,
            rng=rng,
            max_tries=max_tries,
            program_max_motifs=program_max_motifs,
            program_max_motif_len=program_max_motif_len,
            program_repeat_lambda=program_repeat_lambda,
            program_long_repeat_lambda=program_long_repeat_lambda,
            program_long_repeat_prob=program_long_repeat_prob,
            program_single_repeat_prob=program_single_repeat_prob,
        )
        sequence_tensor = sequence_to_tensor(sequence)
        seq_len = len(sequence)
        target[batch_index, :seq_len] = sequence_tensor
        mask[batch_index, :seq_len] = 1.0
        lengths[batch_index] = seq_len
        bits[batch_index] = sequence_bits
        entropies[batch_index] = entropy
        kinds.append(kind)
    return (
        target.to(device),
        mask.to(device),
        lengths.to(device),
        bits.to(device),
        entropies.to(device),
        kinds,
    )


def make_single_token_entropy_family_batch(
    *,
    seed_count: int,
    family_size: int,
    curriculum_mode: str,
    max_seq_len: int,
    min_seq_len: int,
    bit_budget: float,
    kmer_sizes: list[int],
    entropy_percentile: float,
    families: list[str],
    rng: random.Random,
    device: torch.device,
    max_tries: int,
    substitution_rate: float,
    indel_rate: float,
    repeat_rate: float,
    max_edits: int,
    program_max_motifs: int,
    program_max_motif_len: int,
    program_repeat_lambda: float,
    program_long_repeat_lambda: float,
    program_long_repeat_prob: float,
    program_single_repeat_prob: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
    torch.Tensor,
]:
    if seed_count <= 0 or family_size <= 0:
        raise ValueError("seed_count and family_size must be positive.")
    total = seed_count * family_size
    target = torch.zeros(total, max_seq_len, dtype=torch.long)
    mask = torch.zeros(total, max_seq_len, dtype=torch.float32)
    lengths = torch.zeros(total, dtype=torch.float32)
    bits = torch.zeros(total, dtype=torch.float32)
    entropies = torch.zeros(total, dtype=torch.float32)
    sequences: list[str] = []
    kinds: list[str] = []
    family_ids = torch.zeros(total, dtype=torch.long)
    row = 0
    for family_index in range(seed_count):
        seed_sequence, kind, _seed_bits, _seed_entropy = generate_single_token_curriculum_sequence(
            curriculum_mode=curriculum_mode,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            bit_budget=bit_budget,
            kmer_sizes=kmer_sizes,
            entropy_percentile=entropy_percentile,
            families=families,
            rng=rng,
            max_tries=max_tries,
            program_max_motifs=program_max_motifs,
            program_max_motif_len=program_max_motif_len,
            program_repeat_lambda=program_repeat_lambda,
            program_long_repeat_lambda=program_long_repeat_lambda,
            program_long_repeat_prob=program_long_repeat_prob,
            program_single_repeat_prob=program_single_repeat_prob,
        )
        variants = [seed_sequence]
        for variant_index in range(1, family_size):
            scale = 1.0 + float(variant_index - 1) / float(max(1, family_size - 1))
            variants.append(
                augment_sequence(
                    seed_sequence,
                    max_seq_len=max_seq_len,
                    rng=rng,
                    substitution_rate=substitution_rate * scale,
                    indel_rate=indel_rate * scale,
                    repeat_rate=min(1.0, repeat_rate * scale),
                    max_edits=max_edits,
                )
            )
        for sequence in variants:
            sequence_tensor = sequence_to_tensor(sequence)
            seq_len = len(sequence)
            entropy = multiscale_kmer_complexity_per_base(
                sequence,
                k_values=kmer_sizes,
                percentile=entropy_percentile,
            )
            target[row, :seq_len] = sequence_tensor
            mask[row, :seq_len] = 1.0
            lengths[row] = seq_len
            bits[row] = seq_len * entropy
            entropies[row] = entropy
            sequences.append(sequence)
            kinds.append(kind)
            family_ids[row] = family_index
            row += 1
    return (
        target.to(device),
        mask.to(device),
        lengths.to(device),
        bits.to(device),
        entropies.to(device),
        sequences,
        kinds,
        family_ids.to(device),
    )


def augment_sequence(
    sequence: str,
    *,
    max_seq_len: int,
    rng: random.Random,
    substitution_rate: float,
    indel_rate: float,
    repeat_rate: float,
    max_edits: int,
) -> str:
    augmented = list(sequence)
    max_edits = max(1, max_edits)

    substitution_count = min(max_edits, max(0, int(round(len(sequence) * substitution_rate))))
    if substitution_rate > 0 and substitution_count == 0 and rng.random() < len(sequence) * substitution_rate:
        substitution_count = 1
    for _ in range(substitution_count):
        if not augmented:
            break
        index = rng.randrange(len(augmented))
        current = augmented[index]
        choices = [base for base in DNA_BASES if base != current]
        augmented[index] = rng.choice(choices)

    indel_count = min(max_edits, max(0, int(round(len(sequence) * indel_rate))))
    if indel_rate > 0 and indel_count == 0 and rng.random() < len(sequence) * indel_rate:
        indel_count = 1
    for _ in range(indel_count):
        if rng.random() < 0.5 and len(augmented) > 1:
            del augmented[rng.randrange(len(augmented))]
        elif len(augmented) < max_seq_len:
            augmented.insert(rng.randrange(len(augmented) + 1), rng.choice(DNA_BASES))

    if repeat_rate > 0 and rng.random() < repeat_rate and augmented:
        start = rng.randrange(len(augmented))
        span = rng.randint(1, min(6, len(augmented) - start))
        motif = augmented[start : start + span]
        if rng.random() < 0.5 and len(augmented) + span <= max_seq_len:
            augmented[start:start] = motif
        elif len(augmented) > span + 1:
            del augmented[start : start + span]

    if not augmented:
        augmented = [rng.choice(DNA_BASES)]
    return "".join(augmented[:max_seq_len])


def make_batch(
    *,
    batch_size: int,
    max_seq_len: int,
    min_seq_len: int,
    short_max_len_frac: float,
    long_min_len_frac: float,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    target = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.float32)
    kinds = []
    for batch_index in range(batch_size):
        sequence, kind = generate_variable_sequence(
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            short_max_len_frac=short_max_len_frac,
            long_min_len_frac=long_min_len_frac,
            rng=rng,
        )
        sequence_tensor = sequence_to_tensor(sequence)
        seq_len = len(sequence)
        target[batch_index, :seq_len] = sequence_tensor
        mask[batch_index, :seq_len] = 1.0
        lengths[batch_index] = seq_len
        kinds.append(kind)
    return target.to(device), mask.to(device), lengths.to(device), kinds


def make_augmented_pair_batch(
    *,
    batch_size: int,
    max_seq_len: int,
    min_seq_len: int,
    short_max_len_frac: float,
    long_min_len_frac: float,
    rng: random.Random,
    device: torch.device,
    substitution_rate: float,
    indel_rate: float,
    repeat_rate: float,
    max_edits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    target_a = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mask_a = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    lengths_a = torch.zeros(batch_size, dtype=torch.float32)
    target_b = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    mask_b = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    lengths_b = torch.zeros(batch_size, dtype=torch.float32)
    kinds = []
    for batch_index in range(batch_size):
        sequence, kind = generate_variable_sequence(
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            short_max_len_frac=short_max_len_frac,
            long_min_len_frac=long_min_len_frac,
            rng=rng,
        )
        augmented = augment_sequence(
            sequence,
            max_seq_len=max_seq_len,
            rng=rng,
            substitution_rate=substitution_rate,
            indel_rate=indel_rate,
            repeat_rate=repeat_rate,
            max_edits=max_edits,
        )
        for target, mask, lengths, seq in (
            (target_a, mask_a, lengths_a, sequence),
            (target_b, mask_b, lengths_b, augmented),
        ):
            sequence_tensor = sequence_to_tensor(seq)
            seq_len = len(seq)
            target[batch_index, :seq_len] = sequence_tensor
            mask[batch_index, :seq_len] = 1.0
            lengths[batch_index] = seq_len
        kinds.append(kind)
    return (
        target_a.to(device),
        mask_a.to(device),
        lengths_a.to(device),
        target_b.to(device),
        mask_b.to(device),
        lengths_b.to(device),
        kinds,
    )


def make_augmented_family_batch(
    *,
    seed_count: int,
    family_size: int,
    max_seq_len: int,
    min_seq_len: int,
    short_max_len_frac: float,
    long_min_len_frac: float,
    rng: random.Random,
    device: torch.device,
    substitution_rate: float,
    indel_rate: float,
    repeat_rate: float,
    max_edits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    if seed_count <= 0 or family_size <= 0:
        raise ValueError("seed_count and family_size must be positive.")
    total = seed_count * family_size
    target = torch.zeros(total, max_seq_len, dtype=torch.long)
    mask = torch.zeros(total, max_seq_len, dtype=torch.float32)
    lengths = torch.zeros(total, dtype=torch.float32)
    sequences: list[str] = []
    family_ids = torch.zeros(total, dtype=torch.long)
    row = 0
    for family_index in range(seed_count):
        seed_sequence, _kind = generate_variable_sequence(
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            short_max_len_frac=short_max_len_frac,
            long_min_len_frac=long_min_len_frac,
            rng=rng,
        )
        variants = [seed_sequence]
        for variant_index in range(1, family_size):
            # Later variants get slightly more opportunity to drift, producing a
            # useful within-family range of RapidFuzz similarities.
            scale = 1.0 + float(variant_index - 1) / float(max(1, family_size - 1))
            variants.append(
                augment_sequence(
                    seed_sequence,
                    max_seq_len=max_seq_len,
                    rng=rng,
                    substitution_rate=substitution_rate * scale,
                    indel_rate=indel_rate * scale,
                    repeat_rate=min(1.0, repeat_rate * scale),
                    max_edits=max_edits,
                )
            )
        for sequence in variants:
            sequence_tensor = sequence_to_tensor(sequence)
            seq_len = len(sequence)
            target[row, :seq_len] = sequence_tensor
            mask[row, :seq_len] = 1.0
            lengths[row] = seq_len
            sequences.append(sequence)
            family_ids[row] = family_index
            row += 1
    return target.to(device), mask.to(device), lengths.to(device), sequences, family_ids.to(device)


def make_probe_batch(
    *,
    max_seq_len: int,
    seq_lengths: list[int],
    block_types: list[str],
    block_lengths: list[int],
    examples_per_condition: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[str, int, int]]]:
    total_examples = len(seq_lengths) * len(block_types) * len(block_lengths) * examples_per_condition
    target = torch.zeros(total_examples, max_seq_len, dtype=torch.long)
    mask = torch.zeros(total_examples, max_seq_len, dtype=torch.float32)
    lengths = torch.zeros(total_examples, dtype=torch.float32)
    labels: list[tuple[str, int, int]] = []
    index = 0
    for block_type in block_types:
        for seq_len in seq_lengths:
            for block_len in block_lengths:
                for _ in range(examples_per_condition):
                    sequence = generate_controlled_sequence(
                        seq_len=seq_len,
                        block_type=block_type,
                        block_len=min(block_len, seq_len),
                        rng=rng,
                    )
                    sequence_tensor = sequence_to_tensor(sequence)
                    target[index, :seq_len] = sequence_tensor
                    mask[index, :seq_len] = 1.0
                    lengths[index] = seq_len
                    labels.append((block_type, seq_len, block_len))
                    index += 1
    return target.to(device), mask.to(device), lengths.to(device), labels


class AdaptiveTokenizerAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len: int,
        max_tokens: int,
        latent_dim: int,
        max_slots_per_token: int,
        hidden_dim: int,
        encoder_layers: int,
        decoder_hidden_dim: int,
        slot_dim: int,
        token_rank_temperature: float,
        token_usage_temperature: float,
        gate_temperature: float,
        pack_temperature: float,
        initial_token_stride: float,
    ):
        super().__init__()
        if max_tokens <= 0:
            raise ValueError("--max-tokens must be positive.")
        if initial_token_stride <= 0:
            raise ValueError("--initial-token-stride must be positive.")
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.max_slots_per_token = max_slots_per_token
        self.token_rank_temperature = token_rank_temperature
        self.token_usage_temperature = token_usage_temperature
        self.gate_temperature = gate_temperature
        self.pack_temperature = pack_temperature

        blocks = []
        in_channels = 6
        for _ in range(encoder_layers):
            blocks.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.GELU(),
                ]
            )
            in_channels = hidden_dim
        self.encoder = nn.Sequential(*blocks)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.token_head = nn.Linear(hidden_dim, 1)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.slot_embedding = nn.Embedding(max_slots_per_token, slot_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + slot_dim + 1, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, 4),
        )
        self.length_head = nn.Linear(latent_dim, 1)

        initial_token_prob = min(0.95, max(0.01, 1.0 / initial_token_stride))
        nn.init.constant_(self.token_head.bias, math.log(initial_token_prob / (1.0 - initial_token_prob)))
        nn.init.normal_(self.token_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.length_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.length_head.bias, math.log(8.0 / max(1.0, max_slots_per_token - 8.0)))

    def encode_features(self, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(target, num_classes=4).to(dtype=torch.float32)
        mask = mask.to(dtype=one_hot.dtype)
        positions = torch.linspace(-1.0, 1.0, target.shape[1], device=target.device, dtype=one_hot.dtype)
        positions = positions[None, :, None].expand(target.shape[0], -1, -1)
        encoder_input = torch.cat([one_hot * mask[..., None], positions, mask[..., None]], dim=-1)
        features = self.encoder(encoder_input.transpose(1, 2)).transpose(1, 2)
        return self.encoder_norm(features)

    def tokenize(self, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_features(target, mask)
        token_logits = self.token_head(features).squeeze(-1)
        token_prob = torch.sigmoid(token_logits) * mask

        # Ensure every non-empty sequence can open at least one token while keeping
        # the remaining token count differentiable.
        first_position = F.one_hot(torch.zeros(target.shape[0], dtype=torch.long, device=target.device), target.shape[1])
        token_prob = torch.maximum(token_prob, first_position.to(dtype=token_prob.dtype) * mask[:, :1])

        token_centres = torch.cumsum(token_prob, dim=1) - 0.5 * token_prob
        token_ids = torch.arange(self.max_tokens, device=target.device, dtype=features.dtype) + 0.5
        assignment_logits = -(
            token_centres[:, None, :] - token_ids[None, :, None]
        ).pow(2) / max(self.token_rank_temperature, 1e-6)
        assignment_logits = assignment_logits.masked_fill(mask[:, None, :] <= 0, -1e4)
        token_weights = assignment_logits.softmax(dim=-1)

        token_mass = token_prob.sum(dim=1)
        token_usage = torch.sigmoid(
            (token_mass[:, None] - token_ids[None, :]) / max(self.token_usage_temperature, 1e-6)
        )
        token_weights = token_weights * token_usage[:, :, None]
        token_weights_norm = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        pooled = torch.einsum("bkl,blh->bkh", token_weights_norm, features)
        latents = self.to_latent(pooled)
        return {
            "features": features,
            "latents": latents,
            "token_logits": token_logits,
            "token_prob": token_prob,
            "token_mass": token_mass,
            "token_usage": token_usage,
            "token_weights": token_weights,
        }

    def decode(self, latents: torch.Tensor, token_usage: torch.Tensor, out_len: int) -> dict[str, torch.Tensor]:
        batch_size = latents.shape[0]
        slots = torch.arange(self.max_slots_per_token, device=latents.device)
        slot_embed = self.slot_embedding(slots)
        relative_position = ((slots.to(dtype=latents.dtype) + 0.5) / float(self.max_slots_per_token))[None, None, :, None]

        latent_expanded = latents[:, :, None, :].expand(-1, -1, self.max_slots_per_token, -1)
        slot_expanded = slot_embed[None, None, :, :].expand(batch_size, self.max_tokens, -1, -1)
        relative_expanded = relative_position.expand(batch_size, self.max_tokens, -1, -1)
        decoder_input = torch.cat([latent_expanded, slot_expanded, relative_expanded], dim=-1)
        base_logits_segmented = self.decoder(decoder_input)
        base_probs_segmented = base_logits_segmented.softmax(dim=-1)

        raw_lengths = self.max_slots_per_token * torch.sigmoid(self.length_head(latents).squeeze(-1))
        lengths = raw_lengths * token_usage
        slot_centres = slots.to(dtype=latents.dtype) + 0.5
        keep_segmented = torch.sigmoid((lengths[..., None] - slot_centres[None, None, :]) / self.gate_temperature)

        num_slots = self.max_tokens * self.max_slots_per_token
        base_probs = base_probs_segmented.reshape(batch_size, num_slots, 4)
        keep = keep_segmented.reshape(batch_size, num_slots)
        end = torch.cumsum(keep, dim=1)
        start = end - keep
        coords = torch.arange(out_len, device=latents.device, dtype=latents.dtype) + 0.5
        weights = torch.sigmoid((coords[None, None, :] - start[:, :, None]) / self.pack_temperature) - torch.sigmoid(
            (coords[None, None, :] - end[:, :, None]) / self.pack_temperature
        )
        weights = weights.clamp_min(0.0)
        weights_norm = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        soft_dna = torch.einsum("bjl,bjc->blc", weights_norm, base_probs)

        return {
            "base_logits_segmented": base_logits_segmented,
            "base_probs_segmented": base_probs_segmented,
            "lengths": lengths,
            "keep_segmented": keep_segmented,
            "keep": keep,
            "total_len": keep.sum(dim=1),
            "weights": weights_norm,
            "soft_dna": soft_dna,
            "base_entropy": -(base_probs.clamp_min(1e-8) * base_probs.clamp_min(1e-8).log()).sum(dim=-1).mean(),
            "pack_entropy": -(weights_norm.clamp_min(1e-8) * weights_norm.clamp_min(1e-8).log()).sum(dim=1).mean(),
            "pack_confidence": weights_norm.max(dim=1).values.mean(),
        }

    def forward(self, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        tokenized = self.tokenize(target, mask)
        decoded = self.decode(tokenized["latents"], tokenized["token_usage"], target.shape[1])
        return {**tokenized, **decoded}


def reconstruction_metrics(soft_dna: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    predicted = soft_dna.argmax(dim=-1)
    correct = predicted == target
    accuracy = (correct.float() * mask).sum() / mask.sum().clamp_min(1.0)
    exact = (correct | (mask <= 0)).all(dim=1).float().mean()
    return {"accuracy": float(accuracy.item()), "exact": float(exact.item())}


def softmin(values: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(temperature, 1e-6)
    weights = torch.softmax(-values / temperature, dim=0)
    return (weights * values).sum()


def soft_alignment_nll_one(
    soft_dna: torch.Tensor,
    target: torch.Tensor,
    target_len: int,
    *,
    temperature: float,
    gap_cost: float,
) -> torch.Tensor:
    target_len = max(1, min(target_len, soft_dna.shape[0]))
    probs = soft_dna[:target_len].clamp_min(1e-8)
    target_slice = target[:target_len]
    pair_cost = -probs[:, None, :].expand(target_len, target_len, 4).gather(
        -1,
        target_slice[None, :, None].expand(target_len, target_len, 1),
    ).squeeze(-1).log()

    gap = pair_cost.new_tensor(gap_cost)
    previous = [pair_cost.new_tensor(float(j) * gap_cost) for j in range(target_len + 1)]
    for out_index in range(1, target_len + 1):
        current = [pair_cost.new_tensor(float(out_index) * gap_cost)]
        for target_index in range(1, target_len + 1):
            candidates = torch.stack(
                [
                    previous[target_index - 1] + pair_cost[out_index - 1, target_index - 1],
                    previous[target_index] + gap,
                    current[target_index - 1] + gap,
                ]
            )
            current.append(softmin(candidates, temperature))
        previous = current
    return previous[target_len] / float(target_len)


def soft_alignment_nll(
    soft_dna: torch.Tensor,
    target: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    temperature: float,
    gap_cost: float,
) -> torch.Tensor:
    losses = []
    for batch_index in range(soft_dna.shape[0]):
        losses.append(
            soft_alignment_nll_one(
                soft_dna[batch_index],
                target[batch_index],
                int(round(float(target_lengths[batch_index].detach().cpu().item()))),
                temperature=temperature,
                gap_cost=gap_cost,
            )
        )
    return torch.stack(losses).mean()


def local_window_alignment_nll(
    soft_dna: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    window: int,
    temperature: float,
    shift_cost: float,
) -> torch.Tensor:
    window = max(0, int(window))
    temperature = max(temperature, 1e-6)
    batch_size, seq_len, _ = soft_dna.shape
    costs = []
    validity = []
    large = soft_dna.new_tensor(1e4)

    for shift in range(-window, window + 1):
        shifted_cost = soft_dna.new_full((batch_size, seq_len), 1e4)
        shifted_valid = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=soft_dna.device)
        if shift >= 0:
            target_slice = target[:, : seq_len - shift]
            prob_slice = soft_dna[:, shift:, :]
            valid_slice = mask[:, : seq_len - shift] > 0
            nll = -prob_slice.gather(-1, target_slice[..., None]).squeeze(-1).clamp_min(1e-8).log()
            shifted_cost[:, : seq_len - shift] = nll + abs(shift) * shift_cost
            shifted_valid[:, : seq_len - shift] = valid_slice
        else:
            offset = -shift
            target_slice = target[:, offset:]
            prob_slice = soft_dna[:, : seq_len - offset, :]
            valid_slice = mask[:, offset:] > 0
            nll = -prob_slice.gather(-1, target_slice[..., None]).squeeze(-1).clamp_min(1e-8).log()
            shifted_cost[:, offset:] = nll + abs(shift) * shift_cost
            shifted_valid[:, offset:] = valid_slice
        costs.append(shifted_cost)
        validity.append(shifted_valid)

    cost_stack = torch.stack(costs, dim=1)
    valid_stack = torch.stack(validity, dim=1)
    cost_stack = torch.where(valid_stack, cost_stack, large)
    weights = torch.softmax(-cost_stack / temperature, dim=1)
    position_loss = (weights * cost_stack).sum(dim=1)
    return (position_loss * mask).sum() / mask.sum().clamp_min(1.0)


def global_shift_alignment_nll(
    soft_dna: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
    shift_cost: float,
    max_shift: int,
) -> torch.Tensor:
    temperature = max(temperature, 1e-6)
    batch_size, seq_len, _ = soft_dna.shape
    if max_shift < 0:
        max_shift = seq_len - 1
    max_shift = min(max_shift, seq_len - 1)
    costs = []
    large = soft_dna.new_tensor(1e4)

    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            target_slice = target[:, : seq_len - shift]
            prob_slice = soft_dna[:, shift:, :]
            valid = mask[:, : seq_len - shift]
        else:
            offset = -shift
            target_slice = target[:, offset:]
            prob_slice = soft_dna[:, : seq_len - offset, :]
            valid = mask[:, offset:]
        nll = -prob_slice.gather(-1, target_slice[..., None]).squeeze(-1).clamp_min(1e-8).log()
        denom = valid.sum(dim=1).clamp_min(1.0)
        cost = (nll * valid).sum(dim=1) / denom
        cost = cost + abs(shift) * shift_cost
        cost = torch.where(valid.sum(dim=1) > 0, cost, large.expand(batch_size))
        costs.append(cost)

    cost_stack = torch.stack(costs, dim=1)
    weights = torch.softmax(-cost_stack / temperature, dim=1)
    return (weights * cost_stack).sum(dim=1).mean()


def alignment_nll(
    soft_dna: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    mode: str,
    temperature: float,
    gap_cost: float,
    window: int,
    shift_cost: float,
    global_weight: float,
    global_window: int,
) -> torch.Tensor:
    if mode == "local_window":
        local_loss = local_window_alignment_nll(
            soft_dna,
            target,
            mask,
            window=window,
            temperature=temperature,
            shift_cost=shift_cost,
        )
        if global_weight <= 0:
            return local_loss
        global_loss = global_shift_alignment_nll(
            soft_dna,
            target,
            mask,
            temperature=temperature,
            shift_cost=shift_cost,
            max_shift=global_window,
        )
        return local_loss + global_weight * global_loss
    if mode == "dp":
        return soft_alignment_nll(
            soft_dna,
            target,
            target_lengths,
            temperature=temperature,
            gap_cost=gap_cost,
        )
    if mode == "none":
        return soft_dna.sum() * 0.0
    raise ValueError(f"Unknown alignment mode: {mode}")


def loss_for_batch(
    *,
    model: AdaptiveTokenizerAutoencoder,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_lengths: torch.Tensor,
    length_weight: float,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
    latent_l2_weight: float,
    alignment_loss_weight: float = 0.0,
    alignment_mode: str = "local_window",
    alignment_temperature: float = 0.1,
    alignment_gap_cost: float = 0.75,
    alignment_window: int = 3,
    alignment_shift_cost: float = 0.02,
    alignment_global_weight: float = 0.5,
    alignment_global_window: int = -1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rendered = model(target, mask)
    soft_dna = rendered["soft_dna"].clamp_min(1e-8)
    token_nll = -soft_dna.gather(-1, target[..., None]).squeeze(-1).log()
    recon_loss = (token_nll * mask).sum() / mask.sum().clamp_min(1.0)
    alignment_loss = (
        alignment_nll(
            soft_dna,
            target,
            mask,
            target_lengths,
            mode=alignment_mode,
            temperature=alignment_temperature,
            gap_cost=alignment_gap_cost,
            window=alignment_window,
            shift_cost=alignment_shift_cost,
            global_weight=alignment_global_weight,
            global_window=alignment_global_window,
        )
        if alignment_loss_weight > 0
        else recon_loss.detach() * 0.0
    )
    length_loss = F.smooth_l1_loss(rendered["total_len"], target_lengths)
    token_count = rendered["token_usage"].sum(dim=1)
    token_cost = token_count.mean()
    token_sharp = (rendered["token_prob"] * (1.0 - rendered["token_prob"]) * mask).sum() / mask.sum().clamp_min(1.0)
    decoder_sharp = (rendered["keep"] * (1.0 - rendered["keep"])).mean()
    latent_l2 = rendered["latents"].pow(2).mean()
    loss = (
        recon_loss
        + alignment_loss_weight * alignment_loss
        + length_weight * length_loss
        + token_cost_weight * token_cost
        + token_sharp_weight * token_sharp
        + decoder_sharp_weight * decoder_sharp
        + latent_l2_weight * latent_l2
    )
    return loss, {
        **rendered,
        "recon_loss": recon_loss.detach(),
        "alignment_loss": alignment_loss.detach(),
        "length_loss": length_loss.detach(),
        "token_cost": token_cost.detach(),
        "token_sharp": token_sharp.detach(),
        "decoder_sharp": decoder_sharp.detach(),
        "latent_l2": latent_l2.detach(),
        "token_count": token_count.detach(),
    }


def expected_token_centres(rendered: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    seq_len = mask.shape[1]
    coords = torch.linspace(0.0, 1.0, seq_len, device=mask.device, dtype=mask.dtype)
    weights = rendered["token_weights"].to(dtype=mask.dtype) * mask[:, None, :]
    denom = weights.sum(dim=-1).clamp_min(1e-8)
    return (weights * coords[None, None, :]).sum(dim=-1) / denom


def soft_position_aligned_latent_loss(
    rendered_a: dict[str, torch.Tensor],
    mask_a: torch.Tensor,
    rendered_b: dict[str, torch.Tensor],
    mask_b: torch.Tensor,
    *,
    position_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    latents_a = F.normalize(rendered_a["latents"], dim=-1)
    latents_b = F.normalize(rendered_b["latents"], dim=-1)
    centres_a = expected_token_centres(rendered_a, mask_a)
    centres_b = expected_token_centres(rendered_b, mask_b)
    usage_a = rendered_a["token_usage"]
    usage_b = rendered_b["token_usage"]

    position_distance = (centres_a[:, :, None] - centres_b[:, None, :]).abs()
    match_logits_ab = -position_distance / max(temperature, 1e-6)
    match_logits_ab = match_logits_ab + usage_b[:, None, :].clamp_min(1e-4).log()
    match_ab = match_logits_ab.softmax(dim=-1)
    aligned_b = torch.einsum("bkj,bjd->bkd", match_ab, latents_b)

    match_logits_ba = -position_distance.transpose(1, 2) / max(temperature, 1e-6)
    match_logits_ba = match_logits_ba + usage_a[:, None, :].clamp_min(1e-4).log()
    match_ba = match_logits_ba.softmax(dim=-1)
    aligned_a = torch.einsum("bjk,bkd->bjd", match_ba, latents_a)

    latent_loss_ab = (latents_a - aligned_b).pow(2).sum(dim=-1)
    latent_loss_ba = (latents_b - aligned_a).pow(2).sum(dim=-1)
    pos_loss_ab = (match_ab * position_distance).sum(dim=-1)
    pos_loss_ba = (match_ba * position_distance.transpose(1, 2)).sum(dim=-1)

    weighted_ab = usage_a * (latent_loss_ab + position_weight * pos_loss_ab)
    weighted_ba = usage_b * (latent_loss_ba + position_weight * pos_loss_ba)
    denom = usage_a.sum() + usage_b.sum()
    loss = (weighted_ab.sum() + weighted_ba.sum()) / denom.clamp_min(1e-8)
    mean_position_shift = (usage_a * pos_loss_ab).sum() / usage_a.sum().clamp_min(1e-8)
    mean_latent_distance = (usage_a * latent_loss_ab).sum() / usage_a.sum().clamp_min(1e-8)
    return loss, {
        "manifold_loss": loss.detach(),
        "manifold_position_shift": mean_position_shift.detach(),
        "manifold_latent_distance": mean_latent_distance.detach(),
    }


def rapidfuzz_ratio01(a: str, b: str) -> float:
    if rapidfuzz_fuzz is not None:
        return float(rapidfuzz_fuzz.ratio(a, b)) / 100.0
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def rapidfuzz_similarity_matrix(sequences: list[str], device: torch.device) -> torch.Tensor:
    count = len(sequences)
    matrix = torch.eye(count, dtype=torch.float32)
    for i in range(count):
        for j in range(i + 1, count):
            value = rapidfuzz_ratio01(sequences[i], sequences[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix.to(device)


def sequence_latent_embeddings(rendered: dict[str, torch.Tensor]) -> torch.Tensor:
    latents = rendered["latents"]
    usage = rendered["token_usage"].to(dtype=latents.dtype)
    weighted = (latents * usage[..., None]).sum(dim=1) / usage.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.normalize(weighted, dim=-1)


def centred_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.square().mean().sqrt() * b.square().mean().sqrt()).clamp_min(1e-8)


def rapidfuzz_geometry_loss(
    rendered: dict[str, torch.Tensor],
    sequences: list[str],
    family_ids: torch.Tensor,
    *,
    within_family_only: bool,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_similarity = rapidfuzz_similarity_matrix(sequences, rendered["latents"].device)
    embeddings = sequence_latent_embeddings(rendered)
    latent_similarity = embeddings @ embeddings.transpose(0, 1)
    latent_similarity = 0.5 * (latent_similarity + 1.0)

    count = len(sequences)
    pair_mask = ~torch.eye(count, dtype=torch.bool, device=latent_similarity.device)
    if within_family_only:
        pair_mask = pair_mask & (family_ids[:, None] == family_ids[None, :])
    if margin > 0:
        pair_mask = pair_mask & (target_similarity >= margin)
    if pair_mask.sum() == 0:
        zero = latent_similarity.sum() * 0.0
        return zero, {
            "rapidfuzz_geometry_loss": zero.detach(),
            "rapidfuzz_corr": zero.detach(),
            "rapidfuzz_target_mean": zero.detach(),
            "latent_similarity_mean": zero.detach(),
        }
    target_pairs = target_similarity[pair_mask]
    latent_pairs = latent_similarity[pair_mask]
    loss = F.mse_loss(latent_pairs, target_pairs)
    corr = centred_correlation(latent_pairs.detach(), target_pairs.detach())
    return loss, {
        "rapidfuzz_geometry_loss": loss.detach(),
        "rapidfuzz_corr": corr.detach(),
        "rapidfuzz_target_mean": target_pairs.detach().mean(),
        "latent_similarity_mean": latent_pairs.detach().mean(),
    }


def loss_for_positive_pair_batch(
    *,
    model: AdaptiveTokenizerAutoencoder,
    target_a: torch.Tensor,
    mask_a: torch.Tensor,
    lengths_a: torch.Tensor,
    target_b: torch.Tensor,
    mask_b: torch.Tensor,
    lengths_b: torch.Tensor,
    length_weight: float,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
    latent_l2_weight: float,
    manifold_weight: float,
    manifold_position_weight: float,
    manifold_temperature: float,
    alignment_loss_weight: float = 0.0,
    alignment_mode: str = "local_window",
    alignment_temperature: float = 0.1,
    alignment_gap_cost: float = 0.75,
    alignment_window: int = 3,
    alignment_shift_cost: float = 0.02,
    alignment_global_weight: float = 0.5,
    alignment_global_window: int = -1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_a, rendered_a = loss_for_batch(
        model=model,
        target=target_a,
        mask=mask_a,
        target_lengths=lengths_a,
        length_weight=length_weight,
        token_cost_weight=token_cost_weight,
        token_sharp_weight=token_sharp_weight,
        decoder_sharp_weight=decoder_sharp_weight,
        latent_l2_weight=latent_l2_weight,
        alignment_loss_weight=alignment_loss_weight,
        alignment_mode=alignment_mode,
        alignment_temperature=alignment_temperature,
        alignment_gap_cost=alignment_gap_cost,
        alignment_window=alignment_window,
        alignment_shift_cost=alignment_shift_cost,
        alignment_global_weight=alignment_global_weight,
        alignment_global_window=alignment_global_window,
    )
    loss_b, rendered_b = loss_for_batch(
        model=model,
        target=target_b,
        mask=mask_b,
        target_lengths=lengths_b,
        length_weight=length_weight,
        token_cost_weight=token_cost_weight,
        decoder_sharp_weight=decoder_sharp_weight,
        token_sharp_weight=token_sharp_weight,
        latent_l2_weight=latent_l2_weight,
        alignment_loss_weight=alignment_loss_weight,
        alignment_mode=alignment_mode,
        alignment_temperature=alignment_temperature,
        alignment_gap_cost=alignment_gap_cost,
        alignment_window=alignment_window,
        alignment_shift_cost=alignment_shift_cost,
        alignment_global_weight=alignment_global_weight,
        alignment_global_window=alignment_global_window,
    )
    manifold_loss, manifold_metrics = soft_position_aligned_latent_loss(
        rendered_a,
        mask_a,
        rendered_b,
        mask_b,
        position_weight=manifold_position_weight,
        temperature=manifold_temperature,
    )
    loss = 0.5 * (loss_a + loss_b) + manifold_weight * manifold_loss
    rendered_a = {
        **rendered_a,
        "paired_recon_loss": (0.5 * (rendered_a["recon_loss"] + rendered_b["recon_loss"])).detach(),
        "manifold_weight": torch.tensor(manifold_weight, device=target_a.device),
        **manifold_metrics,
    }
    return loss, rendered_a


def loss_for_family_geometry_batch(
    *,
    model: AdaptiveTokenizerAutoencoder,
    target: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    sequences: list[str],
    family_ids: torch.Tensor,
    length_weight: float,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
    latent_l2_weight: float,
    manifold_weight: float,
    within_family_only: bool,
    similarity_margin: float,
    alignment_loss_weight: float = 0.0,
    alignment_mode: str = "local_window",
    alignment_temperature: float = 0.1,
    alignment_gap_cost: float = 0.75,
    alignment_window: int = 3,
    alignment_shift_cost: float = 0.02,
    alignment_global_weight: float = 0.5,
    alignment_global_window: int = -1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recon_loss, rendered = loss_for_batch(
        model=model,
        target=target,
        mask=mask,
        target_lengths=lengths,
        length_weight=length_weight,
        token_cost_weight=token_cost_weight,
        token_sharp_weight=token_sharp_weight,
        decoder_sharp_weight=decoder_sharp_weight,
        latent_l2_weight=latent_l2_weight,
        alignment_loss_weight=alignment_loss_weight,
        alignment_mode=alignment_mode,
        alignment_temperature=alignment_temperature,
        alignment_gap_cost=alignment_gap_cost,
        alignment_window=alignment_window,
        alignment_shift_cost=alignment_shift_cost,
        alignment_global_weight=alignment_global_weight,
        alignment_global_window=alignment_global_window,
    )
    geometry_loss, geometry_metrics = rapidfuzz_geometry_loss(
        rendered,
        sequences,
        family_ids,
        within_family_only=within_family_only,
        margin=similarity_margin,
    )
    loss = recon_loss + manifold_weight * geometry_loss
    return loss, {
        **rendered,
        "manifold_loss": geometry_loss.detach(),
        "manifold_position_shift": torch.zeros((), device=target.device),
        "manifold_latent_distance": (1.0 - geometry_metrics["latent_similarity_mean"]).detach(),
        **geometry_metrics,
    }


def pick_device(args: argparse.Namespace) -> torch.device:
    if args.mps:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --mps, but MPS is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def current_weight(step: int, steps: int, weight: float, warmup_frac: float) -> float:
    if weight <= 0:
        return 0.0
    if warmup_frac <= 0:
        return weight
    warmup_steps = int(steps * warmup_frac)
    if step < warmup_steps:
        return 0.0
    ramp_steps = max(1, steps - warmup_steps)
    return weight * min(1.0, float(step - warmup_steps + 1) / float(ramp_steps))


def parse_int_list(values: str) -> list[int]:
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def parse_str_list(values: str) -> list[str]:
    parsed = [value.strip() for value in values.split(",") if value.strip()]
    unknown = [value for value in parsed if value not in BLOCK_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown block type(s): {unknown}. Known: {sorted(BLOCK_BUILDERS)}")
    return parsed


def parse_family_list(values: str) -> list[str]:
    parsed = [value.strip() for value in values.split(",") if value.strip()]
    unknown = [value for value in parsed if value not in TRAIN_FAMILIES or value == "hybrid"]
    if unknown:
        raise ValueError(
            f"Unknown single-token family/families: {unknown}. "
            f"Use non-hybrid families from: {sorted(BLOCK_BUILDERS)}"
        )
    return parsed


def parse_kmer_sizes(values: str) -> list[int]:
    parsed = parse_int_list(values)
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("--single-token-kmer-sizes must contain positive integers.")
    return sorted(set(parsed))


def single_token_families(args: argparse.Namespace) -> list[str]:
    return parse_family_list(args.single_token_families)


def single_token_kmer_sizes(args: argparse.Namespace) -> list[int]:
    if args.single_token_kmer_size is not None:
        return [args.single_token_kmer_size]
    return parse_kmer_sizes(args.single_token_kmer_sizes)


def single_token_generation_kwargs(
    args: argparse.Namespace,
    *,
    families: list[str] | None = None,
    kmer_sizes: list[int] | None = None,
) -> dict:
    return {
        "curriculum_mode": args.curriculum_mode,
        "max_seq_len": args.seq_len,
        "min_seq_len": args.single_token_min_len,
        "bit_budget": args.single_token_bit_budget,
        "kmer_sizes": kmer_sizes if kmer_sizes is not None else single_token_kmer_sizes(args),
        "entropy_percentile": args.single_token_entropy_percentile,
        "families": families if families is not None else single_token_families(args),
        "max_tries": args.single_token_max_tries,
        "program_max_motifs": args.program_max_motifs,
        "program_max_motif_len": args.program_max_motif_len,
        "program_repeat_lambda": args.program_repeat_lambda,
        "program_long_repeat_lambda": args.program_long_repeat_lambda,
        "program_long_repeat_prob": args.program_long_repeat_prob,
        "program_single_repeat_prob": args.program_single_repeat_prob,
    }


def curriculum_summary(rendered: dict[str, torch.Tensor]) -> str:
    if "info_bits" not in rendered or "entropy_per_base" not in rendered:
        return ""
    bits = rendered["info_bits"]
    entropy = rendered["entropy_per_base"]
    return (
        f" | bits {bits.mean().item():.1f}/{bits.std(unbiased=False).item():.1f} "
        f"range {bits.min().item():.1f}-{bits.max().item():.1f} "
        f"entropy {entropy.mean().item():.2f}/{entropy.std(unbiased=False).item():.2f}"
    )


def summarise_batch(rendered: dict[str, torch.Tensor], target_lengths: torch.Tensor) -> str:
    token_count = rendered["token_count"]
    token_density = token_count / target_lengths.clamp_min(1.0)
    lengths = rendered["lengths"]
    active = (lengths > 1.0).float().sum(dim=1)
    return (
        f"target_len {target_lengths.mean().item():.1f}/{target_lengths.std(unbiased=False).item():.1f} "
        f"range {target_lengths.min().item():.0f}-{target_lengths.max().item():.0f} | "
        f"out {rendered['total_len'].mean().item():.1f}/{rendered['total_len'].std(unbiased=False).item():.1f} "
        f"range {rendered['total_len'].min().item():.1f}-{rendered['total_len'].max().item():.1f} | "
        f"tokens {token_count.mean().item():.2f}/{token_count.std(unbiased=False).item():.2f} "
        f"range {token_count.min().item():.1f}-{token_count.max().item():.1f} | "
        f"tok_per_base {token_density.mean().item():.3f} | "
        f"active_emit {active.mean().item():.1f}/{active.std(unbiased=False).item():.1f}"
        f"{curriculum_summary(rendered)}"
    )


def compact_status(
    *,
    step: int,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
    manifold_weight: float,
    train_loss: torch.Tensor,
    train_rendered: dict[str, torch.Tensor],
    train_target: torch.Tensor,
    train_mask: torch.Tensor,
    train_lengths: torch.Tensor,
    val_loss: float,
    val_rendered: dict[str, torch.Tensor],
    val_target: torch.Tensor,
    val_mask: torch.Tensor,
    val_lengths: torch.Tensor,
) -> list[str]:
    train_metrics = reconstruction_metrics(train_rendered["soft_dna"], train_target, train_mask)
    val_metrics = reconstruction_metrics(val_rendered["soft_dna"], val_target, val_mask)
    train_tokens = train_rendered["token_count"]
    val_tokens = val_rendered["token_count"]
    manifold_text = ""
    if "manifold_loss" in train_rendered:
        manifold_text = (
            f" manifold {train_rendered['manifold_loss'].item():.4f}"
            f" pos {train_rendered['manifold_position_shift'].item():.3f}"
            f" zdist {train_rendered['manifold_latent_distance'].item():.3f}"
        )
        if "rapidfuzz_corr" in train_rendered:
            manifold_text += (
                f" rf_corr {train_rendered['rapidfuzz_corr'].item():.3f}"
                f" rf_mean {train_rendered['rapidfuzz_target_mean'].item():.3f}"
                f" lat_mean {train_rendered['latent_similarity_mean'].item():.3f}"
            )
    return [
        (
            f"\nstep {step:06d} | "
            f"val loss {val_loss:.4f} acc {val_metrics['accuracy']:.3f} exact {val_metrics['exact']:.3f} | "
            f"token_w {token_cost_weight:.2e} sharp_w {token_sharp_weight:.2e}/{decoder_sharp_weight:.2e} "
            f"manifold_w {manifold_weight:.2e}"
        ),
        (
            f"train loss {train_loss.item():.4f} ce {train_rendered['recon_loss'].item():.4f} "
            f"align {train_rendered['alignment_loss'].item():.4f} "
            f"acc {train_metrics['accuracy']:.3f} exact {train_metrics['exact']:.3f} "
            f"len_loss {train_rendered['length_loss'].item():.3f} "
            f"tokens {train_tokens.mean().item():.2f}/{train_tokens.std(unbiased=False).item():.2f} "
            f"target_len {train_lengths.mean().item():.1f}/{train_lengths.std(unbiased=False).item():.1f} "
            f"out {train_rendered['total_len'].mean().item():.1f}/{train_rendered['total_len'].std(unbiased=False).item():.1f}"
            f"{manifold_text}"
            f"{curriculum_summary(train_rendered)}"
        ),
        (
            f"val   ce {val_rendered['recon_loss'].item():.4f} "
            f"align {val_rendered['alignment_loss'].item():.4f} "
            f"len_loss {val_rendered['length_loss'].item():.3f} "
            f"tokens {val_tokens.mean().item():.2f}/{val_tokens.std(unbiased=False).item():.2f} "
            f"range {val_tokens.min().item():.1f}-{val_tokens.max().item():.1f} "
            f"tok/base {(val_tokens / val_lengths.clamp_min(1.0)).mean().item():.3f} "
            f"target_len {val_lengths.mean().item():.1f}/{val_lengths.std(unbiased=False).item():.1f} "
            f"out {val_rendered['total_len'].mean().item():.1f}/{val_rendered['total_len'].std(unbiased=False).item():.1f}"
            f"{curriculum_summary(val_rendered)}"
        ),
    ]


def format_metrics(prefix: str, loss: torch.Tensor, rendered: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.Tensor) -> str:
    metrics = reconstruction_metrics(rendered["soft_dna"], target, mask)
    manifold_text = ""
    if "manifold_loss" in rendered:
        manifold_text = (
            f" manifold {rendered['manifold_loss'].item():.4f}"
            f" pos {rendered['manifold_position_shift'].item():.3f}"
            f" zdist {rendered['manifold_latent_distance'].item():.3f}"
        )
        if "rapidfuzz_corr" in rendered:
            manifold_text += (
                f" rf_corr {rendered['rapidfuzz_corr'].item():.3f}"
                f" rf_mean {rendered['rapidfuzz_target_mean'].item():.3f}"
                f" lat_mean {rendered['latent_similarity_mean'].item():.3f}"
            )
    return (
        f"{prefix:<5} loss {loss.item():.4f} ce {rendered['recon_loss'].item():.4f} "
        f"align {rendered['alignment_loss'].item():.4f} "
        f"acc {metrics['accuracy']:.3f} exact {metrics['exact']:.3f} "
        f"len {rendered['length_loss'].item():.3f} "
        f"token_cost {rendered['token_cost'].item():.3f} "
        f"tok_sharp {rendered['token_sharp'].item():.3f} "
        f"dec_sharp {rendered['decoder_sharp'].item():.3f} "
        f"base_ent {rendered['base_entropy'].item():.3f} "
        f"pack_ent {rendered['pack_entropy'].item():.3f} "
        f"pack_conf {rendered['pack_confidence'].item():.3f}"
        f"{manifold_text}"
    )


def mean_for_indices(values: torch.Tensor, indices: list[int], device: torch.device) -> float:
    if not indices:
        return float("nan")
    index_tensor = torch.tensor(indices, device=device)
    return float(values[index_tensor].mean().item())


def fmt_mean(values: torch.Tensor, indices: list[int], device: torch.device, precision: int = 2) -> str:
    if not indices:
        return "na"
    return f"{mean_for_indices(values, indices, device):.{precision}f}"


def evaluate(
    *,
    model: AdaptiveTokenizerAutoencoder,
    rng: random.Random,
    device: torch.device,
    args: argparse.Namespace,
    token_cost_weight: float,
    token_sharp_weight: float,
    decoder_sharp_weight: float,
    manifold_weight: float,
) -> tuple[float, dict[str, float], dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    total_loss = 0.0
    total_acc = 0.0
    total_exact = 0.0
    last_rendered: dict[str, torch.Tensor] | None = None
    last_target: torch.Tensor | None = None
    last_mask: torch.Tensor | None = None
    last_lengths: torch.Tensor | None = None
    families = single_token_families(args)
    kmer_sizes = single_token_kmer_sizes(args)
    for _ in range(args.val_batches):
        if manifold_weight > 0:
            family_size = max(2, args.manifold_family_size)
            seed_count = args.manifold_seed_count if args.manifold_seed_count > 0 else max(1, args.batch_size // family_size)
            target, mask, lengths, bits, entropies, sequences, _kinds, family_ids = make_single_token_entropy_family_batch(
                seed_count=seed_count,
                family_size=family_size,
                rng=rng,
                device=device,
                substitution_rate=args.augment_substitution_rate,
                indel_rate=args.augment_indel_rate,
                repeat_rate=args.augment_repeat_rate,
                max_edits=args.augment_max_edits,
                **single_token_generation_kwargs(args, families=families, kmer_sizes=kmer_sizes),
            )
            loss, rendered = loss_for_family_geometry_batch(
                model=model,
                target=target,
                mask=mask,
                lengths=lengths,
                sequences=sequences,
                family_ids=family_ids,
                length_weight=args.length_weight,
                token_cost_weight=token_cost_weight,
                token_sharp_weight=token_sharp_weight,
                decoder_sharp_weight=decoder_sharp_weight,
                latent_l2_weight=args.latent_l2_weight,
                manifold_weight=manifold_weight,
                within_family_only=args.rapidfuzz_within_family_only,
                similarity_margin=args.rapidfuzz_similarity_margin,
                alignment_loss_weight=args.alignment_loss_weight,
                alignment_mode=args.alignment_mode,
                alignment_temperature=args.alignment_temperature,
                alignment_gap_cost=args.alignment_gap_cost,
                alignment_window=args.alignment_window,
                alignment_shift_cost=args.alignment_shift_cost,
                alignment_global_weight=args.alignment_global_weight,
                alignment_global_window=args.alignment_global_window,
            )
        else:
            target, mask, lengths, bits, entropies, _ = make_single_token_entropy_batch(
                batch_size=args.batch_size,
                rng=rng,
                device=device,
                **single_token_generation_kwargs(args, families=families, kmer_sizes=kmer_sizes),
            )
            loss, rendered = loss_for_batch(
                model=model,
                target=target,
                mask=mask,
                target_lengths=lengths,
                length_weight=args.length_weight,
                token_cost_weight=token_cost_weight,
                token_sharp_weight=token_sharp_weight,
                decoder_sharp_weight=decoder_sharp_weight,
                latent_l2_weight=args.latent_l2_weight,
                alignment_loss_weight=args.alignment_loss_weight,
                alignment_mode=args.alignment_mode,
                alignment_temperature=args.alignment_temperature,
                alignment_gap_cost=args.alignment_gap_cost,
                alignment_window=args.alignment_window,
                alignment_shift_cost=args.alignment_shift_cost,
                alignment_global_weight=args.alignment_global_weight,
                alignment_global_window=args.alignment_global_window,
            )
        rendered["info_bits"] = bits.detach()
        rendered["entropy_per_base"] = entropies.detach()
        metrics = reconstruction_metrics(rendered["soft_dna"], target, mask)
        total_loss += loss.item()
        total_acc += metrics["accuracy"]
        total_exact += metrics["exact"]
        last_rendered = rendered
        last_target = target
        last_mask = mask
        last_lengths = lengths
    assert last_rendered is not None and last_target is not None and last_mask is not None and last_lengths is not None
    return (
        total_loss / float(args.val_batches),
        {"accuracy": total_acc / float(args.val_batches), "exact": total_exact / float(args.val_batches)},
        last_rendered,
        last_target,
        last_mask,
        last_lengths,
    )


def controlled_probe(
    *,
    model: AdaptiveTokenizerAutoencoder,
    rng: random.Random,
    device: torch.device,
    args: argparse.Namespace,
) -> list[str]:
    if args.probe_examples <= 0:
        return []
    seq_lengths = parse_int_list(args.probe_seq_lengths)
    block_lengths = parse_int_list(args.probe_block_lengths)
    block_types = parse_str_list(args.probe_block_types)
    target, mask, lengths, labels = make_probe_batch(
        max_seq_len=args.seq_len,
        seq_lengths=[min(args.seq_len, value) for value in seq_lengths],
        block_types=block_types,
        block_lengths=block_lengths,
        examples_per_condition=args.probe_examples,
        rng=rng,
        device=device,
    )
    with torch.no_grad():
        rendered = model(target, mask)
    predicted = rendered["soft_dna"].argmax(dim=-1)
    correct = predicted == target
    per_acc = (correct.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    per_exact = (correct | (mask <= 0)).all(dim=1).float()
    token_count = rendered["token_usage"].sum(dim=1)
    active_emit = (rendered["lengths"] > 1.0).float().sum(dim=1)

    label_to_indices: dict[tuple[str, int, int], list[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    if not args.verbose_probe:
        lines = [
            "probe summary (L=target length; tok=latent tokens; base=base accuracy; exact=whole-seq exact; out=rendered length):"
        ]
        for seq_len in seq_lengths:
            clipped_seq_len = min(args.seq_len, seq_len)
            parts = []
            for block_type in block_types:
                indices = [
                    index
                    for key, key_indices in label_to_indices.items()
                    if key[0] == block_type and key[1] == clipped_seq_len
                    for index in key_indices
                ]
                short_name = {
                    "random": "rand",
                    "gc": "gc",
                    "at": "at",
                    "pyrimidine": "pyr",
                    "purine": "pur",
                    "homopolymer": "hpoly",
                    "dinucleotide": "di",
                    "motif": "motif",
                }.get(block_type, block_type)
                parts.append(
                    f"{short_name} tok {fmt_mean(token_count, indices, device)} "
                    f"base {fmt_mean(per_acc, indices, device, 3)} "
                    f"exact {fmt_mean(per_exact, indices, device, 3)} "
                    f"out {fmt_mean(rendered['total_len'], indices, device, 1)}"
                )
            lines.append(f"probe L{clipped_seq_len:03d} | " + " | ".join(parts))
        return lines

    lines = ["probe detail:"]
    for block_type in block_types:
        for seq_len in seq_lengths:
            parts = []
            for block_len in block_lengths:
                key = (block_type, min(args.seq_len, seq_len), block_len)
                indices = label_to_indices.get(key, [])
                if not indices:
                    continue
                index_tensor = torch.tensor(indices, device=device)
                first_index = indices[0]
                token_probs = rendered["token_prob"][first_index, : min(args.seq_len, seq_len)].detach().cpu()
                lengths_text = ",".join(f"{value:.1f}" for value in rendered["lengths"][first_index].detach().cpu().tolist())
                top_token_prob = token_probs.topk(min(8, token_probs.numel())).values.mean().item()
                parts.append(
                    f"b{block_len:02d} acc{per_acc[index_tensor].mean().item():.2f} "
                    f"ex{per_exact[index_tensor].mean().item():.2f} "
                    f"out{rendered['total_len'][index_tensor].mean().item():.1f} "
                    f"tok{token_count[index_tensor].mean().item():.1f} "
                    f"act{active_emit[index_tensor].mean().item():.1f} "
                    f"top_p{top_token_prob:.2f} "
                    f"lens[{lengths_text}]"
                )
            if parts:
                lines.append(f"probe {block_type:<12} L{min(args.seq_len, seq_len):03d} " + " | ".join(parts))
    return lines


def run_diagnostics(model: AdaptiveTokenizerAutoencoder, rng: random.Random, device: torch.device, args: argparse.Namespace) -> None:
    model.eval()
    with torch.no_grad():
        target, mask, lengths, bits, entropies, kinds = make_single_token_entropy_batch(
            batch_size=1,
            rng=rng,
            device=device,
            **single_token_generation_kwargs(args),
        )
        rendered = model(target, mask)
        seq_len = int(lengths[0].item())
        decoded = rendered["soft_dna"].argmax(dim=-1)[0, :seq_len]
        print("\nDiagnostics:")
        print(
            "kind:",
            kinds[0],
            "length:",
            seq_len,
            "bits:",
            f"{bits[0].item():.1f}",
            "entropy/base:",
            f"{entropies[0].item():.2f}",
        )
        print("target:       ", tensor_to_sequence(target[0, :seq_len]))
        print("reconstructed:", tensor_to_sequence(decoded))
        print("token count:", f"{rendered['token_usage'].sum(dim=1)[0].item():.2f}")
        print("token probs:", ", ".join(f"{value:.2f}" for value in rendered["token_prob"][0, :seq_len].detach().cpu().tolist()[:80]))
        print("emit lengths:", ", ".join(f"{value:.1f}" for value in rendered["lengths"][0].detach().cpu().tolist()))
    model.train()


def build_model(args: argparse.Namespace) -> AdaptiveTokenizerAutoencoder:
    return AdaptiveTokenizerAutoencoder(
        max_seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        latent_dim=args.latent_dim,
        max_slots_per_token=args.max_slots_per_token,
        hidden_dim=args.encoder_hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_hidden_dim=args.decoder_hidden_dim,
        slot_dim=args.slot_dim,
        token_rank_temperature=args.token_rank_temperature,
        token_usage_temperature=args.token_usage_temperature,
        gate_temperature=args.gate_temperature,
        pack_temperature=args.pack_temperature,
        initial_token_stride=args.initial_token_stride,
    )


def train(args: argparse.Namespace) -> None:
    if args.max_tokens * args.max_slots_per_token < args.seq_len:
        raise ValueError("--max-tokens * --max-slots-per-token must be at least --seq-len.")
    if args.max_tokens != 1:
        raise ValueError("This stage-1 curriculum script is intentionally single-token only; set --max-tokens 1.")
    if args.max_slots_per_token < args.seq_len:
        raise ValueError("For single-token training, --max-slots-per-token must be at least --seq-len.")
    if args.single_token_min_len <= 0 or args.single_token_min_len > args.seq_len:
        raise ValueError("--single-token-min-len must be in [1, seq_len].")
    if args.single_token_bit_budget <= 0:
        raise ValueError("--single-token-bit-budget must be positive.")
    if args.single_token_kmer_size is not None and args.single_token_kmer_size <= 0:
        raise ValueError("--single-token-kmer-size must be positive.")
    if not 0.0 <= args.single_token_entropy_percentile <= 1.0:
        raise ValueError("--single-token-entropy-percentile must be in [0, 1].")
    if args.program_max_motifs <= 0 or args.program_max_motif_len <= 0:
        raise ValueError("--program-max-motifs and --program-max-motif-len must be positive.")
    if args.program_repeat_lambda < 0 or args.program_long_repeat_lambda < 0:
        raise ValueError("--program-repeat-lambda values must be non-negative.")
    if not 0.0 <= args.program_single_repeat_prob <= 1.0:
        raise ValueError("--program-single-repeat-prob must be in [0, 1].")
    if not 0.0 <= args.program_long_repeat_prob <= 1.0:
        raise ValueError("--program-long-repeat-prob must be in [0, 1].")
    if args.program_single_repeat_prob + args.program_long_repeat_prob > 1.0:
        raise ValueError("program single-repeat and long-repeat probabilities must sum to <= 1.")
    if args.alignment_loss_weight < 0:
        raise ValueError("--alignment-loss-weight must be non-negative.")
    if args.alignment_temperature <= 0:
        raise ValueError("--alignment-temperature must be positive.")
    if args.alignment_gap_cost < 0:
        raise ValueError("--alignment-gap-cost must be non-negative.")
    if args.alignment_window < 0:
        raise ValueError("--alignment-window must be non-negative.")
    if args.alignment_shift_cost < 0:
        raise ValueError("--alignment-shift-cost must be non-negative.")
    if args.alignment_global_weight < 0:
        raise ValueError("--alignment-global-weight must be non-negative.")
    if args.alignment_global_window < -1:
        raise ValueError("--alignment-global-window must be -1 or non-negative.")
    if args.batch_size <= 0 or args.val_batches <= 0:
        raise ValueError("--batch-size and --val-batches must be positive.")
    families = single_token_families(args)
    kmer_sizes = single_token_kmer_sizes(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device(args)
    train_rng = random.Random(args.seed)
    val_rng = random.Random(args.seed + 1_000_000)
    probe_rng = random.Random(args.seed + 2_000_000)
    diagnostic_rng = random.Random(args.seed + 3_000_000)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")

    print(f"device: {device}")
    print(
        f"single-token entropy curriculum; seq_len={args.seq_len}; max_tokens={args.max_tokens}; "
        f"latent_dim={args.latent_dim}; params={sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(
        f"token_cost={args.token_cost_weight}; token_sharp={args.token_sharp_weight}; "
        f"length_weight={args.length_weight}; initial_stride={args.initial_token_stride}; "
        f"mode={args.curriculum_mode}; program_motifs<= {args.program_max_motifs}; "
        f"program_motif_len<= {args.program_max_motif_len}; "
        f"bit_budget={args.single_token_bit_budget}; kmers={','.join(str(k) for k in kmer_sizes)}; "
        f"entropy_percentile={args.single_token_entropy_percentile}; "
        f"families={','.join(families)}; manifold_weight={args.manifold_weight}; "
        f"manifold_family_size={max(2, args.manifold_family_size)}; manifold_warmup={args.manifold_warmup_frac}; "
        f"alignment_weight={args.alignment_loss_weight}; alignment_mode={args.alignment_mode}; "
        f"alignment_window={args.alignment_window}; alignment_global_weight={args.alignment_global_weight}; "
        f"alignment_gap={args.alignment_gap_cost}"
    )

    for step in range(args.steps):
        token_cost_weight = current_weight(step, args.steps, args.token_cost_weight, args.token_cost_warmup_frac)
        token_sharp_weight = current_weight(step, args.steps, args.token_sharp_weight, args.token_sharp_warmup_frac)
        decoder_sharp_weight = current_weight(step, args.steps, args.decoder_sharp_weight, args.decoder_sharp_warmup_frac)
        manifold_weight = current_weight(step, args.steps, args.manifold_weight, args.manifold_warmup_frac)
        if manifold_weight > 0:
            family_size = max(2, args.manifold_family_size)
            seed_count = args.manifold_seed_count if args.manifold_seed_count > 0 else max(1, args.batch_size // family_size)
            target, mask, lengths, bits, entropies, sequences, _kinds, family_ids = make_single_token_entropy_family_batch(
                seed_count=seed_count,
                family_size=family_size,
                rng=train_rng,
                device=device,
                substitution_rate=args.augment_substitution_rate,
                indel_rate=args.augment_indel_rate,
                repeat_rate=args.augment_repeat_rate,
                max_edits=args.augment_max_edits,
                **single_token_generation_kwargs(args, families=families, kmer_sizes=kmer_sizes),
            )
            loss, rendered = loss_for_family_geometry_batch(
                model=model,
                target=target,
                mask=mask,
                lengths=lengths,
                sequences=sequences,
                family_ids=family_ids,
                length_weight=args.length_weight,
                token_cost_weight=token_cost_weight,
                token_sharp_weight=token_sharp_weight,
                decoder_sharp_weight=decoder_sharp_weight,
                latent_l2_weight=args.latent_l2_weight,
                manifold_weight=manifold_weight,
                within_family_only=args.rapidfuzz_within_family_only,
                similarity_margin=args.rapidfuzz_similarity_margin,
                alignment_loss_weight=args.alignment_loss_weight,
                alignment_mode=args.alignment_mode,
                alignment_temperature=args.alignment_temperature,
                alignment_gap_cost=args.alignment_gap_cost,
                alignment_window=args.alignment_window,
                alignment_shift_cost=args.alignment_shift_cost,
                alignment_global_weight=args.alignment_global_weight,
                alignment_global_window=args.alignment_global_window,
            )
        else:
            target, mask, lengths, bits, entropies, _ = make_single_token_entropy_batch(
                batch_size=args.batch_size,
                rng=train_rng,
                device=device,
                **single_token_generation_kwargs(args, families=families, kmer_sizes=kmer_sizes),
            )
            loss, rendered = loss_for_batch(
                model=model,
                target=target,
                mask=mask,
                target_lengths=lengths,
                length_weight=args.length_weight,
                token_cost_weight=token_cost_weight,
                token_sharp_weight=token_sharp_weight,
                decoder_sharp_weight=decoder_sharp_weight,
                latent_l2_weight=args.latent_l2_weight,
                alignment_loss_weight=args.alignment_loss_weight,
                alignment_mode=args.alignment_mode,
                alignment_temperature=args.alignment_temperature,
                alignment_gap_cost=args.alignment_gap_cost,
                alignment_window=args.alignment_window,
                alignment_shift_cost=args.alignment_shift_cost,
                alignment_global_weight=args.alignment_global_weight,
                alignment_global_window=args.alignment_global_window,
            )
        rendered["info_bits"] = bits.detach()
        rendered["entropy_per_base"] = entropies.detach()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_rendered, val_target, val_mask, val_lengths = evaluate(
                    model=model,
                    rng=val_rng,
                    device=device,
                    args=args,
                    token_cost_weight=token_cost_weight,
                    token_sharp_weight=token_sharp_weight,
                    decoder_sharp_weight=decoder_sharp_weight,
                    manifold_weight=manifold_weight,
                )
                probe_lines = controlled_probe(model=model, rng=probe_rng, device=device, args=args)
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_payload = {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "step": step,
                    "validation_loss": best_val_loss,
                }
                torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
                if args.save_all_bests:
                    torch.save(checkpoint_payload, checkpoint_dir / f"best_step_{step:09d}.pt")

            for line in compact_status(
                step=step,
                token_cost_weight=token_cost_weight,
                token_sharp_weight=token_sharp_weight,
                decoder_sharp_weight=decoder_sharp_weight,
                manifold_weight=manifold_weight,
                train_loss=loss,
                train_rendered=rendered,
                train_target=target,
                train_mask=mask,
                train_lengths=lengths,
                val_loss=val_loss,
                val_rendered=val_rendered,
                val_target=val_target,
                val_mask=val_mask,
                val_lengths=val_lengths,
            ):
                print(line)
            if args.verbose_metrics:
                print(format_metrics("train", loss, rendered, target, mask))
                print(format_metrics("val", torch.tensor(val_loss), val_rendered, val_target, val_mask))
                print(summarise_batch(val_rendered, val_lengths))
            for line in probe_lines:
                print(line)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "step": args.steps - 1,
            "validation_loss": best_val_loss,
        },
        checkpoint_dir / "latest.pt",
    )
    run_diagnostics(model, diagnostic_rng, device, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-1 single-token DNA SoftPack autoencoder with entropy-budgeted sequence curriculum."
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--variable-min-len", type=int, default=20)
    parser.add_argument("--short-max-len-frac", type=float, default=0.4)
    parser.add_argument("--long-min-len-frac", type=float, default=0.75)
    parser.add_argument("--single-token-min-len", type=int, default=4)
    parser.add_argument("--single-token-bit-budget", type=float, default=32.0)
    parser.add_argument("--single-token-kmer-size", type=int, default=None)
    parser.add_argument("--single-token-kmer-sizes", default="1,2,3,4,5")
    parser.add_argument("--single-token-entropy-percentile", type=float, default=0.1)
    parser.add_argument("--single-token-max-tries", type=int, default=200)
    parser.add_argument("--single-token-families", default="random,gc,at,homopolymer,dinucleotide,motif")
    parser.add_argument("--curriculum-mode", choices=("program", "entropy"), default="program")
    parser.add_argument("--program-max-motifs", type=int, default=3)
    parser.add_argument("--program-max-motif-len", type=int, default=5)
    parser.add_argument("--program-repeat-lambda", type=float, default=0.8)
    parser.add_argument("--program-long-repeat-lambda", type=float, default=12.0)
    parser.add_argument("--program-long-repeat-prob", type=float, default=0.15)
    parser.add_argument("--program-single-repeat-prob", type=float, default=0.55)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--max-slots-per-token", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--token-cost-weight", type=float, default=0.0)
    parser.add_argument("--token-cost-warmup-frac", type=float, default=0.0)
    parser.add_argument("--token-sharp-weight", type=float, default=0.001)
    parser.add_argument("--token-sharp-warmup-frac", type=float, default=0.1)
    parser.add_argument("--decoder-sharp-weight", type=float, default=0.001)
    parser.add_argument("--decoder-sharp-warmup-frac", type=float, default=0.1)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)
    parser.add_argument("--alignment-loss-weight", type=float, default=1.0)
    parser.add_argument("--alignment-mode", choices=("local_window", "dp", "none"), default="local_window")
    parser.add_argument("--alignment-temperature", type=float, default=0.1)
    parser.add_argument("--alignment-gap-cost", type=float, default=0.75)
    parser.add_argument("--alignment-window", type=int, default=3)
    parser.add_argument("--alignment-shift-cost", type=float, default=0.02)
    parser.add_argument("--alignment-global-weight", type=float, default=0.5)
    parser.add_argument("--alignment-global-window", type=int, default=-1)
    parser.add_argument("--manifold-weight", type=float, default=0.05)
    parser.add_argument("--manifold-warmup-frac", type=float, default=0.0)
    parser.add_argument("--manifold-position-weight", type=float, default=0.05)
    parser.add_argument("--manifold-temperature", type=float, default=0.08)
    parser.add_argument("--manifold-seed-count", type=int, default=0)
    parser.add_argument("--manifold-family-size", type=int, default=4)
    parser.add_argument("--rapidfuzz-similarity-margin", type=float, default=0.0)
    parser.add_argument("--rapidfuzz-within-family-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment-substitution-rate", type=float, default=0.02)
    parser.add_argument("--augment-indel-rate", type=float, default=0.01)
    parser.add_argument("--augment-repeat-rate", type=float, default=0.25)
    parser.add_argument("--augment-max-edits", type=int, default=4)
    parser.add_argument("--token-rank-temperature", type=float, default=0.35)
    parser.add_argument("--token-usage-temperature", type=float, default=0.5)
    parser.add_argument("--gate-temperature", type=float, default=0.2)
    parser.add_argument("--pack-temperature", type=float, default=0.1)
    parser.add_argument("--initial-token-stride", type=float, default=1.0)
    parser.add_argument("--encoder-hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-hidden-dim", type=int, default=96)
    parser.add_argument("--slot-dim", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--probe-examples", type=int, default=0)
    parser.add_argument("--probe-seq-lengths", default="8,16,32,48,64")
    parser.add_argument("--probe-block-lengths", default="4,16,48")
    parser.add_argument("--probe-block-types", default="random,gc,at,homopolymer,dinucleotide,motif")
    parser.add_argument("--verbose-probe", action="store_true")
    parser.add_argument("--verbose-metrics", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/dna_single_token_entropy_curriculum")
    parser.add_argument("--save-all-bests", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
