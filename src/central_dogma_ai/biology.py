"""Basic DNA, codon, and amino-acid utilities."""

from __future__ import annotations

from collections.abc import Iterable
import random

DNA_BASES = ("A", "C", "G", "T")
DNA_TO_INDEX = {base: index for index, base in enumerate(DNA_BASES)}

AMINO_ACIDS = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "*",
)
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}
CODON_VOCAB = tuple(
    first_base + second_base + third_base
    for first_base in DNA_BASES
    for second_base in DNA_BASES
    for third_base in DNA_BASES
)
CODON_TO_INDEX = {codon: index for index, codon in enumerate(CODON_VOCAB)}
STOP_CODONS = tuple(codon for codon, aa in CODON_TABLE.items() if aa == "*")
NON_STOP_CODONS = tuple(codon for codon, aa in CODON_TABLE.items() if aa != "*")
CODONS_BY_AA: dict[str, tuple[str, ...]] = {
    amino_acid: tuple(codon for codon, aa in CODON_TABLE.items() if aa == amino_acid)
    for amino_acid in AMINO_ACIDS
}


def clean_dna(sequence: str) -> str:
    """Normalize a DNA string and reject ambiguous bases."""

    normalized = sequence.upper().replace("U", "T").replace(" ", "").replace("\n", "")
    invalid = sorted(set(normalized) - set(DNA_BASES))
    if invalid:
        raise ValueError(f"Invalid DNA bases: {''.join(invalid)}")
    return normalized


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""

    sequence = clean_dna(sequence)
    complement = str.maketrans("ACGT", "TGCA")
    return sequence.translate(complement)[::-1]


def chunk_codons(sequence: str, start_offset: int = 0, keep_partial: bool = False) -> tuple[str, ...]:
    """Split a sequence into codons from a reading-frame offset."""

    sequence = clean_dna(sequence)
    if start_offset < 0 or start_offset > 2:
        raise ValueError("start_offset must be 0, 1, or 2")
    framed = sequence[start_offset:]
    complete_length = len(framed) - (len(framed) % 3)
    codons = [framed[index : index + 3] for index in range(0, complete_length, 3)]
    if keep_partial and complete_length < len(framed):
        codons.append(framed[complete_length:])
    return tuple(codons)


def translate_codon(codon: str) -> str:
    """Translate one complete DNA codon."""

    codon = clean_dna(codon)
    if len(codon) != 3:
        raise ValueError(f"Codon must be three bases, got {codon!r}")
    return CODON_TABLE[codon]


def translate_codons(codons: Iterable[str]) -> str:
    """Translate complete codons into an amino-acid string."""

    return "".join(translate_codon(codon) for codon in codons)


def translate_sequence(sequence: str, start_offset: int = 0) -> str:
    """Translate all complete codons in a sequence."""

    return translate_codons(chunk_codons(sequence, start_offset=start_offset))


def one_hot_dna(sequence: str) -> list[list[float]]:
    """Encode DNA as an L x 4 one-hot matrix in A, C, G, T order."""

    sequence = clean_dna(sequence)
    rows: list[list[float]] = []
    for base in sequence:
        row = [0.0] * len(DNA_BASES)
        row[DNA_TO_INDEX[base]] = 1.0
        rows.append(row)
    return rows


def one_hot_amino_acids(protein: str) -> list[list[float]]:
    """Encode amino acids as an N x 21 one-hot matrix, including stop."""

    rows: list[list[float]] = []
    for amino_acid in protein:
        if amino_acid not in AA_TO_INDEX:
            raise ValueError(f"Unsupported amino acid: {amino_acid!r}")
        row = [0.0] * len(AMINO_ACIDS)
        row[AA_TO_INDEX[amino_acid]] = 1.0
        rows.append(row)
    return rows


def decode_one_hot_dna(rows: Iterable[Iterable[float]]) -> str:
    """Decode an A/C/G/T one-hot or probability matrix by argmax."""

    sequence = []
    for row in rows:
        values = list(row)
        if len(values) != len(DNA_BASES):
            raise ValueError("Each DNA row must have four values")
        sequence.append(DNA_BASES[max(range(len(values)), key=values.__getitem__)])
    return "".join(sequence)


def random_coding_sequence(amino_acid_codons: int, rng: random.Random | None = None) -> str:
    """Create a random CDS that starts with ATG and ends with a stop codon.

    `amino_acid_codons` counts translated amino-acid codons and excludes the
    terminal stop. Values greater than zero include an ATG start codon.
    """

    if amino_acid_codons < 1:
        raise ValueError("amino_acid_codons must be at least 1")
    rng = rng or random.Random()
    codons = ["ATG"]
    codons.extend(rng.choice(NON_STOP_CODONS) for _ in range(amino_acid_codons - 1))
    codons.append(rng.choice(STOP_CODONS))
    return "".join(codons)


def random_dna(length: int, rng: random.Random | None = None) -> str:
    """Create random DNA of a requested length."""

    if length < 0:
        raise ValueError("length must be non-negative")
    rng = rng or random.Random()
    return "".join(rng.choice(DNA_BASES) for _ in range(length))