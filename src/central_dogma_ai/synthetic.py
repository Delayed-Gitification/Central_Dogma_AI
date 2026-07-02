"""Synthetic splice-aware translation examples."""

from __future__ import annotations

from dataclasses import dataclass
import random

from central_dogma_ai.biology import one_hot_amino_acids, one_hot_dna, random_coding_sequence, random_dna
from central_dogma_ai.splicing import (
    Exon,
    Isoform,
    SpliceProgram,
    exon_mask,
    splice_feature_tracks,
    transcript_genomic_indices,
    translate_isoform,
)


@dataclass(frozen=True, slots=True)
class SyntheticGeneConfig:
    """Configuration for random spliced coding examples."""

    amino_acid_codons: int = 48
    min_exons: int = 2
    max_exons: int = 6
    min_intron_length: int = 12
    max_intron_length: int = 48
    include_skip_isoform: bool = True

    def __post_init__(self) -> None:
        if self.amino_acid_codons < 2:
            raise ValueError("amino_acid_codons must be at least 2")
        if self.min_exons < 1 or self.max_exons < self.min_exons:
            raise ValueError("Invalid exon count range")
        if self.min_intron_length < 0 or self.max_intron_length < self.min_intron_length:
            raise ValueError("Invalid intron length range")


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One supervised example for splice-aware translation."""

    program: SpliceProgram
    isoform_name: str
    dna_one_hot: list[list[float]]
    exon_mask: list[int]
    splice_tracks: list[list[float]]
    genomic_path: list[int]
    amino_acid_one_hot: list[list[float]]
    protein: str
    frame_valid: bool
    has_premature_stop: bool


def _choose_exon_lengths(total_length: int, exon_count: int, rng: random.Random) -> list[int]:
    if exon_count < 1:
        raise ValueError("exon_count must be positive")
    if total_length < exon_count:
        raise ValueError("total_length must be at least exon_count")
    cut_points = sorted(rng.sample(range(1, total_length), exon_count - 1))
    positions = [0, *cut_points, total_length]
    return [positions[index + 1] - positions[index] for index in range(exon_count)]


def make_random_splice_program(
    config: SyntheticGeneConfig | None = None,
    rng: random.Random | None = None,
    name: str = "synthetic",
) -> SpliceProgram:
    """Generate a random coding transcript split by introns.

    The productive isoform includes every coding exon. If enabled and possible,
    a second isoform skips one internal exon, creating a useful negative/control
    path that may frameshift or create an early stop.
    """

    config = config or SyntheticGeneConfig()
    rng = rng or random.Random()
    cds = random_coding_sequence(config.amino_acid_codons, rng=rng)
    max_exons = min(config.max_exons, len(cds))
    exon_count = rng.randint(config.min_exons, max_exons)
    exon_lengths = _choose_exon_lengths(len(cds), exon_count, rng)

    dna_parts: list[str] = []
    exons: list[Exon] = []
    cds_cursor = 0
    genomic_cursor = 0
    for index, exon_length in enumerate(exon_lengths):
        exon_sequence = cds[cds_cursor : cds_cursor + exon_length]
        cds_cursor += exon_length
        start = genomic_cursor
        dna_parts.append(exon_sequence)
        genomic_cursor += exon_length
        end = genomic_cursor
        exons.append(Exon(name=f"E{index + 1}", start=start, end=end))
        if index < exon_count - 1:
            intron_length = rng.randint(config.min_intron_length, config.max_intron_length)
            intron = random_dna(intron_length, rng=rng)
            dna_parts.append(intron)
            genomic_cursor += intron_length

    isoforms = [Isoform(name="productive", exon_names=tuple(exon.name for exon in exons), intended=True)]
    if config.include_skip_isoform and len(exons) >= 3:
        skipped_index = rng.randrange(1, len(exons) - 1)
        isoforms.append(
            Isoform(
                name=f"skip_{exons[skipped_index].name}",
                exon_names=tuple(exon.name for index, exon in enumerate(exons) if index != skipped_index),
                intended=False,
            )
        )
    return SpliceProgram(dna="".join(dna_parts), exons=tuple(exons), isoforms=tuple(isoforms), name=name)


def make_training_example(program: SpliceProgram, isoform_name: str = "productive") -> TrainingExample:
    """Convert one isoform into one-hot DNA/mask/protein supervision."""

    result = translate_isoform(program, isoform_name)
    return TrainingExample(
        program=program,
        isoform_name=isoform_name,
        dna_one_hot=one_hot_dna(program.dna),
        exon_mask=exon_mask(program, isoform_name),
        splice_tracks=splice_feature_tracks(program, isoform_name),
        genomic_path=list(transcript_genomic_indices(program, isoform_name)),
        amino_acid_one_hot=one_hot_amino_acids(result.protein),
        protein=result.protein,
        frame_valid=result.frame_valid,
        has_premature_stop=result.has_premature_stop,
    )


def generate_examples(
    count: int,
    config: SyntheticGeneConfig | None = None,
    seed: int | None = None,
    isoform_name: str = "productive",
) -> list[TrainingExample]:
    """Generate supervised examples with reproducible randomness."""

    if count < 0:
        raise ValueError("count must be non-negative")
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        program = make_random_splice_program(config=config, rng=rng, name=f"synthetic_{index}")
        examples.append(make_training_example(program, isoform_name=isoform_name))
    return examples