"""Exact splice-aware transcript assembly and translation."""

from __future__ import annotations

from dataclasses import dataclass

from central_dogma_ai.biology import chunk_codons, clean_dna, translate_codons


@dataclass(frozen=True, slots=True)
class Exon:
    """A half-open genomic interval [start, end)."""

    name: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("Exon start must be non-negative")
        if self.end <= self.start:
            raise ValueError("Exon end must be greater than exon start")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Isoform:
    """An ordered exon path through a splice program."""

    name: str
    exon_names: tuple[str, ...]
    intended: bool = True

    def __post_init__(self) -> None:
        if not self.exon_names:
            raise ValueError("Isoform must include at least one exon")


@dataclass(frozen=True, slots=True)
class ExonPhase:
    """Reading-frame phase at the boundaries of one exon in an isoform."""

    exon_name: str
    phase_before: int
    phase_after: int
    transcript_start: int
    transcript_end: int


@dataclass(frozen=True, slots=True)
class SpliceProgram:
    """Genomic DNA plus a typed splice graph."""

    dna: str
    exons: tuple[Exon, ...]
    isoforms: tuple[Isoform, ...]
    name: str = "program"

    def __post_init__(self) -> None:
        clean = clean_dna(self.dna)
        object.__setattr__(self, "dna", clean)
        exon_names = [exon.name for exon in self.exons]
        if len(exon_names) != len(set(exon_names)):
            raise ValueError("Exon names must be unique")
        for exon in self.exons:
            if exon.end > len(clean):
                raise ValueError(f"Exon {exon.name!r} extends beyond the DNA sequence")
        known = set(exon_names)
        for isoform in self.isoforms:
            missing = set(isoform.exon_names) - known
            if missing:
                raise ValueError(f"Isoform {isoform.name!r} references unknown exons: {missing}")

    @property
    def exon_by_name(self) -> dict[str, Exon]:
        return {exon.name: exon for exon in self.exons}

    @property
    def isoform_by_name(self) -> dict[str, Isoform]:
        return {isoform.name: isoform for isoform in self.isoforms}


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Translation and frame annotations for one isoform."""

    isoform_name: str
    transcript_sequence: str
    codons: tuple[str, ...]
    protein: str
    frame_valid: bool
    exon_phases: tuple[ExonPhase, ...]
    junction_offsets: tuple[int, ...]
    first_stop_codon_index: int | None
    has_terminal_stop: bool
    has_premature_stop: bool
    nmd_risk: bool
    trailing_bases: str

    @property
    def coding_length(self) -> int:
        return len(self.codons) * 3


def assemble_transcript(program: SpliceProgram, isoform: Isoform | str) -> str:
    """Concatenate the exonic DNA for one isoform."""

    if isinstance(isoform, str):
        isoform = program.isoform_by_name[isoform]
    exons = program.exon_by_name
    return "".join(program.dna[exons[name].start : exons[name].end] for name in isoform.exon_names)


def exon_mask(program: SpliceProgram, isoform: Isoform | str) -> list[int]:
    """Return a genomic-position mask for bases included in one isoform."""

    if isinstance(isoform, str):
        isoform = program.isoform_by_name[isoform]
    mask = [0] * len(program.dna)
    exons = program.exon_by_name
    for name in isoform.exon_names:
        exon = exons[name]
        for index in range(exon.start, exon.end):
            mask[index] = 1
    return mask


def transcript_genomic_indices(program: SpliceProgram, isoform: Isoform | str) -> tuple[int, ...]:
    """Return genomic positions in the order they appear in the transcript."""

    if isinstance(isoform, str):
        isoform = program.isoform_by_name[isoform]
    exons = program.exon_by_name
    indices: list[int] = []
    for name in isoform.exon_names:
        exon = exons[name]
        indices.extend(range(exon.start, exon.end))
    return tuple(indices)


def splice_feature_tracks(program: SpliceProgram, isoform: Isoform | str) -> list[list[float]]:
    """Return L x 3 tracks for included bases, acceptor markers, and donor markers."""

    if isinstance(isoform, str):
        isoform = program.isoform_by_name[isoform]
    tracks = [[0.0, 0.0, 0.0] for _ in program.dna]
    exons = program.exon_by_name
    for exon_index, exon_name in enumerate(isoform.exon_names):
        exon = exons[exon_name]
        for genomic_index in range(exon.start, exon.end):
            tracks[genomic_index][0] = 1.0
        if exon_index > 0:
            tracks[exon.start][1] = 1.0
        if exon_index < len(isoform.exon_names) - 1:
            tracks[exon.end - 1][2] = 1.0
    return tracks


def exon_phases(program: SpliceProgram, isoform: Isoform | str, start_offset: int = 0) -> tuple[ExonPhase, ...]:
    """Compute frame phase before and after each exon in a path."""

    if start_offset < 0 or start_offset > 2:
        raise ValueError("start_offset must be 0, 1, or 2")
    if isinstance(isoform, str):
        isoform = program.isoform_by_name[isoform]
    exons = program.exon_by_name
    phases: list[ExonPhase] = []
    transcript_cursor = 0
    phase = start_offset % 3
    for name in isoform.exon_names:
        exon = exons[name]
        phase_before = phase
        transcript_start = transcript_cursor
        transcript_cursor += exon.length
        phase = (phase + exon.length) % 3
        phases.append(
            ExonPhase(
                exon_name=name,
                phase_before=phase_before,
                phase_after=phase,
                transcript_start=transcript_start,
                transcript_end=transcript_cursor,
            )
        )
    return tuple(phases)


def junction_offsets_for_isoform(program: SpliceProgram, isoform: Isoform | str) -> tuple[int, ...]:
    """Return transcript offsets immediately after each internal exon."""

    if isinstance(isoform, str):
        isoform = program.isoform_by_name[isoform]
    exons = program.exon_by_name
    offsets: list[int] = []
    cursor = 0
    for name in isoform.exon_names[:-1]:
        cursor += exons[name].length
        offsets.append(cursor)
    return tuple(offsets)


def translate_isoform(
    program: SpliceProgram,
    isoform: Isoform | str,
    start_offset: int = 0,
    nmd_distance_nt: int = 55,
) -> TranslationResult:
    """Assemble and translate one isoform, preserving frame diagnostics."""

    if isinstance(isoform, str):
        isoform = program.isoform_by_name[isoform]
    transcript = assemble_transcript(program, isoform)
    codons = chunk_codons(transcript, start_offset=start_offset)
    protein = translate_codons(codons)
    coding_length = start_offset + len(codons) * 3
    trailing_bases = transcript[coding_length:]
    frame_valid = not trailing_bases
    stop_indices = [index for index, amino_acid in enumerate(protein) if amino_acid == "*"]
    first_stop = stop_indices[0] if stop_indices else None
    has_terminal_stop = bool(protein) and protein[-1] == "*"
    has_premature_stop = first_stop is not None and first_stop < len(codons) - 1
    junction_offsets = junction_offsets_for_isoform(program, isoform)
    stop_end_nt = ((first_stop + 1) * 3 + start_offset) if first_stop is not None else None
    last_junction = junction_offsets[-1] if junction_offsets else None
    nmd_risk = bool(
        has_premature_stop
        and stop_end_nt is not None
        and last_junction is not None
        and (last_junction - stop_end_nt) >= nmd_distance_nt
    )
    return TranslationResult(
        isoform_name=isoform.name,
        transcript_sequence=transcript,
        codons=codons,
        protein=protein,
        frame_valid=frame_valid,
        exon_phases=exon_phases(program, isoform, start_offset=start_offset),
        junction_offsets=junction_offsets,
        first_stop_codon_index=first_stop,
        has_terminal_stop=has_terminal_stop,
        has_premature_stop=has_premature_stop,
        nmd_risk=nmd_risk,
        trailing_bases=trailing_bases,
    )


def translate_all(program: SpliceProgram, start_offset: int = 0) -> dict[str, TranslationResult]:
    """Translate every isoform in a splice program."""

    return {
        isoform.name: translate_isoform(program, isoform, start_offset=start_offset)
        for isoform in program.isoforms
    }