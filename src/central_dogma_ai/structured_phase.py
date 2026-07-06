"""Differentiable structured helper for translated reading-frame phase.

This first prototype is intentionally small and gene-by-gene.  It supports a
single or small set of mature transcript paths represented as genomic index
lists.  The structured layer sums over latent start/stop codon choices in
log-space and returns per-genomic-base marginals over translation states.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import nn
import torch.nn.functional as F

from central_dogma_ai.biology import DNA_BASES, DNA_TO_INDEX, NON_STOP_CODONS, STOP_CODONS, clean_dna


PHASE_STATES = ("N", "C0", "C1", "C2", "T")
N_STATE = 0
C0_STATE = 1
C1_STATE = 2
C2_STATE = 3
T_STATE = 4


def one_hot_dna_tensor(sequence: str) -> torch.Tensor:
    """Encode DNA as an L x 4 float tensor in A,C,G,T order."""

    sequence = clean_dna(sequence)
    encoded = torch.zeros(len(sequence), len(DNA_BASES), dtype=torch.float32)
    for index, base in enumerate(sequence):
        encoded[index, DNA_TO_INDEX[base]] = 1.0
    return encoded


def _random_dna_without_atg(length: int, rng: random.Random) -> str:
    """Generate a short UTR while avoiding ATG so the first ATG rule is clean."""

    bases = []
    while len(bases) < length:
        bases.append(rng.choice(DNA_BASES))
        if len(bases) >= 3 and "".join(bases[-3:]) == "ATG":
            bases.pop()
    return "".join(bases)


@dataclass(frozen=True)
class SplicePath:
    """One mature transcript path through genomic coordinates."""

    genomic_indices: torch.Tensor
    log_weight: torch.Tensor | None = None


@dataclass(frozen=True)
class SyntheticPhaseGene:
    """Synthetic positive-strand gene for the phase helper prototype."""

    dna: str
    dna_one_hot: torch.Tensor
    paths: tuple[SplicePath, ...]
    target_states: torch.Tensor
    start_codon_start: int
    stop_codon_start: int
    utr5_length: int
    utr3_length: int
    exon_lengths: tuple[int, ...] = ()
    intron_lengths: tuple[int, ...] = ()


@dataclass
class PhaseLayerOutput:
    state_log_probs: torch.Tensor
    state_probs: torch.Tensor
    initiation_log_probs: torch.Tensor
    termination_log_probs: torch.Tensor
    log_partition: torch.Tensor


def generate_single_exon_phase_gene(
    *,
    utr5_length: int = 12,
    coding_codons: int = 6,
    utr3_length: int = 10,
    seed: int = 1,
) -> SyntheticPhaseGene:
    """Generate a single-exon gene for test 1.

    `coding_codons` includes the ATG start codon and terminal stop codon.
    The 5' UTR is sampled without ATG, so the first ATG in the mature
    transcript is the true initiation codon.
    """

    if coding_codons < 2:
        raise ValueError("coding_codons must include at least ATG and stop")
    rng = random.Random(seed)
    utr5 = _random_dna_without_atg(utr5_length, rng)
    internal_codons = []
    internal_choices = tuple(codon for codon in NON_STOP_CODONS if codon != "ATG")
    for _ in range(coding_codons - 2):
        internal_codons.append(rng.choice(internal_choices))
    stop_codon = rng.choice(STOP_CODONS)
    cds = "ATG" + "".join(internal_codons) + stop_codon
    utr3 = "".join(rng.choice(DNA_BASES) for _ in range(utr3_length))
    dna = utr5 + cds + utr3

    start = utr5_length
    stop = utr5_length + len(cds) - 3
    target = torch.full((len(dna),), N_STATE, dtype=torch.long)
    for genomic_index in range(start, stop + 3):
        target[genomic_index] = C0_STATE + ((genomic_index - start) % 3)
    if stop + 3 < len(dna):
        target[stop + 3 :] = T_STATE

    path = SplicePath(genomic_indices=torch.arange(len(dna), dtype=torch.long))
    return SyntheticPhaseGene(
        dna=dna,
        dna_one_hot=one_hot_dna_tensor(dna),
        paths=(path,),
        target_states=target,
        start_codon_start=start,
        stop_codon_start=stop,
        utr5_length=utr5_length,
        utr3_length=utr3_length,
        exon_lengths=(len(dna),),
        intron_lengths=(),
    )


def _split_lengths(total_length: int, parts: int, *, min_part: int, rng: random.Random) -> tuple[int, ...]:
    if parts < 1:
        raise ValueError("parts must be at least 1")
    if total_length < parts * min_part:
        raise ValueError("total_length is too short for the requested split")
    remaining = total_length
    lengths = []
    for part in range(parts - 1):
        max_length = remaining - min_part * (parts - part - 1)
        length = rng.randint(min_part, max_length)
        lengths.append(length)
        remaining -= length
    lengths.append(remaining)
    return tuple(lengths)


def _build_phase_target_for_path(
    *,
    genome_length: int,
    path_indices: list[int],
    start_offset: int,
    stop_offset: int,
) -> torch.Tensor:
    target = torch.full((genome_length,), N_STATE, dtype=torch.long)
    for transcript_offset, genomic_index in enumerate(path_indices):
        if transcript_offset < start_offset:
            target[genomic_index] = N_STATE
        elif transcript_offset <= stop_offset + 2:
            target[genomic_index] = C0_STATE + ((transcript_offset - start_offset) % 3)
        else:
            target[genomic_index] = T_STATE
    return target


def generate_multiexon_phase_gene(
    *,
    utr5_length: int = 12,
    coding_codons: int = 8,
    utr3_length: int = 10,
    exon_count: int = 3,
    exon_lengths: tuple[int, ...] | None = None,
    min_exon_length: int = 4,
    min_intron_length: int = 8,
    max_intron_length: int = 20,
    seed: int = 1,
) -> SyntheticPhaseGene:
    """Generate a positive-strand multiexon gene with one mature transcript path.

    The start and stop codons are still chosen in the mature spliced transcript.
    Internal coding codons may cross exon junctions depending on `exon_lengths`.
    For now, callers should avoid splitting the start/stop codons themselves if
    they want the simple genomic-window motif extractor to recover them exactly.
    """

    if coding_codons < 2:
        raise ValueError("coding_codons must include at least ATG and stop")
    if min_intron_length < 4:
        raise ValueError("min_intron_length must allow GT...AG introns")
    if max_intron_length < min_intron_length:
        raise ValueError("max_intron_length must be >= min_intron_length")

    rng = random.Random(seed)
    utr5 = _random_dna_without_atg(utr5_length, rng)
    internal_choices = tuple(codon for codon in NON_STOP_CODONS if codon != "ATG")
    internal_codons = [rng.choice(internal_choices) for _ in range(coding_codons - 2)]
    cds = "ATG" + "".join(internal_codons) + rng.choice(STOP_CODONS)
    utr3 = "".join(rng.choice(DNA_BASES) for _ in range(utr3_length))
    transcript = utr5 + cds + utr3
    transcript_length = len(transcript)

    if exon_lengths is None:
        exon_lengths = _split_lengths(transcript_length, exon_count, min_part=min_exon_length, rng=rng)
    else:
        exon_lengths = tuple(exon_lengths)
        if sum(exon_lengths) != transcript_length:
            raise ValueError("exon_lengths must sum to the mature transcript length")
        exon_count = len(exon_lengths)
        if any(length < 1 for length in exon_lengths):
            raise ValueError("exon lengths must be positive")

    genome_parts: list[str] = []
    path_indices: list[int] = []
    intron_lengths = []
    transcript_cursor = 0
    for exon_index, exon_length in enumerate(exon_lengths):
        exon = transcript[transcript_cursor : transcript_cursor + exon_length]
        for base in exon:
            path_indices.append(len(genome_parts))
            genome_parts.append(base)
        transcript_cursor += exon_length
        if exon_index < exon_count - 1:
            intron_length = rng.randint(min_intron_length, max_intron_length)
            intron_lengths.append(intron_length)
            genome_parts.extend("GT" + "".join(rng.choice(DNA_BASES) for _ in range(intron_length - 4)) + "AG")

    dna = "".join(genome_parts)
    start_offset = utr5_length
    stop_offset = utr5_length + len(cds) - 3
    start_genomic = path_indices[start_offset]
    stop_genomic = path_indices[stop_offset]
    target = _build_phase_target_for_path(
        genome_length=len(dna),
        path_indices=path_indices,
        start_offset=start_offset,
        stop_offset=stop_offset,
    )

    path = SplicePath(genomic_indices=torch.tensor(path_indices, dtype=torch.long))
    return SyntheticPhaseGene(
        dna=dna,
        dna_one_hot=one_hot_dna_tensor(dna),
        paths=(path,),
        target_states=target,
        start_codon_start=start_genomic,
        stop_codon_start=stop_genomic,
        utr5_length=utr5_length,
        utr3_length=utr3_length,
        exon_lengths=exon_lengths,
        intron_lengths=tuple(intron_lengths),
    )


class DNACodonFeatureExtractor(nn.Module):
    """Small learnable DNA adapter that scores ATG and stop codon starts."""

    def __init__(self, *, motif_strength: float = 0.02):
        super().__init__()
        self.start_weight = nn.Parameter(torch.empty(3, 4))
        self.start_bias = nn.Parameter(torch.zeros(()))
        self.stop_weight = nn.Parameter(torch.empty(len(STOP_CODONS), 3, 4))
        self.stop_bias = nn.Parameter(torch.zeros(len(STOP_CODONS)))
        nn.init.normal_(self.start_weight, mean=0.0, std=motif_strength)
        nn.init.normal_(self.stop_weight, mean=0.0, std=motif_strength)

    def initialize_textbook_motifs(self, *, strength: float = 4.0, bias: float = -8.0) -> None:
        """Initialize parameters to sharply detect ATG and standard stops."""

        with torch.no_grad():
            self.start_weight.fill_(-strength)
            for pos, base in enumerate("ATG"):
                self.start_weight[pos, DNA_TO_INDEX[base]] = strength
            self.start_bias.fill_(bias)

            self.stop_weight.fill_(-strength)
            for motif_index, codon in enumerate(STOP_CODONS):
                for pos, base in enumerate(codon):
                    self.stop_weight[motif_index, pos, DNA_TO_INDEX[base]] = strength
            self.stop_bias.fill_(bias)

    def forward(self, dna_one_hot: torch.Tensor) -> dict[str, torch.Tensor]:
        if dna_one_hot.ndim != 2 or dna_one_hot.shape[-1] != 4:
            raise ValueError("dna_one_hot must have shape L x 4")
        length = dna_one_hot.shape[0]
        if length < 3:
            low = dna_one_hot.new_full((length,), -1.0e4)
            return {"start_logits": low, "stop_logits": low}

        windows = dna_one_hot.unfold(dimension=0, size=3, step=1).transpose(1, 2)
        start_valid = (windows * self.start_weight).sum(dim=(1, 2)) + self.start_bias
        stop_by_motif = (windows[:, None, :, :] * self.stop_weight[None, :, :, :]).sum(dim=(2, 3)) + self.stop_bias
        stop_valid = torch.logsumexp(stop_by_motif, dim=1)

        pad = dna_one_hot.new_full((2,), -1.0e4)
        return {
            "start_logits": torch.cat([start_valid, pad], dim=0),
            "stop_logits": torch.cat([stop_valid, pad], dim=0),
        }


class StructuredPhaseLayer(nn.Module):
    """Log-space latent start/stop phase layer over mature transcript paths."""

    def __init__(self, *, min_orf_codons: int = 2):
        super().__init__()
        if min_orf_codons < 2:
            raise ValueError("min_orf_codons must include at least start and stop codons")
        self.min_orf_codons = min_orf_codons

    @staticmethod
    def _empty_buckets(length: int, states: int) -> list[list[list[torch.Tensor]]]:
        return [[[] for _ in range(states)] for _ in range(length)]

    @staticmethod
    def _frame_prefix_logsum(values: torch.Tensor, max_codon_start: int) -> torch.Tensor:
        """prefix[frame, i] = logsumexp(values[j]) for j <= i and j % 3 == frame."""

        length = int(values.numel())
        offsets = torch.arange(length, device=values.device)
        prefix = values.new_full((3, length), -torch.inf)
        valid_codon_start = offsets <= max_codon_start
        for frame in range(3):
            framed = values.masked_fill((offsets % 3 != frame) | ~valid_codon_start, -torch.inf)
            prefix[frame] = torch.logcumsumexp(framed, dim=0)
        return prefix

    @staticmethod
    def _frame_suffix_logsum(values: torch.Tensor, max_codon_start: int) -> torch.Tensor:
        """suffix[frame, i] = logsumexp(values[j]) for j >= i and j % 3 == frame."""

        length = int(values.numel())
        offsets = torch.arange(length, device=values.device)
        suffix = values.new_full((3, length + 1), -torch.inf)
        valid_codon_start = offsets <= max_codon_start
        for frame in range(3):
            framed = values.masked_fill((offsets % 3 != frame) | ~valid_codon_start, -torch.inf)
            suffix[frame, :length] = torch.flip(torch.logcumsumexp(torch.flip(framed, dims=(0,)), dim=0), dims=(0,))
        return suffix

    @staticmethod
    def _append_if_finite(bucket: list[torch.Tensor], value: torch.Tensor) -> None:
        if bool(torch.isfinite(value).item()):
            bucket.append(value)

    def forward(
        self,
        *,
        start_logits: torch.Tensor,
        stop_logits: torch.Tensor,
        paths: tuple[SplicePath, ...],
    ) -> PhaseLayerOutput:
        if start_logits.shape != stop_logits.shape:
            raise ValueError("start_logits and stop_logits must have matching shape")
        genome_length = int(start_logits.shape[0])
        state_buckets = self._empty_buckets(genome_length, len(PHASE_STATES))
        initiation_buckets: list[list[torch.Tensor]] = [[] for _ in range(genome_length)]
        termination_buckets: list[list[torch.Tensor]] = [[] for _ in range(genome_length)]
        all_path_scores: list[torch.Tensor] = []

        for path in paths:
            indices = path.genomic_indices.to(device=start_logits.device, dtype=torch.long)
            transcript_length = int(indices.numel())
            if transcript_length < self.min_orf_codons * 3:
                continue
            path_weight = path.log_weight
            if path_weight is None:
                path_weight = start_logits.new_zeros(())
            else:
                path_weight = path_weight.to(device=start_logits.device, dtype=start_logits.dtype)

            max_codon_start = transcript_length - 3
            min_stop_delta = (self.min_orf_codons - 1) * 3
            offsets = torch.arange(transcript_length, device=start_logits.device)
            start_by_offset = start_logits[indices]
            stop_by_offset = stop_logits[indices]

            suffix_stop = self._frame_suffix_logsum(stop_by_offset, max_codon_start)
            prefix_start = self._frame_prefix_logsum(start_by_offset, max_codon_start)

            min_stop_by_start = offsets + min_stop_delta
            has_valid_stop = min_stop_by_start <= max_codon_start
            start_stop_logsum = start_by_offset + suffix_stop[offsets % 3, min_stop_by_start.clamp(max=transcript_length)]
            init_by_offset = (path_weight + start_stop_logsum).masked_fill(~has_valid_stop, -torch.inf)
            path_log_partition = torch.logsumexp(init_by_offset, dim=0)
            if not bool(torch.isfinite(path_log_partition).item()):
                continue
            all_path_scores.append(path_log_partition)

            term_by_offset = stop_by_offset.new_full((transcript_length,), -torch.inf)
            stop_valid = offsets <= max_codon_start
            start_limit_by_stop = offsets - min_stop_delta
            valid_stop_with_start = stop_valid & (start_limit_by_stop >= 0)
            term_by_offset[valid_stop_with_start] = (
                path_weight
                + stop_by_offset[valid_stop_with_start]
                + prefix_start[
                    offsets[valid_stop_with_start] % 3,
                    start_limit_by_stop[valid_stop_with_start],
                ]
            )

            suffix_init = torch.flip(torch.logcumsumexp(torch.flip(init_by_offset, dims=(0,)), dim=0), dims=(0,))
            prefix_term = torch.logcumsumexp(term_by_offset, dim=0)

            for transcript_offset, genomic_tensor in enumerate(indices):
                genomic_index = int(genomic_tensor.item())

                if transcript_offset + 1 < transcript_length:
                    self._append_if_finite(state_buckets[genomic_index][N_STATE], suffix_init[transcript_offset + 1])
                if transcript_offset >= 3:
                    self._append_if_finite(state_buckets[genomic_index][T_STATE], prefix_term[transcript_offset - 3])

                possible_starts = offsets[: transcript_offset + 1]
                if possible_starts.numel() == 0:
                    continue
                minimum_stop = torch.maximum(
                    possible_starts + min_stop_delta,
                    possible_starts.new_full(possible_starts.shape, transcript_offset - 2),
                ).clamp(min=0, max=transcript_length)
                coding_scores = (
                    path_weight
                    + start_by_offset[possible_starts]
                    + suffix_stop[possible_starts % 3, minimum_stop]
                )
                for phase in range(3):
                    phase_mask = (transcript_offset - possible_starts) % 3 == phase
                    if bool(phase_mask.any().item()):
                        phase_score = torch.logsumexp(coding_scores[phase_mask], dim=0)
                        self._append_if_finite(state_buckets[genomic_index][C0_STATE + phase], phase_score)

            for start_offset in range(transcript_length):
                start_genomic = int(indices[start_offset].item())
                self._append_if_finite(initiation_buckets[start_genomic], init_by_offset[start_offset])
            for stop_offset in range(transcript_length):
                stop_genomic = int(indices[stop_offset].item())
                self._append_if_finite(termination_buckets[stop_genomic], term_by_offset[stop_offset])

        if not all_path_scores:
            raise ValueError("No valid start/stop paths were available")
        log_partition = torch.logsumexp(torch.stack(all_path_scores), dim=0)
        state_log_probs = start_logits.new_full((genome_length, len(PHASE_STATES)), -torch.inf)
        initiation_log_probs = start_logits.new_full((genome_length,), -torch.inf)
        termination_log_probs = start_logits.new_full((genome_length,), -torch.inf)

        for genomic_index in range(genome_length):
            if not any(state_buckets[genomic_index]):
                state_log_probs[genomic_index, N_STATE] = start_logits.new_zeros(())
                continue
            for state in range(len(PHASE_STATES)):
                if state_buckets[genomic_index][state]:
                    state_log_probs[genomic_index, state] = (
                        torch.logsumexp(torch.stack(state_buckets[genomic_index][state]), dim=0) - log_partition
                    )
            if initiation_buckets[genomic_index]:
                initiation_log_probs[genomic_index] = (
                    torch.logsumexp(torch.stack(initiation_buckets[genomic_index]), dim=0) - log_partition
                )
            if termination_buckets[genomic_index]:
                termination_log_probs[genomic_index] = (
                    torch.logsumexp(torch.stack(termination_buckets[genomic_index]), dim=0) - log_partition
                )

        return PhaseLayerOutput(
            state_log_probs=state_log_probs,
            state_probs=state_log_probs.exp(),
            initiation_log_probs=initiation_log_probs,
            termination_log_probs=termination_log_probs,
            log_partition=log_partition,
        )


class StructuredTranslationPhaseModel(nn.Module):
    """DNA feature extractor plus structured phase helper layer."""

    def __init__(self, *, min_orf_codons: int = 2):
        super().__init__()
        self.feature_extractor = DNACodonFeatureExtractor()
        self.phase_layer = StructuredPhaseLayer(min_orf_codons=min_orf_codons)

    def forward(self, dna_one_hot: torch.Tensor, paths: tuple[SplicePath, ...]) -> PhaseLayerOutput:
        features = self.feature_extractor(dna_one_hot)
        return self.phase_layer(
            start_logits=features["start_logits"],
            stop_logits=features["stop_logits"],
            paths=paths,
        )


def phase_nll_loss(output: PhaseLayerOutput, target_states: torch.Tensor) -> torch.Tensor:
    """Per-base negative log likelihood for phase states."""

    return F.nll_loss(output.state_log_probs, target_states.to(output.state_log_probs.device))
