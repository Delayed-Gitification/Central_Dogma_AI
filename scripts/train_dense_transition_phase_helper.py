from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from central_dogma_ai.biology import CODONS_BY_AA, DNA_BASES, DNA_TO_INDEX, STOP_CODONS  # noqa: E402


STATE_NAMES = (
    "U0",
    "U1",
    "U2",
    "C00",
    "C01",
    "C02",
    "C10",
    "C11",
    "C12",
    "C20",
    "C21",
    "C22",
    "I00",
    "I01",
    "I02",
    "I10",
    "I11",
    "I12",
    "I20",
    "I21",
    "I22",
    "T0",
    "T1",
    "T2",
)
TRANSITION_TYPES = ("u5", "start", "coding", "donor", "intron", "acceptor", "stop", "u3")
STATE_TO_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}
TYPE_TO_INDEX = {name: index for index, name in enumerate(TRANSITION_TYPES)}
S = len(STATE_NAMES)
K = len(TRANSITION_TYPES)


def idx_U(g: int) -> int:
    return g


def idx_C(g: int, p: int) -> int:
    return 3 + g * 3 + p


def idx_I(g: int, p: int) -> int:
    return 12 + g * 3 + p


def idx_T(g: int) -> int:
    return 21 + g


def build_edge_type_matrix() -> torch.Tensor:
    edge = torch.full((S, S), -1, dtype=torch.long)

    def add(src: int, dst: int, transition_type: str) -> None:
        edge[src, dst] = TYPE_TO_INDEX[transition_type]

    for g in range(3):
        g_next = (g + 1) % 3
        add(idx_U(g), idx_U(g_next), "u5")
        add(idx_U(g), idx_C(g, 0), "start")
        add(idx_T(g), idx_T(g_next), "u3")
        for p in range(3):
            p_next = (p + 1) % 3
            add(idx_C(g, p), idx_C(g, p_next), "coding")
            add(idx_C(g, p), idx_I(g_next, p_next), "donor")
            add(idx_I(g, p), idx_I(g_next, p), "intron")
            add(idx_I(g, p), idx_C(g, p), "acceptor")
        add(idx_C(g, 2), idx_T(g), "stop")
    return edge


EDGE_TYPE_MATRIX = build_edge_type_matrix()


def edge_list_from_matrix(edge_type_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_from, edge_to = torch.nonzero(edge_type_matrix >= 0, as_tuple=True)
    edge_type = edge_type_matrix[edge_from, edge_to]
    return edge_from.to(torch.long), edge_to.to(torch.long), edge_type.to(torch.long)


@dataclass(frozen=True)
class DenseGene:
    dna: str
    dna_one_hot: torch.Tensor
    splice_tracks: torch.Tensor
    target_states: torch.Tensor
    evidence_targets: torch.Tensor
    start_codon_start: int
    stop_codon_start: int
    stop_transition_position: int
    donor_positions: tuple[int, ...]
    acceptor_positions: tuple[int, ...]
    utr5_length: int
    utr3_length: int
    exon_lengths: tuple[int, ...]
    intron_lengths: tuple[int, ...]


@dataclass(frozen=True)
class DenseLayerOutput:
    state_probs: torch.Tensor
    state_log_probs: torch.Tensor
    transition_type_posteriors: torch.Tensor
    start_posterior: torch.Tensor
    stop_posterior: torch.Tensor
    donor_posterior: torch.Tensor
    acceptor_posterior: torch.Tensor


@dataclass(frozen=True)
class DenseBatch:
    dna_one_hot: torch.Tensor
    splice_tracks: torch.Tensor
    evidence_targets: torch.Tensor
    target_states: torch.Tensor
    mask: torch.Tensor
    start_positions: torch.Tensor
    stop_transition_positions: torch.Tensor
    donor_mask: torch.Tensor
    acceptor_mask: torch.Tensor
    lengths: torch.Tensor
    intron_counts: torch.Tensor


def one_hot_dna(sequence: str) -> torch.Tensor:
    encoded = torch.zeros(len(sequence), 4, dtype=torch.float32)
    for index, base in enumerate(sequence):
        encoded[index, DNA_TO_INDEX[base]] = 1.0
    return encoded


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def random_protein(codons: int, rng: random.Random) -> str:
    amino_acids = tuple(aa for aa in CODONS_BY_AA if aa not in {"*", "M"})
    return "M" + "".join(rng.choice(amino_acids) for _ in range(codons - 1))


def reverse_translate_with_stop(protein: str, rng: random.Random) -> str:
    codons = []
    for index, amino_acid in enumerate(protein):
        codons.append("ATG" if index == 0 else rng.choice(CODONS_BY_AA[amino_acid]))
    codons.append(rng.choice(tuple(STOP_CODONS)))
    return "".join(codons)


def intron_length_with_mod(min_length: int, max_length: int, mode: str, rng: random.Random) -> int:
    choices = list(range(min_length, max_length + 1))
    if mode == "0":
        choices = [length for length in choices if length % 3 == 0]
    elif mode == "1":
        choices = [length for length in choices if length % 3 == 1]
    elif mode == "2":
        choices = [length for length in choices if length % 3 == 2]
    elif mode == "nonzero":
        choices = [length for length in choices if length % 3 != 0]
    elif mode != "any":
        raise ValueError(f"Unknown intron modulo mode: {mode}")
    if not choices:
        raise ValueError("No intron lengths available for requested modulo mode")
    return rng.choice(choices)


def sample_cds_cut_points(cds_length: int, intron_count: int, min_exon_length: int, rng: random.Random) -> tuple[int, ...]:
    if intron_count == 0:
        return ()
    possible = list(range(3, cds_length - 3 + 1))
    for _ in range(1000):
        cuts = tuple(sorted(rng.sample(possible, intron_count)))
        lengths = (cuts[0],) + tuple(right - left for left, right in zip(cuts, cuts[1:])) + (cds_length - cuts[-1],)
        if all(length >= min_exon_length for length in lengths):
            return cuts
    raise ValueError("Could not sample CDS cut points")


def generate_dense_gene(
    *,
    utr5_length: int = 100,
    coding_codons: int = 60,
    utr3_length: int = 120,
    exon_count: int = 4,
    min_exon_length: int = 6,
    min_intron_length: int = 50,
    max_intron_length: int = 300,
    intron_mod: str = "any",
    seed: int = 1,
) -> DenseGene:
    if coding_codons < 2:
        raise ValueError("coding_codons includes start and stop, so it must be at least 2")
    if exon_count < 1:
        raise ValueError("exon_count must be at least 1")
    if min_intron_length < 4:
        raise ValueError("min_intron_length must allow GT...AG introns")

    rng = random.Random(seed)
    utr5 = random_dna(utr5_length, rng)
    cds = reverse_translate_with_stop(random_protein(coding_codons - 1, rng), rng)
    utr3 = random_dna(utr3_length, rng)
    cuts = sample_cds_cut_points(len(cds), exon_count - 1, min_exon_length, rng)

    rows: list[tuple[str, str, int | None]] = []
    rows.extend((base, "U", None) for base in utr5)
    start_pos = len(rows)
    donor_positions: list[int] = []
    acceptor_positions: list[int] = []
    intron_lengths: list[int] = []
    exon_lengths: list[int] = [utr5_length]

    previous_cut = 0
    stop_pos = -1
    for cut_index, cut in enumerate(cuts + (len(cds),)):
        exon_start = len(rows)
        for cds_offset in range(previous_cut, cut):
            if cds_offset == len(cds) - 3:
                stop_pos = len(rows)
            rows.append((cds[cds_offset], "C", cds_offset % 3))
        exon_lengths[-1] += len(rows) - exon_start
        previous_cut = cut
        if cut_index < len(cuts):
            donor_positions.append(len(rows) - 1)
            intron_length = intron_length_with_mod(min_intron_length, max_intron_length, intron_mod, rng)
            intron_lengths.append(intron_length)
            carried_phase = cut % 3
            intron = "GT" + random_dna(intron_length - 4, rng) + "AG"
            rows.extend((base, "I", carried_phase) for base in intron)
            acceptor_positions.append(len(rows))
            exon_lengths.append(0)

    rows.extend((base, "T", None) for base in utr3)
    exon_lengths[-1] += utr3_length

    dna = "".join(base for base, _region, _phase in rows)
    target = torch.empty(len(rows), dtype=torch.long)
    splice_tracks = torch.zeros(len(rows), 2, dtype=torch.float32)
    evidence_targets = torch.zeros(len(rows), K, dtype=torch.float32)
    transition_donor_positions = tuple(position + 1 for position in donor_positions)
    donor_set = set(transition_donor_positions)
    acceptor_set = set(acceptor_positions)
    transition_types = ["u5" for _ in rows]

    for pos, (base, region, phase) in enumerate(rows):
        if region == "U":
            transition_types[pos] = "u5"
        elif region == "C":
            if phase is None:
                raise AssertionError("coding rows need mature CDS phase")
            transition_types[pos] = "coding"
        elif region == "I":
            if phase is None:
                raise AssertionError("intron rows need carried mature CDS phase")
            transition_types[pos] = "intron"
        elif region == "T":
            transition_types[pos] = "u3"
        else:
            raise AssertionError(f"unknown region {region}")
    transition_types[start_pos] = "start"
    for pos in transition_donor_positions:
        transition_types[pos] = "donor"
    for pos in acceptor_positions:
        transition_types[pos] = "acceptor"
    for pos in transition_donor_positions:
        splice_tracks[pos, 0] = 1.0
    for pos in acceptor_positions:
        splice_tracks[pos, 1] = 1.0

    current_state = idx_U(0)
    stop_transition_pos = None
    stop_type_index = TYPE_TO_INDEX["stop"]
    for pos, transition_type in enumerate(transition_types):
        if pos >= stop_pos and bool((EDGE_TYPE_MATRIX[current_state] == stop_type_index).any().item()):
            stop_transition_pos = pos
            break
        type_index = TYPE_TO_INDEX[transition_type]
        next_states = torch.nonzero(EDGE_TYPE_MATRIX[current_state] == type_index, as_tuple=False).flatten()
        if int(next_states.numel()) != 1:
            raise RuntimeError(
                f"Transition table has {int(next_states.numel())} exits from "
                f"{STATE_NAMES[current_state]} using {transition_type} at position {pos}"
            )
        current_state = int(next_states[0].item())
    if stop_transition_pos is None:
        raise RuntimeError("Could not find a legal stop transition in synthetic gene")
    transition_types[stop_transition_pos] = "stop"
    for pos in range(stop_transition_pos + 1, len(transition_types)):
        transition_types[pos] = "u3"

    current_state = idx_U(0)
    for pos, transition_type in enumerate(transition_types):
        type_index = TYPE_TO_INDEX[transition_type]
        evidence_targets[pos, type_index] = 1.0
        next_states = torch.nonzero(EDGE_TYPE_MATRIX[current_state] == type_index, as_tuple=False).flatten()
        if int(next_states.numel()) != 1:
            raise RuntimeError(
                f"Transition table has {int(next_states.numel())} exits from "
                f"{STATE_NAMES[current_state]} using {transition_type} at position {pos}"
            )
        current_state = int(next_states[0].item())
        target[pos] = current_state

    return DenseGene(
        dna=dna,
        dna_one_hot=one_hot_dna(dna),
        splice_tracks=splice_tracks,
        target_states=target,
        evidence_targets=evidence_targets,
        start_codon_start=start_pos,
        stop_codon_start=stop_pos,
        stop_transition_position=stop_transition_pos,
        donor_positions=transition_donor_positions,
        acceptor_positions=tuple(acceptor_positions),
        utr5_length=utr5_length,
        utr3_length=utr3_length,
        exon_lengths=tuple(exon_lengths),
        intron_lengths=tuple(intron_lengths),
    )


class DenseTransitionPhaseLayer(nn.Module):
    def __init__(self, edge_type_matrix: torch.Tensor | None = None, *, initial_state: int = 0, materialize_transitions: bool = False):
        super().__init__()
        matrix = EDGE_TYPE_MATRIX if edge_type_matrix is None else edge_type_matrix
        self.register_buffer("edge_type_matrix", matrix.to(torch.long), persistent=False)
        type_masks = torch.stack([(matrix == type_index).to(torch.float32) for type_index in range(K)], dim=0)
        self.register_buffer("edge_type_masks", type_masks, persistent=False)
        edge_from, edge_to, edge_type = edge_list_from_matrix(matrix)
        self.register_buffer("edge_from", edge_from, persistent=False)
        self.register_buffer("edge_to", edge_to, persistent=False)
        self.register_buffer("edge_type", edge_type, persistent=False)
        self.initial_state = initial_state
        self.materialize_transitions = materialize_transitions

    def transition_log_probs(self, evidence_logits: torch.Tensor) -> torch.Tensor:
        if evidence_logits.ndim != 3 or evidence_logits.shape[-1] != K:
            raise ValueError(f"evidence_logits must have shape B x L x {K}")
        allowed = self.edge_type_matrix >= 0
        safe_edge_type = self.edge_type_matrix.clamp_min(0)
        logits = evidence_logits[:, :, safe_edge_type]
        logits = logits.masked_fill(~allowed[None, None, :, :], -torch.inf)
        return torch.log_softmax(logits, dim=-1)

    def step_transition_log_probs(self, evidence_t: torch.Tensor) -> torch.Tensor:
        allowed = self.edge_type_matrix >= 0
        safe_edge_type = self.edge_type_matrix.clamp_min(0)
        logits = evidence_t[:, safe_edge_type]
        logits = logits.masked_fill(~allowed[None, :, :], -torch.inf)
        return torch.log_softmax(logits, dim=-1)

    def transition_type_posterior(self, transition_marginal_t: torch.Tensor) -> torch.Tensor:
        return (transition_marginal_t[:, None, :, :] * self.edge_type_masks[None, :, :, :]).sum(dim=(-2, -1))

    def edge_log_probs(self, evidence_logits: torch.Tensor) -> torch.Tensor:
        """Return row-normalised log-probs for the fixed legal edge list."""

        edge_logits = evidence_logits[..., self.edge_type]
        edge_log_probs = torch.empty_like(edge_logits)
        for state_index in range(S):
            edge_mask = self.edge_from == state_index
            edge_log_probs[..., edge_mask] = torch.log_softmax(edge_logits[..., edge_mask], dim=-1)
        return edge_log_probs

    def step_edge_log_probs(self, evidence_t: torch.Tensor) -> torch.Tensor:
        edge_logits = evidence_t[:, self.edge_type]
        edge_log_probs = torch.empty_like(edge_logits)
        for state_index in range(S):
            edge_mask = self.edge_from == state_index
            edge_log_probs[:, edge_mask] = torch.log_softmax(edge_logits[:, edge_mask], dim=-1)
        return edge_log_probs

    def forward(self, evidence_logits: torch.Tensor) -> DenseLayerOutput:
        squeeze_batch = False
        if evidence_logits.ndim == 2:
            evidence_logits = evidence_logits.unsqueeze(0)
            squeeze_batch = True
        batch, length, _channels = evidence_logits.shape
        state = evidence_logits.new_zeros((batch, S))
        state[:, self.initial_state] = 1.0
        states = []
        type_posteriors = []

        if self.materialize_transitions:
            edge_log_probs = self.edge_log_probs(evidence_logits)
            for t in range(length):
                edge_prob_t = edge_log_probs[:, t].exp()
                edge_marginal_t = state[:, self.edge_from] * edge_prob_t
                next_state = state.new_zeros((batch, S))
                next_state.scatter_add_(1, self.edge_to.expand(batch, -1), edge_marginal_t)
                type_posterior = state.new_zeros((batch, K))
                type_posterior.scatter_add_(1, self.edge_type.expand(batch, -1), edge_marginal_t)
                type_posteriors.append(type_posterior)
                state = next_state
                states.append(state)
        else:
            for t in range(length):
                edge_prob_t = self.step_edge_log_probs(evidence_logits[:, t]).exp()
                edge_marginal_t = state[:, self.edge_from] * edge_prob_t
                next_state = state.new_zeros((batch, S))
                next_state.scatter_add_(1, self.edge_to.expand(batch, -1), edge_marginal_t)
                type_posterior = state.new_zeros((batch, K))
                type_posterior.scatter_add_(1, self.edge_type.expand(batch, -1), edge_marginal_t)
                type_posteriors.append(type_posterior)
                state = next_state
                states.append(state)

        state_probs = torch.stack(states, dim=1)
        state_log_probs = torch.log(state_probs.clamp_min(1.0e-30))
        transition_type_posteriors = torch.stack(type_posteriors, dim=1)
        output = DenseLayerOutput(
            state_probs=state_probs,
            state_log_probs=state_log_probs,
            transition_type_posteriors=transition_type_posteriors,
            start_posterior=transition_type_posteriors[:, :, TYPE_TO_INDEX["start"]],
            stop_posterior=transition_type_posteriors[:, :, TYPE_TO_INDEX["stop"]],
            donor_posterior=transition_type_posteriors[:, :, TYPE_TO_INDEX["donor"]],
            acceptor_posterior=transition_type_posteriors[:, :, TYPE_TO_INDEX["acceptor"]],
        )
        if squeeze_batch:
            return DenseLayerOutput(
                state_probs=output.state_probs.squeeze(0),
                state_log_probs=output.state_log_probs.squeeze(0),
                transition_type_posteriors=output.transition_type_posteriors.squeeze(0),
                start_posterior=output.start_posterior.squeeze(0),
                stop_posterior=output.stop_posterior.squeeze(0),
                donor_posterior=output.donor_posterior.squeeze(0),
                acceptor_posterior=output.acceptor_posterior.squeeze(0),
            )
        return output


class DilatedConvEvidenceModel(nn.Module):
    def __init__(self, *, hidden_dim: int, layers: int, use_splice_tracks: bool):
        super().__init__()
        self.use_splice_tracks = use_splice_tracks
        input_dim = 4 + (2 if use_splice_tracks else 0)
        blocks = [nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3), nn.GELU()]
        for layer_index in range(layers):
            dilation = 2 ** (layer_index % 5)
            blocks.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2 * dilation, dilation=dilation),
                    nn.GELU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.GELU(),
                ]
            )
        blocks.append(nn.Conv1d(hidden_dim, K, kernel_size=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, dna_one_hot: torch.Tensor, splice_tracks: torch.Tensor | None = None) -> torch.Tensor:
        squeeze_batch = False
        if dna_one_hot.ndim == 2:
            dna_one_hot = dna_one_hot.unsqueeze(0)
            squeeze_batch = True
        if dna_one_hot.ndim != 3:
            raise ValueError("dna_one_hot must have shape L x 4 or B x L x 4")
        features = dna_one_hot
        if self.use_splice_tracks:
            if splice_tracks is None:
                raise ValueError("splice_tracks are required when --use-splice-tracks is enabled")
            if splice_tracks.ndim == 2:
                splice_tracks = splice_tracks.unsqueeze(0)
            if splice_tracks.shape[-1] != 2:
                raise ValueError("splice_tracks must contain donor and acceptor channels")
            features = torch.cat([dna_one_hot, splice_tracks], dim=-1)
        output = self.net(features.transpose(1, 2)).transpose(1, 2)
        return output.squeeze(0) if squeeze_batch else output


class DenseTransitionPhaseModel(nn.Module):
    def __init__(self, *, hidden_dim: int, conv_layers: int, use_splice_tracks: bool, materialize_transitions: bool):
        super().__init__()
        self.evidence = DilatedConvEvidenceModel(
            hidden_dim=hidden_dim,
            layers=conv_layers,
            use_splice_tracks=use_splice_tracks,
        )
        self.phase_layer = DenseTransitionPhaseLayer(materialize_transitions=materialize_transitions)

    def forward(self, dna_one_hot: torch.Tensor, splice_tracks: torch.Tensor | None = None) -> tuple[DenseLayerOutput, torch.Tensor]:
        evidence_logits = self.evidence(dna_one_hot, splice_tracks)
        return self.phase_layer(evidence_logits), evidence_logits


def randint(rng: random.Random, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"Invalid range: {low}..{high}")
    return rng.randint(low, high)


def make_gene(args: argparse.Namespace, rng: random.Random) -> DenseGene:
    return generate_dense_gene(
        utr5_length=randint(rng, args.min_utr5_length, args.max_utr5_length),
        coding_codons=randint(rng, args.min_coding_codons, args.max_coding_codons),
        utr3_length=randint(rng, args.min_utr3_length, args.max_utr3_length),
        exon_count=randint(rng, args.min_exons, args.max_exons),
        min_exon_length=args.min_exon_length,
        min_intron_length=args.min_intron_length,
        max_intron_length=args.max_intron_length,
        intron_mod=args.intron_mod,
        seed=rng.randrange(2**31),
    )


def tensors_to_device(gene: DenseGene, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return gene.dna_one_hot.to(device), gene.splice_tracks.to(device), gene.target_states.to(device)


def batch_to_device(genes: list[DenseGene], device: torch.device) -> DenseBatch:
    batch = len(genes)
    max_length = max(len(gene.dna) for gene in genes)
    dna = torch.zeros(batch, max_length, 4, dtype=torch.float32, device=device)
    splice_tracks = torch.zeros(batch, max_length, 2, dtype=torch.float32, device=device)
    evidence = torch.zeros(batch, max_length, K, dtype=torch.float32, device=device)
    targets = torch.zeros(batch, max_length, dtype=torch.long, device=device)
    mask = torch.zeros(batch, max_length, dtype=torch.bool, device=device)
    donor_mask = torch.zeros(batch, max_length, dtype=torch.bool, device=device)
    acceptor_mask = torch.zeros(batch, max_length, dtype=torch.bool, device=device)
    start_positions = torch.empty(batch, dtype=torch.long, device=device)
    stop_positions = torch.empty(batch, dtype=torch.long, device=device)
    lengths = torch.empty(batch, dtype=torch.long, device=device)
    intron_counts = torch.empty(batch, dtype=torch.long, device=device)

    for index, gene in enumerate(genes):
        length = len(gene.dna)
        dna[index, :length] = gene.dna_one_hot.to(device)
        splice_tracks[index, :length] = gene.splice_tracks.to(device)
        evidence[index, :length] = gene.evidence_targets.to(device)
        targets[index, :length] = gene.target_states.to(device)
        mask[index, :length] = True
        start_positions[index] = gene.start_codon_start
        stop_positions[index] = gene.stop_transition_position
        lengths[index] = length
        intron_counts[index] = len(gene.intron_lengths)
        if gene.donor_positions:
            donor_mask[index, torch.tensor(gene.donor_positions, dtype=torch.long, device=device)] = True
        if gene.acceptor_positions:
            acceptor_mask[index, torch.tensor(gene.acceptor_positions, dtype=torch.long, device=device)] = True

    return DenseBatch(
        dna_one_hot=dna,
        splice_tracks=splice_tracks,
        evidence_targets=evidence,
        target_states=targets,
        mask=mask,
        start_positions=start_positions,
        stop_transition_positions=stop_positions,
        donor_mask=donor_mask,
        acceptor_mask=acceptor_mask,
        lengths=lengths,
        intron_counts=intron_counts,
    )


def compute_loss(
    output: DenseLayerOutput,
    evidence_logits: torch.Tensor,
    target_states: torch.Tensor,
    evidence_targets: torch.Tensor,
    gene: DenseGene,
    *,
    evidence_loss_weight: float,
    start_loss_weight: float,
    stop_loss_weight: float,
    donor_loss_weight: float,
    acceptor_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    phase_loss = F.nll_loss(output.state_log_probs, target_states)
    evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, evidence_targets)
    start_loss = -torch.log(output.start_posterior[gene.start_codon_start].clamp_min(1.0e-30))
    stop_loss = -torch.log(output.stop_posterior[gene.stop_transition_position].clamp_min(1.0e-30))
    if gene.donor_positions:
        donor_index = torch.tensor(gene.donor_positions, dtype=torch.long, device=target_states.device)
        donor_loss = -torch.log(output.donor_posterior[donor_index].clamp_min(1.0e-30)).mean()
    else:
        donor_loss = phase_loss.new_zeros(())
    if gene.acceptor_positions:
        acceptor_index = torch.tensor(gene.acceptor_positions, dtype=torch.long, device=target_states.device)
        acceptor_loss = -torch.log(output.acceptor_posterior[acceptor_index].clamp_min(1.0e-30)).mean()
    else:
        acceptor_loss = phase_loss.new_zeros(())
    total = (
        phase_loss
        + evidence_loss_weight * evidence_loss
        + start_loss_weight * start_loss
        + stop_loss_weight * stop_loss
        + donor_loss_weight * donor_loss
        + acceptor_loss_weight * acceptor_loss
    )
    return total, {
        "total": total.detach(),
        "phase": phase_loss.detach(),
        "evidence": evidence_loss.detach(),
        "start": start_loss.detach(),
        "stop": stop_loss.detach(),
        "donor": donor_loss.detach(),
        "acceptor": acceptor_loss.detach(),
    }


def compute_batch_loss(
    output: DenseLayerOutput,
    evidence_logits: torch.Tensor,
    batch: DenseBatch,
    *,
    evidence_loss_weight: float,
    start_loss_weight: float,
    stop_loss_weight: float,
    donor_loss_weight: float,
    acceptor_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    flat_phase_loss = F.nll_loss(
        output.state_log_probs.reshape(-1, S),
        batch.target_states.reshape(-1),
        reduction="none",
    ).reshape_as(batch.target_states)
    phase_loss = flat_phase_loss.masked_select(batch.mask).mean()

    evidence_loss_by_pos = F.binary_cross_entropy_with_logits(
        evidence_logits,
        batch.evidence_targets,
        reduction="none",
    ).mean(dim=-1)
    evidence_loss = evidence_loss_by_pos.masked_select(batch.mask).mean()

    batch_index = torch.arange(batch.target_states.shape[0], device=batch.target_states.device)
    start_loss = -torch.log(output.start_posterior[batch_index, batch.start_positions].clamp_min(1.0e-30)).mean()
    stop_loss = -torch.log(output.stop_posterior[batch_index, batch.stop_transition_positions].clamp_min(1.0e-30)).mean()
    donor_weight = batch.donor_mask.to(output.donor_posterior.dtype)
    donor_loss = (
        -torch.log(output.donor_posterior.clamp_min(1.0e-30)) * donor_weight
    ).sum() / donor_weight.sum().clamp_min(1.0)
    acceptor_weight = batch.acceptor_mask.to(output.acceptor_posterior.dtype)
    acceptor_loss = (
        -torch.log(output.acceptor_posterior.clamp_min(1.0e-30)) * acceptor_weight
    ).sum() / acceptor_weight.sum().clamp_min(1.0)

    total = (
        phase_loss
        + evidence_loss_weight * evidence_loss
        + start_loss_weight * start_loss
        + stop_loss_weight * stop_loss
        + donor_loss_weight * donor_loss
        + acceptor_loss_weight * acceptor_loss
    )
    return total, {
        "total": total.detach(),
        "phase": phase_loss.detach(),
        "evidence": evidence_loss.detach(),
        "start": start_loss.detach(),
        "stop": stop_loss.detach(),
        "donor": donor_loss.detach(),
        "acceptor": acceptor_loss.detach(),
    }


def group_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    groups = {
        "U": [idx_U(g) for g in range(3)],
        "C": [idx_C(g, p) for g in range(3) for p in range(3)],
        "I": [idx_I(g, p) for g in range(3) for p in range(3)],
        "T": [idx_T(g) for g in range(3)],
    }
    result = {}
    for name, states in groups.items():
        mask = torch.zeros_like(target, dtype=torch.bool)
        for state in states:
            mask |= target == state
        result[name] = float((predicted[mask] == target[mask]).float().mean().item()) if bool(mask.any().item()) else float("nan")
    return result


@torch.no_grad()
def evaluate(model: DenseTransitionPhaseModel, args: argparse.Namespace, *, device: torch.device, seed: int, examples: int) -> dict[str, object]:
    rng = random.Random(seed)
    model.eval()
    loss_sums = {key: 0.0 for key in ("total", "phase", "evidence", "start", "stop", "donor", "acceptor")}
    bases = 0
    correct = 0
    exact = 0
    start_exact = 0
    stop_exact = 0
    donor_exact = 0
    acceptor_exact = 0
    length_sum = 0
    intron_sum = 0
    group_correct = {"U": 0, "C": 0, "I": 0, "T": 0}
    group_total = {"U": 0, "C": 0, "I": 0, "T": 0}
    remaining = examples
    while remaining > 0:
        chunk_size = min(args.eval_batch_size, remaining)
        genes = [make_gene(args, rng) for _ in range(chunk_size)]
        batch = batch_to_device(genes, device)
        output, evidence_logits = model(batch.dna_one_hot, batch.splice_tracks if args.use_splice_tracks else None)
        _loss, parts = compute_batch_loss(
            output,
            evidence_logits,
            batch,
            evidence_loss_weight=args.evidence_loss_weight,
            start_loss_weight=args.start_loss_weight,
            stop_loss_weight=args.stop_loss_weight,
            donor_loss_weight=args.donor_loss_weight,
            acceptor_loss_weight=args.acceptor_loss_weight,
        )
        for key, value in parts.items():
            loss_sums[key] += float(value.item()) * chunk_size
        predicted = output.state_log_probs.argmax(dim=-1)
        matches = (predicted == batch.target_states) & batch.mask
        bases += int(batch.mask.sum().item())
        correct += int(matches.sum().item())
        exact += int(((predicted == batch.target_states) | ~batch.mask).all(dim=1).sum().item())
        batch_index = torch.arange(chunk_size, device=device)
        start_exact += int((output.start_posterior.argmax(dim=1) == batch.start_positions).sum().item())
        stop_exact += int((output.stop_posterior.argmax(dim=1) == batch.stop_transition_positions).sum().item())
        donor_exact += int(batch.donor_mask[batch_index, output.donor_posterior.argmax(dim=1)].sum().item())
        acceptor_exact += int(batch.acceptor_mask[batch_index, output.acceptor_posterior.argmax(dim=1)].sum().item())
        length_sum += int(batch.lengths.sum().item())
        intron_sum += int(batch.intron_counts.sum().item())
        groups = {
            "U": [idx_U(g) for g in range(3)],
            "C": [idx_C(g, p) for g in range(3) for p in range(3)],
            "I": [idx_I(g, p) for g in range(3) for p in range(3)],
            "T": [idx_T(g) for g in range(3)],
        }
        for key, states in groups.items():
            group_mask = torch.zeros_like(batch.mask)
            for state in states:
                group_mask |= batch.target_states == state
            group_mask &= batch.mask
            group_total[key] += int(group_mask.sum().item())
            group_correct[key] += int((matches & group_mask).sum().item())
        remaining -= chunk_size
    return {
        "loss": loss_sums["total"] / examples,
        "phase_loss": loss_sums["phase"] / examples,
        "evidence_loss": loss_sums["evidence"] / examples,
        "start_loss": loss_sums["start"] / examples,
        "stop_loss": loss_sums["stop"] / examples,
        "donor_loss": loss_sums["donor"] / examples,
        "acceptor_loss": loss_sums["acceptor"] / examples,
        "base_accuracy": correct / bases,
        "gene_exact": exact / examples,
        "start_exact": start_exact / examples,
        "stop_exact": stop_exact / examples,
        "donor_exact": donor_exact / examples,
        "acceptor_exact": acceptor_exact / examples,
        "mean_length": length_sum / examples,
        "mean_introns": intron_sum / examples,
        "groups": {
            key: (group_correct[key] / group_total[key] if group_total[key] else float("nan"))
            for key in group_total
        },
    }


def format_metrics(prefix: str, metrics: dict[str, object]) -> str:
    groups = metrics["groups"]
    return (
        f"{prefix} loss={metrics['loss']:.4f} "
        f"(phase={metrics['phase_loss']:.4f}, evidence={metrics['evidence_loss']:.4f}, "
        f"start={metrics['start_loss']:.4f}, stop={metrics['stop_loss']:.4f}, "
        f"donor={metrics['donor_loss']:.4f}, acceptor={metrics['acceptor_loss']:.4f})\n"
        f"           24-state base={metrics['base_accuracy']:.3f}, gene exact={metrics['gene_exact']:.3f}, "
        f"start peak={metrics['start_exact']:.3f}, stop peak={metrics['stop_exact']:.3f}, "
        f"donor peak={metrics['donor_exact']:.3f}, acceptor peak={metrics['acceptor_exact']:.3f}\n"
        f"           groups U={groups['U']:.3f}, C={groups['C']:.3f}, I={groups['I']:.3f}, T={groups['T']:.3f}"
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def save_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, args: argparse.Namespace, step: int, validation: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "validation": validation,
            "state_names": STATE_NAMES,
            "transition_types": TRANSITION_TYPES,
            "edge_type_matrix": EDGE_TYPE_MATRIX,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dense transition-matrix phase helper.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "dense_transition_phase_helper")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--examples-per-step", type=int, default=4)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--conv-layers", type=int, default=4)
    parser.add_argument("--use-splice-tracks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--materialize-transitions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.1)
    parser.add_argument("--start-loss-weight", type=float, default=0.25)
    parser.add_argument("--stop-loss-weight", type=float, default=0.25)
    parser.add_argument("--donor-loss-weight", type=float, default=0.1)
    parser.add_argument("--acceptor-loss-weight", type=float, default=0.1)
    parser.add_argument("--min-utr5-length", type=int, default=100)
    parser.add_argument("--max-utr5-length", type=int, default=300)
    parser.add_argument("--min-coding-codons", type=int, default=40)
    parser.add_argument("--max-coding-codons", type=int, default=140)
    parser.add_argument("--min-utr3-length", type=int, default=100)
    parser.add_argument("--max-utr3-length", type=int, default=400)
    parser.add_argument("--min-exons", type=int, default=2)
    parser.add_argument("--max-exons", type=int, default=8)
    parser.add_argument("--min-exon-length", type=int, default=6)
    parser.add_argument("--min-intron-length", type=int, default=50)
    parser.add_argument("--max-intron-length", type=int, default=300)
    parser.add_argument("--intron-mod", choices=("any", "0", "1", "2", "nonzero"), default="any")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = DenseTransitionPhaseModel(
        hidden_dim=args.hidden_dim,
        conv_layers=args.conv_layers,
        use_splice_tracks=args.use_splice_tracks,
        materialize_transitions=args.materialize_transitions,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_rng = random.Random(args.seed + 101)
    best_loss = float("inf")
    start_time = time.time()

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print("task: dense transition-matrix phase helper")
    print(f"states: {S}; transition types: {TRANSITION_TYPES}")
    print(f"CNN evidence model: hidden={args.hidden_dim}, layers={args.conv_layers}, splice_tracks={args.use_splice_tracks}")
    print(f"dense transitions materialized: {args.materialize_transitions}")
    print(
        "synthetic data: "
        f"UTR5={args.min_utr5_length}-{args.max_utr5_length}, "
        f"coding={args.min_coding_codons}-{args.max_coding_codons} codons, "
        f"UTR3={args.min_utr3_length}-{args.max_utr3_length}, "
        f"exons={args.min_exons}-{args.max_exons}, "
        f"introns={args.min_intron_length}-{args.max_intron_length}, mod={args.intron_mod}"
    )
    print(f"checkpoint directory: {args.checkpoint_dir}")
    print("starting training")

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        should_print = step == 1 or step % args.print_every == 0 or step == args.steps
        should_validate = step == 1 or step == args.steps or (args.validate_every > 0 and step % args.validate_every == 0)
        should_save = step % args.checkpoint_every == 0 or step == args.steps
        should_report = should_print or should_validate or should_save
        loss_parts = {key: 0.0 for key in ("total", "phase", "evidence", "start", "stop", "donor", "acceptor")}
        train_counts = {"bases": 0, "correct": 0, "exact": 0, "start": 0, "stop": 0, "donor": 0, "acceptor": 0}
        group_correct = {"U": 0, "C": 0, "I": 0, "T": 0}
        group_total = {"U": 0, "C": 0, "I": 0, "T": 0}

        genes = [make_gene(args, train_rng) for _ in range(args.examples_per_step)]
        batch = batch_to_device(genes, device)
        output, evidence_logits = model(batch.dna_one_hot, batch.splice_tracks if args.use_splice_tracks else None)
        step_loss, parts = compute_batch_loss(
            output,
            evidence_logits,
            batch,
            evidence_loss_weight=args.evidence_loss_weight,
            start_loss_weight=args.start_loss_weight,
            stop_loss_weight=args.stop_loss_weight,
            donor_loss_weight=args.donor_loss_weight,
            acceptor_loss_weight=args.acceptor_loss_weight,
        )
        if should_report:
            with torch.no_grad():
                predicted = output.state_log_probs.argmax(dim=-1)
                matches = (predicted == batch.target_states) & batch.mask
                train_counts["bases"] += int(batch.mask.sum().item())
                train_counts["correct"] += int(matches.sum().item())
                train_counts["exact"] += int(((predicted == batch.target_states) | ~batch.mask).all(dim=1).sum().item())
                batch_index = torch.arange(args.examples_per_step, device=device)
                train_counts["start"] += int((output.start_posterior.argmax(dim=1) == batch.start_positions).sum().item())
                train_counts["stop"] += int((output.stop_posterior.argmax(dim=1) == batch.stop_transition_positions).sum().item())
                train_counts["donor"] += int(batch.donor_mask[batch_index, output.donor_posterior.argmax(dim=1)].sum().item())
                train_counts["acceptor"] += int(batch.acceptor_mask[batch_index, output.acceptor_posterior.argmax(dim=1)].sum().item())
                groups = {
                    "U": [idx_U(g) for g in range(3)],
                    "C": [idx_C(g, p) for g in range(3) for p in range(3)],
                    "I": [idx_I(g, p) for g in range(3) for p in range(3)],
                    "T": [idx_T(g) for g in range(3)],
                }
                for key, states in groups.items():
                    group_mask = torch.zeros_like(batch.mask)
                    for state in states:
                        group_mask |= batch.target_states == state
                    group_mask &= batch.mask
                    group_total[key] += int(group_mask.sum().item())
                    group_correct[key] += int((matches & group_mask).sum().item())
            for key, value in parts.items():
                loss_parts[key] = float(value.item())

        step_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        validation = None
        if should_validate:
            validation = evaluate(model, args, device=device, seed=args.seed + 100000 + step, examples=args.validation_examples)
            if validation["loss"] < best_loss:
                best_loss = float(validation["loss"])
                save_checkpoint(
                    args.checkpoint_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    step=step,
                    validation=validation,
                )
                print(f"saved best checkpoint: {args.checkpoint_dir / 'best.pt'}")
        if should_save:
            if validation is None:
                validation = evaluate(model, args, device=device, seed=args.seed + 100000 + step, examples=args.validation_examples)
            save_checkpoint(
                args.checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                args=args,
                step=step,
                validation=validation,
            )
            with (args.checkpoint_dir / "latest_metrics.json").open("w") as handle:
                json.dump({"step": step, "validation": validation}, handle, indent=2)
            print(f"saved checkpoint: {args.checkpoint_dir / 'latest.pt'}")

        if should_report:
            elapsed = time.time() - start_time
            print(f"\nStep {step:06d} | learning_rate={optimizer.param_groups[0]['lr']:.2e} | elapsed={elapsed:.1f}s")
            print(
                "Batch shape | "
                f"genes={args.examples_per_step}, genome mean={float(batch.lengths.float().mean().item()):.1f} bp, "
                f"genome max={int(batch.lengths.max().item())} bp, "
                f"introns mean={float(batch.intron_counts.float().mean().item()):.2f}"
            )
            train_metrics = {
                "loss": loss_parts["total"],
                "phase_loss": loss_parts["phase"],
                "evidence_loss": loss_parts["evidence"],
                "start_loss": loss_parts["start"],
                "stop_loss": loss_parts["stop"],
                "donor_loss": loss_parts["donor"],
                "acceptor_loss": loss_parts["acceptor"],
                "base_accuracy": train_counts["correct"] / train_counts["bases"],
                "gene_exact": train_counts["exact"] / args.examples_per_step,
                "start_exact": train_counts["start"] / args.examples_per_step,
                "stop_exact": train_counts["stop"] / args.examples_per_step,
                "donor_exact": train_counts["donor"] / args.examples_per_step,
                "acceptor_exact": train_counts["acceptor"] / args.examples_per_step,
                "groups": {
                    key: (group_correct[key] / group_total[key] if group_total[key] else float("nan"))
                    for key in group_total
                },
            }
            print(format_metrics("train     ", train_metrics))
            if validation is not None:
                print(format_metrics("validation", validation))


if __name__ == "__main__":
    main()
