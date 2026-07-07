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
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from central_dogma_ai.biology import CODONS_BY_AA, DNA_BASES, DNA_TO_INDEX, STOP_CODONS  # noqa: E402


SWEEP_STATE_NAMES = (
    "U5_0",
    "U5_1",
    "U5_2",
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
    "U3_0",
    "U3_1",
    "U3_2",
)
NUM_STATES = len(SWEEP_STATE_NAMES)
EVIDENCE_NAMES = ("start", "stop", "donor", "acceptor", "exon", "intron")


def idx_U5(g: int) -> int:
    return g


def idx_C(g: int, p: int) -> int:
    return 3 + g * 3 + p


def idx_I(g: int, p: int) -> int:
    return 12 + g * 3 + p


def idx_U3(g: int) -> int:
    return 21 + g


def one_hot_dna(sequence: str) -> torch.Tensor:
    encoded = torch.zeros(len(sequence), 4, dtype=torch.float32)
    for index, base in enumerate(sequence):
        encoded[index, DNA_TO_INDEX[base]] = 1.0
    return encoded


def random_dna(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA_BASES) for _ in range(length))


def random_protein(codons: int, rng: random.Random) -> str:
    if codons < 1:
        raise ValueError("protein codons must be at least 1")
    amino_acids = tuple(aa for aa in CODONS_BY_AA if aa not in {"*", "M"})
    return "M" + "".join(rng.choice(amino_acids) for _ in range(codons - 1))


def reverse_translate_with_stop(protein: str, rng: random.Random) -> str:
    codons = []
    for index, amino_acid in enumerate(protein):
        if index == 0:
            codons.append("ATG")
        else:
            codons.append(rng.choice(CODONS_BY_AA[amino_acid]))
    codons.append(rng.choice(tuple(STOP_CODONS)))
    return "".join(codons)


@dataclass(frozen=True)
class SweepGene:
    dna: str
    dna_one_hot: torch.Tensor
    target_states: torch.Tensor
    structure_tracks: torch.Tensor
    start_codon_start: int
    stop_codon_start: int
    donor_positions: tuple[int, ...]
    acceptor_positions: tuple[int, ...]
    utr5_length: int
    utr3_length: int
    exon_lengths: tuple[int, ...]
    intron_lengths: tuple[int, ...]


@dataclass(frozen=True)
class SweepLayerOutput:
    state_log_probs: torch.Tensor
    state_probs: torch.Tensor
    initiation_log_probs: torch.Tensor
    termination_log_probs: torch.Tensor
    donor_log_probs: torch.Tensor
    acceptor_log_probs: torch.Tensor


def intron_length_with_mod(
    *,
    min_length: int,
    max_length: int,
    mode: str,
    rng: random.Random,
) -> int:
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


def sample_cds_cut_points(
    *,
    cds_length: int,
    intron_count: int,
    min_exon_length: int,
    rng: random.Random,
) -> tuple[int, ...]:
    if intron_count < 0:
        raise ValueError("intron_count must be non-negative")
    if intron_count == 0:
        return ()
    # Cut positions are after this many CDS bases. Avoid splitting ATG and stop.
    possible = list(range(3, cds_length - 3 + 1))
    for _attempt in range(1000):
        cuts = tuple(sorted(rng.sample(possible, intron_count)))
        exon_lengths = (cuts[0],) + tuple(right - left for left, right in zip(cuts, cuts[1:])) + (cds_length - cuts[-1],)
        if all(length >= min_exon_length for length in exon_lengths):
            return cuts
    raise ValueError("Could not sample CDS cut points for requested exon constraints")


def generate_sweep_gene(
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
) -> SweepGene:
    """Generate genomic DNA and 24-state labels without transcript path objects.

    The matrix is built in register: U5 rows, CDS rows, U3 rows, then introns are
    inserted into CDS positions. 5' UTR sequence is random and may contain ATGs;
    labels mark the deliberately generated main CDS start.
    """

    if coding_codons < 2:
        raise ValueError("coding_codons includes start and stop codons, so it must be at least 2")
    if exon_count < 1:
        raise ValueError("exon_count must be at least 1")
    if min_intron_length < 4:
        raise ValueError("min_intron_length must allow GT...AG introns")

    rng = random.Random(seed)
    utr5 = random_dna(utr5_length, rng)
    protein = random_protein(coding_codons - 1, rng)
    cds = reverse_translate_with_stop(protein, rng)
    utr3 = random_dna(utr3_length, rng)
    intron_count = exon_count - 1
    cuts = sample_cds_cut_points(
        cds_length=len(cds),
        intron_count=intron_count,
        min_exon_length=min_exon_length,
        rng=rng,
    )

    rows: list[tuple[str, str, int | None]] = []
    rows.extend((base, "U5", None) for base in utr5)
    start_pos = len(rows)
    donor_positions: list[int] = []
    acceptor_positions: list[int] = []
    intron_lengths: list[int] = []
    exon_lengths: list[int] = [utr5_length]

    previous_cut = 0
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
            intron_length = intron_length_with_mod(
                min_length=min_intron_length,
                max_length=max_intron_length,
                mode=intron_mod,
                rng=rng,
            )
            intron_lengths.append(intron_length)
            carried_phase = cut % 3
            intron = "GT" + random_dna(intron_length - 4, rng) + "AG"
            rows.extend((base, "I", carried_phase) for base in intron)
            acceptor_positions.append(len(rows))
            exon_lengths.append(0)

    rows.extend((base, "U3", None) for base in utr3)
    exon_lengths[-1] += utr3_length

    dna = "".join(base for base, _region, _phase in rows)
    target = torch.empty(len(rows), dtype=torch.long)
    structure = torch.zeros(len(rows), 4, dtype=torch.float32)
    donor_set = set(donor_positions)
    acceptor_set = set(acceptor_positions)
    for pos, (_base, region, phase) in enumerate(rows):
        g = pos % 3
        if region == "U5":
            target[pos] = idx_U5(g)
            structure[pos, 2] = 1.0
        elif region == "C":
            if phase is None:
                raise AssertionError("coding rows need mature CDS phase")
            target[pos] = idx_C(g, phase)
            structure[pos, 2] = 1.0
        elif region == "I":
            if phase is None:
                raise AssertionError("intron rows need carried mature CDS phase")
            target[pos] = idx_I(g, phase)
            structure[pos, 3] = 1.0
        elif region == "U3":
            target[pos] = idx_U3(g)
            structure[pos, 2] = 1.0
        else:
            raise AssertionError(f"unknown region: {region}")
        if pos in donor_set:
            structure[pos, 0] = 1.0
        if pos in acceptor_set:
            structure[pos, 1] = 1.0

    return SweepGene(
        dna=dna,
        dna_one_hot=one_hot_dna(dna),
        target_states=target,
        structure_tracks=structure,
        start_codon_start=start_pos,
        stop_codon_start=stop_pos,
        donor_positions=tuple(donor_positions),
        acceptor_positions=tuple(acceptor_positions),
        utr5_length=utr5_length,
        utr3_length=utr3_length,
        exon_lengths=tuple(exon_lengths),
        intron_lengths=tuple(intron_lengths),
    )


class GenomeSweepPhaseLayer(nn.Module):
    """Left-to-right genomic phase DP.

    Convention: `state_log_probs[t]` describes the state of genomic base `t`
    after that base has been consumed. The state's genomic track `g` is
    therefore always `t % 3`.
    """

    def __init__(self, *, num_states: int = NUM_STATES):
        super().__init__()
        self.num_states = num_states
        
        edges_from = []
        incoming_edges = []
        
        for g in range(3):
            prev_g = (g - 1) % 3
            e_from = []
            e_to = []
            
            # edge 0: U5 -> U5
            e_from.append(idx_U5(prev_g))
            e_to.append(idx_U5(g))

            # edge 1: U5 -> C0
            e_from.append(idx_U5(prev_g))
            e_to.append(idx_C(g, 0))

            # edge 2: U3 -> U3
            e_from.append(idx_U3(prev_g))
            e_to.append(idx_U3(g))

            # edge 3: C2 -> U3
            e_from.append(idx_C(prev_g, 2))
            e_to.append(idx_U3(g))

            # edge 4,5,6: C -> C
            for p in range(3):
                e_from.append(idx_C(prev_g, p))
                e_to.append(idx_C(g, (p + 1) % 3))

            # edge 7,8,9: C -> I
            for p in range(3):
                e_from.append(idx_C(prev_g, p))
                e_to.append(idx_I(g, (p + 1) % 3))

            # edge 10,11,12: I -> I
            for p in range(3):
                e_from.append(idx_I(prev_g, p))
                e_to.append(idx_I(g, p))

            # edge 13,14,15: I -> C
            for p in range(3):
                e_from.append(idx_I(prev_g, p))
                e_to.append(idx_C(g, p))
                
            edges_from.append(e_from)
            
            incoming = {}
            for e, to_s in enumerate(e_to):
                if to_s not in incoming:
                    incoming[to_s] = []
                incoming[to_s].append(e)
            incoming_edges.append([(to_s, edge_list) for to_s, edge_list in incoming.items()])

        self.register_buffer("edges_from_tensor", torch.tensor(edges_from, dtype=torch.long))
        self.incoming_edges = incoming_edges

    @staticmethod
    def _transition_log_probs(scores: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(scores)
        safe_scores = torch.where(finite, scores, scores.new_full(scores.shape, -1.0e30))
        log_z = torch.logsumexp(safe_scores, dim=-1, keepdim=True)
        return torch.where(finite.any(dim=-1, keepdim=True), scores - log_z, scores)

    def forward(self, evidence_logits: torch.Tensor, padding_mask: torch.Tensor | None = None) -> SweepLayerOutput:
        is_batched = evidence_logits.ndim == 3
        if not is_batched:
            evidence_logits = evidence_logits.unsqueeze(0)
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(0)

        B, L, E = evidence_logits.shape
        if E != len(EVIDENCE_NAMES):
            raise ValueError(f"evidence_logits must have shape [..., {len(EVIDENCE_NAMES)}]")

        alpha_list = [None] * L
        initiation_list = [evidence_logits.new_full((B,), -torch.inf)] * L
        termination_list = [evidence_logits.new_full((B,), -torch.inf)] * L
        donor_list = [evidence_logits.new_full((B,), -torch.inf)] * L
        acceptor_list = [evidence_logits.new_full((B,), -torch.inf)] * L

        start_logits = evidence_logits[:, :, 0]
        stop_logits = evidence_logits[:, :, 1]
        donor_logits = evidence_logits[:, :, 2]
        acceptor_logits = evidence_logits[:, :, 3]
        exon_logits = evidence_logits[:, :, 4]
        intron_logits = evidence_logits[:, :, 5]

        # Vectorized pre-computation of all edge evidence scores
        edge_scores = evidence_logits.new_empty((B, L, 16))
        
        # 0: U5 -> U5
        edge_scores[:, :, 0] = exon_logits
        # 1: U5 -> C0
        edge_scores[:, :, 1] = start_logits + exon_logits
        # 2: U3 -> U3
        edge_scores[:, :, 2] = exon_logits
        
        # 3: C2 -> U3
        edge_scores[:, :, 3] = -torch.inf
        if L >= 3:
            edge_scores[:, 3:, 3] = stop_logits[:, :-3] + exon_logits[:, 3:]

        # 4,5,6: C -> C
        edge_scores[:, :, 4:7] = exon_logits.unsqueeze(2).expand(-1, -1, 3)

        # 7,8,9: C -> I
        edge_scores[:, :, 7:10] = -torch.inf
        if L >= 1:
            edge_scores[:, 1:, 7:10] = (donor_logits[:, :-1] + intron_logits[:, 1:]).unsqueeze(2).expand(-1, -1, 3)

        # 10,11,12: I -> I
        edge_scores[:, :, 10:13] = intron_logits.unsqueeze(2).expand(-1, -1, 3)

        # 13,14,15: I -> C
        edge_scores[:, :, 13:16] = (acceptor_logits + exon_logits).unsqueeze(2).expand(-1, -1, 3)

        previous = evidence_logits.new_full((B, self.num_states), -torch.inf)
        previous[:, idx_U5(2)] = 0.0

        for t in range(L):
            g = t % 3
            
            # current_edge_scores: [B, 16]
            current_edge_scores = previous[:, self.edges_from_tensor[g]] + edge_scores[:, t, :]
            
            initiation_list[t] = current_edge_scores[:, 1]
            if t >= 3:
                termination_list[t - 3] = current_edge_scores[:, 3]
            if t >= 1:
                donor_list[t - 1] = torch.logsumexp(current_edge_scores[:, 7:10], dim=1)
            acceptor_list[t] = torch.logsumexp(current_edge_scores[:, 13:16], dim=1)
            
            current = evidence_logits.new_full((B, self.num_states), -torch.inf)
            for to_s, edge_indices in self.incoming_edges[g]:
                if len(edge_indices) == 1:
                    current[:, to_s] = current_edge_scores[:, edge_indices[0]]
                else:
                    current[:, to_s] = torch.logsumexp(current_edge_scores[:, edge_indices], dim=1)
            
            alpha_list[t] = current
            
            if padding_mask is not None:
                valid = padding_mask[:, t].unsqueeze(1)
                previous = torch.where(valid, current, previous)
            else:
                previous = current

        alpha = torch.stack(alpha_list, dim=1)
        initiation_scores = torch.stack(initiation_list, dim=1)
        termination_scores = torch.stack(termination_list, dim=1)
        donor_scores = torch.stack(donor_list, dim=1)
        acceptor_scores = torch.stack(acceptor_list, dim=1)

        state_log_probs = torch.log_softmax(alpha, dim=-1)
        
        if padding_mask is not None:
            mask_float = (~padding_mask).float() * -1.0e30
            initiation_scores = initiation_scores + mask_float
            termination_scores = termination_scores + mask_float
            donor_scores = donor_scores + mask_float
            acceptor_scores = acceptor_scores + mask_float

        res = SweepLayerOutput(
            state_log_probs=state_log_probs if is_batched else state_log_probs.squeeze(0),
            state_probs=state_log_probs.exp() if is_batched else state_log_probs.exp().squeeze(0),
            initiation_log_probs=self._transition_log_probs(initiation_scores) if is_batched else self._transition_log_probs(initiation_scores).squeeze(0),
            termination_log_probs=self._transition_log_probs(termination_scores) if is_batched else self._transition_log_probs(termination_scores).squeeze(0),
            donor_log_probs=self._transition_log_probs(donor_scores) if is_batched else self._transition_log_probs(donor_scores).squeeze(0),
            acceptor_log_probs=self._transition_log_probs(acceptor_scores) if is_batched else self._transition_log_probs(acceptor_scores).squeeze(0),
        )
        return res

class SweepEvidenceModel(nn.Module):
    def __init__(self, *, hidden_dim: int, use_structure_tracks: bool):
        super().__init__()
        self.use_structure_tracks = use_structure_tracks
        input_dim = 8 if use_structure_tracks else 4
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, len(EVIDENCE_NAMES), kernel_size=1),
        )

    def forward(self, dna_one_hot: torch.Tensor, structure_tracks: torch.Tensor | None = None) -> torch.Tensor:
        if dna_one_hot.ndim not in (2, 3) or dna_one_hot.shape[-1] != 4:
            raise ValueError("dna_one_hot must have shape [L, 4] or [B, L, 4]")
        is_batched = dna_one_hot.ndim == 3
        if not is_batched:
            dna_one_hot = dna_one_hot.unsqueeze(0)
            if structure_tracks is not None:
                structure_tracks = structure_tracks.unsqueeze(0)

        features = dna_one_hot
        if self.use_structure_tracks:
            if structure_tracks is None:
                raise ValueError("structure_tracks are required when use_structure_tracks=True")
            features = torch.cat([dna_one_hot, structure_tracks], dim=-1)
        
        logits = self.net(features.transpose(1, 2)).transpose(1, 2)
        if not is_batched:
            return logits.squeeze(0)
        return logits


class GenomeSweepPhaseModel(nn.Module):
    def __init__(self, *, hidden_dim: int, use_structure_tracks: bool):
        super().__init__()
        self.evidence = SweepEvidenceModel(hidden_dim=hidden_dim, use_structure_tracks=use_structure_tracks)
        self.phase_layer = GenomeSweepPhaseLayer()

    def forward(self, dna_one_hot: torch.Tensor, structure_tracks: torch.Tensor | None = None, padding_mask: torch.Tensor | None = None) -> tuple[SweepLayerOutput, torch.Tensor]:
        evidence_logits = self.evidence(dna_one_hot, structure_tracks)
        return self.phase_layer(evidence_logits, padding_mask), evidence_logits


def randint(rng: random.Random, low: int, high: int) -> int:
    if high < low:
        raise ValueError(f"Invalid range: {low}..{high}")
    return rng.randint(low, high)


def make_gene(args: argparse.Namespace, rng: random.Random) -> SweepGene:
    return generate_sweep_gene(
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



class SweepGeneDataset(IterableDataset):
    def __init__(self, args, seed):
        self.args = args
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + worker_id)
        while True:
            yield make_gene(self.args, rng)

def collate_sweep_genes(genes: list[SweepGene]) -> dict:
    lengths = [len(g.dna) for g in genes]
    max_len = max(lengths)
    batch_size = len(genes)
    
    dna_one_hot = torch.zeros(batch_size, max_len, 4, dtype=torch.float32)
    structure_tracks = torch.zeros(batch_size, max_len, 4, dtype=torch.float32)
    target_states = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, g in enumerate(genes):
        l = len(g.dna)
        dna_one_hot[i, :l] = g.dna_one_hot
        structure_tracks[i, :l] = g.structure_tracks
        target_states[i, :l] = g.target_states
        padding_mask[i, :l] = True

    return {
        "dna_one_hot": dna_one_hot,
        "structure_tracks": structure_tracks,
        "target_states": target_states,
        "padding_mask": padding_mask,
        "genes": genes,
        "lengths": lengths
    }

def tensors_to_device(gene: SweepGene, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        gene.dna_one_hot.to(device),
        gene.structure_tracks.to(device),
        gene.target_states.to(device),
    )


def motif_targets(gene: SweepGene, device: torch.device) -> torch.Tensor:
    targets = torch.zeros(len(gene.dna), len(EVIDENCE_NAMES), dtype=torch.float32, device=device)
    for pos in range(len(gene.dna) - 2):
        codon = gene.dna[pos : pos + 3]
        if codon == "ATG":
            targets[pos, 0] = 1.0
        if codon in STOP_CODONS:
            targets[pos, 1] = 1.0
    for pos in gene.donor_positions:
        targets[pos, 2] = 1.0
    for pos in gene.acceptor_positions:
        targets[pos, 3] = 1.0
    targets[:, 4] = gene.structure_tracks[:, 2].to(device)
    targets[:, 5] = gene.structure_tracks[:, 3].to(device)
    return targets


def compute_loss(
    output: SweepLayerOutput,
    evidence_logits: torch.Tensor,
    target_states: torch.Tensor,
    padding_mask: torch.Tensor,
    genes: list[SweepGene],
    *,
    start_loss_weight: float,
    stop_loss_weight: float,
    donor_loss_weight: float,
    acceptor_loss_weight: float,
    evidence_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    B, L = target_states.shape
    
    state_log_probs_flat = output.state_log_probs.reshape(-1, NUM_STATES)
    target_states_flat = target_states.reshape(-1)
    phase_loss = F.nll_loss(state_log_probs_flat, target_states_flat, reduction='none').reshape(B, L)
    phase_loss = (phase_loss * padding_mask.float()).sum() / padding_mask.float().sum()
    
    start_loss = phase_loss.new_zeros(())
    stop_loss = phase_loss.new_zeros(())
    donor_loss = phase_loss.new_zeros(())
    acceptor_loss = phase_loss.new_zeros(())
    donor_count = 0
    acceptor_count = 0

    evidence_targets = torch.zeros(B, L, len(EVIDENCE_NAMES), dtype=torch.float32, device=target_states.device)

    for i, gene in enumerate(genes):
        start_loss -= output.initiation_log_probs[i, gene.start_codon_start]
        stop_loss -= output.termination_log_probs[i, gene.stop_codon_start]
        
        if gene.donor_positions:
            for p in gene.donor_positions:
                donor_loss -= output.donor_log_probs[i, p]
                donor_count += 1
        
        if gene.acceptor_positions:
            for p in gene.acceptor_positions:
                acceptor_loss -= output.acceptor_log_probs[i, p]
                acceptor_count += 1

        for pos in range(len(gene.dna) - 2):
            codon = gene.dna[pos : pos + 3]
            if codon == "ATG":
                evidence_targets[i, pos, 0] = 1.0
            if codon in STOP_CODONS:
                evidence_targets[i, pos, 1] = 1.0
        for pos in gene.donor_positions:
            evidence_targets[i, pos, 2] = 1.0
        for pos in gene.acceptor_positions:
            evidence_targets[i, pos, 3] = 1.0
        
        evidence_targets[i, :len(gene.dna), 4] = gene.structure_tracks[:, 2].to(evidence_targets.device)
        evidence_targets[i, :len(gene.dna), 5] = gene.structure_tracks[:, 3].to(evidence_targets.device)

    start_loss = start_loss / B
    stop_loss = stop_loss / B
    donor_loss = donor_loss / donor_count if donor_count > 0 else phase_loss.new_zeros(())
    acceptor_loss = acceptor_loss / acceptor_count if acceptor_count > 0 else phase_loss.new_zeros(())

    evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, evidence_targets, reduction='none')
    padding_mask_expanded = padding_mask.unsqueeze(-1).expand_as(evidence_loss)
    evidence_loss = (evidence_loss * padding_mask_expanded.float()).sum() / padding_mask_expanded.float().sum()

    total = (
        phase_loss
        + start_loss_weight * start_loss
        + stop_loss_weight * stop_loss
        + donor_loss_weight * donor_loss
        + acceptor_loss_weight * acceptor_loss
        + evidence_loss_weight * evidence_loss
    )
    return total, {
        "total": total.detach(),
        "phase": phase_loss.detach(),
        "start": start_loss.detach(),
        "stop": stop_loss.detach(),
        "donor": donor_loss.detach(),
        "acceptor": acceptor_loss.detach(),
        "evidence": evidence_loss.detach(),
    }

def split_state_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    result = {}
    groups = {
        "U5": [idx_U5(g) for g in range(3)],
        "C": [idx_C(g, p) for g in range(3) for p in range(3)],
        "I": [idx_I(g, p) for g in range(3) for p in range(3)],
        "U3": [idx_U3(g) for g in range(3)],
    }
    for name, states in groups.items():
        mask = torch.zeros_like(target, dtype=torch.bool)
        for state in states:
            mask |= target == state
        result[name] = float((predicted[mask] == target[mask]).float().mean().item()) if bool(mask.any().item()) else float("nan")
    return result


@torch.no_grad()
@torch.no_grad()
def evaluate(model: GenomeSweepPhaseModel, args: argparse.Namespace, *, device: torch.device, seed: int, examples: int) -> dict[str, object]:
    model.eval()
    dataset = SweepGeneDataset(args, seed)
    loader = DataLoader(dataset, batch_size=args.examples_per_step, collate_fn=collate_sweep_genes, num_workers=2)
    
    sums = {key: 0.0 for key in ("total", "phase", "start", "stop", "donor", "acceptor", "evidence")}
    bases = 0
    correct = 0
    exact = 0
    start_exact = 0
    stop_exact = 0
    donor_exact = 0
    acceptor_exact = 0
    length_sum = 0
    intron_sum = 0
    state_group_sums = {"U5": 0.0, "C": 0.0, "I": 0.0, "U3": 0.0}
    
    processed = 0
    for batch in loader:
        if processed >= examples:
            break
            
        dna_one_hot = batch["dna_one_hot"].to(device)
        structure_tracks = batch["structure_tracks"].to(device)
        target = batch["target_states"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        genes = batch["genes"]
        
        # AMP for eval
        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu"):
            output, evidence_logits = model(dna_one_hot, structure_tracks if args.use_structure_tracks else None, padding_mask=padding_mask)
            _loss, parts = compute_loss(
                output,
                evidence_logits,
                target,
                padding_mask,
                genes,
                start_loss_weight=args.start_loss_weight,
                stop_loss_weight=args.stop_loss_weight,
                donor_loss_weight=args.donor_loss_weight,
                acceptor_loss_weight=args.acceptor_loss_weight,
                evidence_loss_weight=args.evidence_loss_weight,
            )
        
        predicted = output.state_log_probs.argmax(dim=-1)
        matches = predicted == target
        
        for i, gene in enumerate(genes):
            gene_len = len(gene.dna)
            gene_matches = matches[i, :gene_len]
            correct += int(gene_matches.sum().item())
            exact += int(bool(gene_matches.all().item()))
            bases += gene_len
            length_sum += gene_len
            intron_sum += len(gene.intron_lengths)
            
            start_exact += int(output.initiation_log_probs[i].argmax().item() == gene.start_codon_start)
            stop_exact += int(output.termination_log_probs[i].argmax().item() == gene.stop_codon_start)
            if gene.donor_positions:
                donor_exact += int(output.donor_log_probs[i].argmax().item() in gene.donor_positions)
            else:
                donor_exact += 1
            if gene.acceptor_positions:
                acceptor_exact += int(output.acceptor_log_probs[i].argmax().item() in gene.acceptor_positions)
            else:
                acceptor_exact += 1
                
        for key, value in parts.items():
            sums[key] += float(value.item()) * len(genes)
            
        state_acc = split_state_accuracy(predicted, target)
        for key, value in state_acc.items():
            if value == value:
                state_group_sums[key] += value * len(genes)
                
        processed += len(genes)

    return {
        "loss": sums["total"] / processed,
        "phase_loss": sums["phase"] / processed,
        "start_loss": sums["start"] / processed,
        "stop_loss": sums["stop"] / processed,
        "donor_loss": sums["donor"] / processed,
        "acceptor_loss": sums["acceptor"] / processed,
        "evidence_loss": sums["evidence"] / processed,
        "base_accuracy": correct / bases,
        "gene_exact": exact / processed,
        "start_exact": start_exact / processed,
        "stop_exact": stop_exact / processed,
        "donor_exact": donor_exact / processed,
        "acceptor_exact": acceptor_exact / processed,
        "mean_length": length_sum / processed,
        "mean_introns": intron_sum / processed,
        "state_groups": {key: value / processed for key, value in state_group_sums.items()},
    }

def format_metrics(prefix: str, metrics: dict[str, object]) -> str:
    groups = metrics["state_groups"]
    return (
        f"{prefix} loss={metrics['loss']:.4f} "
        f"(phase={metrics['phase_loss']:.4f}, start={metrics['start_loss']:.4f}, "
        f"stop={metrics['stop_loss']:.4f}, donor={metrics['donor_loss']:.4f}, "
        f"acceptor={metrics['acceptor_loss']:.4f}, evidence={metrics['evidence_loss']:.4f})\n"
        f"           24-state base={metrics['base_accuracy']:.3f}, gene exact={metrics['gene_exact']:.3f}, "
        f"start peak={metrics['start_exact']:.3f}, stop peak={metrics['stop_exact']:.3f}, "
        f"donor peak={metrics['donor_exact']:.3f}, acceptor peak={metrics['acceptor_exact']:.3f}\n"
        f"           state groups U5={groups['U5']:.3f}, C={groups['C']:.3f}, "
        f"I={groups['I']:.3f}, U3={groups['U3']:.3f}"
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    validation: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "validation": validation,
            "state_names": SWEEP_STATE_NAMES,
            "evidence_names": EVIDENCE_NAMES,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train genomic sweep 3x3 phase helper.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "genome_sweep_phase_helper")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--examples-per-step", type=int, default=4)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--use-structure-tracks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-loss-weight", type=float, default=0.25)
    parser.add_argument("--stop-loss-weight", type=float, default=0.25)
    parser.add_argument("--donor-loss-weight", type=float, default=0.1)
    parser.add_argument("--acceptor-loss-weight", type=float, default=0.1)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.1)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = GenomeSweepPhaseModel(hidden_dim=args.hidden_dim, use_structure_tracks=args.use_structure_tracks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = SweepGeneDataset(args, args.seed + 101)
    
    prefetch = 2 if args.num_workers > 0 else None
    train_loader = DataLoader(train_dataset, batch_size=args.examples_per_step, collate_fn=collate_sweep_genes, num_workers=args.num_workers, prefetch_factor=prefetch)
    train_iter = iter(train_loader)
    
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if hasattr(torch, "compile") and not args.disable_compile:
        try:
            model.evidence = torch.compile(model.evidence)
            print("Successfully applied torch.compile() to evidence CNN")
        except Exception as e:
            print(f"Could not apply torch.compile(): {e}")

    best_loss = float("inf")
    start_time = time.time()

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print("task: genomic left-to-right sweep phase helper")
    print(f"states: {NUM_STATES} ({', '.join(SWEEP_STATE_NAMES)})")
    print("coding state: C[g,p], intron carry state: I[g,p]")
    print(f"evidence logits: {EVIDENCE_NAMES}; structure tracks={args.use_structure_tracks}")
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
        should_save = step == args.steps or (args.checkpoint_every > 0 and step % args.checkpoint_every == 0)
        should_report = should_print or should_validate or should_save
        
        batch = next(train_iter)
        dna_one_hot = batch["dna_one_hot"].to(device)
        structure_tracks = batch["structure_tracks"].to(device)
        target = batch["target_states"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        genes = batch["genes"]
        
        loss_parts = {key: 0.0 for key in ("total", "phase", "start", "stop", "donor", "acceptor", "evidence")}
        train_counts = {"bases": 0, "correct": 0, "exact": 0, "start": 0, "stop": 0, "donor": 0, "acceptor": 0}
        length_sum = 0
        intron_sum = 0
        group_sums = {"U5": 0.0, "C": 0.0, "I": 0.0, "U3": 0.0}

        # Use AMP
        autocast_ctx = torch.autocast(device_type=device.type) if device.type in ("cuda", "cpu") else torch.autocast(device_type="cpu", enabled=False)
        with autocast_ctx:
            output, evidence_logits = model(dna_one_hot, structure_tracks if args.use_structure_tracks else None, padding_mask=padding_mask)
            loss, parts = compute_loss(
                output,
                evidence_logits,
                target,
                padding_mask,
                genes,
                start_loss_weight=args.start_loss_weight,
                stop_loss_weight=args.stop_loss_weight,
                donor_loss_weight=args.donor_loss_weight,
                acceptor_loss_weight=args.acceptor_loss_weight,
                evidence_loss_weight=args.evidence_loss_weight,
            )
            
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if should_report:
            with torch.no_grad():
                predicted = output.state_log_probs.argmax(dim=-1)
                matches = predicted == target
                
                for i, gene in enumerate(genes):
                    gene_len = len(gene.dna)
                    gene_matches = matches[i, :gene_len]
                    train_counts["bases"] += gene_len
                    train_counts["correct"] += int(gene_matches.sum().item())
                    train_counts["exact"] += int(bool(gene_matches.all().item()))
                    train_counts["start"] += int(output.initiation_log_probs[i].argmax().item() == gene.start_codon_start)
                    train_counts["stop"] += int(output.termination_log_probs[i].argmax().item() == gene.stop_codon_start)
                    train_counts["donor"] += int(output.donor_log_probs[i].argmax().item() in gene.donor_positions) if gene.donor_positions else 1
                    train_counts["acceptor"] += int(output.acceptor_log_probs[i].argmax().item() in gene.acceptor_positions) if gene.acceptor_positions else 1
                    length_sum += gene_len
                    intron_sum += len(gene.intron_lengths)

                state_acc = split_state_accuracy(predicted, target)
                for key, value in state_acc.items():
                    if value == value:
                        group_sums[key] += value
                        
            for key, value in parts.items():
                loss_parts[key] += float(value.item())

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

        if should_print or should_validate or should_save:
            elapsed = time.time() - start_time
            print(f"\nStep {step:06d} | learning_rate={optimizer.param_groups[0]['lr']:.2e} | elapsed={elapsed:.1f}s")
            print(
                "Batch shape | "
                f"genes={args.examples_per_step}, genome mean={length_sum / args.examples_per_step:.1f} bp, "
                f"introns mean={intron_sum / args.examples_per_step:.2f}"
            )
            train_metrics = {
                "loss": loss_parts["total"],
                "phase_loss": loss_parts["phase"],
                "start_loss": loss_parts["start"],
                "stop_loss": loss_parts["stop"],
                "donor_loss": loss_parts["donor"],
                "acceptor_loss": loss_parts["acceptor"],
                "evidence_loss": loss_parts["evidence"],
                "base_accuracy": train_counts["correct"] / train_counts["bases"],
                "gene_exact": train_counts["exact"] / args.examples_per_step,
                "start_exact": train_counts["start"] / args.examples_per_step,
                "stop_exact": train_counts["stop"] / args.examples_per_step,
                "donor_exact": train_counts["donor"] / args.examples_per_step,
                "acceptor_exact": train_counts["acceptor"] / args.examples_per_step,
                "state_groups": {key: value / args.examples_per_step for key, value in group_sums.items()},
            }
            print(format_metrics("train     ", train_metrics))
            if validation is not None:
                print(format_metrics("validation", validation))


if __name__ == "__main__":
    main()
