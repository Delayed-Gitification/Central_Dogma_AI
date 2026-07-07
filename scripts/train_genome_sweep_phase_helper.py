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

    Mature CDS phase `p` advances only when a coding exonic base is consumed.
    During coding introns, genomic track advances but carried `p` is preserved.
    Stop transitions are delayed: a stop logit at codon start `s` moves the DP
    from coding to U3 at base `s + 3`, so the stop codon itself is labelled C0,
    C1, C2.
    """

    def __init__(self, *, num_states: int = NUM_STATES):
        super().__init__()
        self.num_states = num_states

    @staticmethod
    def _scores_from_buckets(buckets: list[list[torch.Tensor]], like: torch.Tensor) -> torch.Tensor:
        values = []
        for bucket in buckets:
            finite_bucket = [value for value in bucket if bool(torch.isfinite(value.detach()).item())]
            if finite_bucket:
                values.append(torch.logsumexp(torch.stack(finite_bucket), dim=0))
            else:
                values.append(like.new_full((), -torch.inf))
        return torch.stack(values)

    @staticmethod
    def _transition_log_probs(scores: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(scores)
        if not bool(finite.any().item()):
            return scores
        return scores - torch.logsumexp(scores[finite], dim=0)

    def forward(self, evidence_logits: torch.Tensor) -> SweepLayerOutput:
        if evidence_logits.ndim != 2 or evidence_logits.shape[-1] != len(EVIDENCE_NAMES):
            raise ValueError(f"evidence_logits must have shape L x {len(EVIDENCE_NAMES)}")

        length = int(evidence_logits.shape[0])
        alpha_by_pos = []
        initiation_buckets: list[list[torch.Tensor]] = [[] for _ in range(length)]
        termination_buckets: list[list[torch.Tensor]] = [[] for _ in range(length)]
        donor_buckets: list[list[torch.Tensor]] = [[] for _ in range(length)]
        acceptor_buckets: list[list[torch.Tensor]] = [[] for _ in range(length)]
        previous = evidence_logits.new_full((self.num_states,), -torch.inf)

        for t in range(length):
            start_logit, _stop_logit, _donor_logit, acceptor_logit, exon_logit, intron_logit = evidence_logits[t]
            g = t % 3
            current_buckets: list[list[torch.Tensor]] = [[] for _ in range(self.num_states)]

            if t == 0:
                current_buckets[idx_U5(g)].append(exon_logit)
                initiation_score = start_logit + exon_logit
                initiation_buckets[t].append(initiation_score)
                current_buckets[idx_C(g, 0)].append(initiation_score)
            else:
                previous_u5 = previous[idx_U5((t - 1) % 3)]
                current_buckets[idx_U5(g)].append(previous_u5 + exon_logit)
                initiation_score = previous_u5 + start_logit + exon_logit
                initiation_buckets[t].append(initiation_score)
                current_buckets[idx_C(g, 0)].append(initiation_score)

                previous_u3 = previous[idx_U3((t - 1) % 3)]
                current_buckets[idx_U3(g)].append(previous_u3 + exon_logit)

                if t >= 3:
                    stop_start = t - 3
                    previous_c_stop = previous[idx_C((t - 1) % 3, 2)]
                    stop_transition = previous_c_stop + evidence_logits[stop_start, 1] + exon_logit
                    termination_buckets[stop_start].append(stop_transition)
                    current_buckets[idx_U3(g)].append(stop_transition)

                previous_donor_logit = evidence_logits[t - 1, 2]
                for p in range(3):
                    previous_c = previous[idx_C((t - 1) % 3, p)]
                    next_p = (p + 1) % 3
                    current_buckets[idx_C(g, next_p)].append(previous_c + exon_logit)
                    donor_transition = previous_c + previous_donor_logit + intron_logit
                    donor_buckets[t - 1].append(donor_transition)
                    current_buckets[idx_I(g, next_p)].append(donor_transition)

                    previous_i = previous[idx_I((t - 1) % 3, p)]
                    current_buckets[idx_I(g, p)].append(previous_i + intron_logit)
                    acceptor_transition = previous_i + acceptor_logit + exon_logit
                    acceptor_buckets[t].append(acceptor_transition)
                    current_buckets[idx_C(g, p)].append(acceptor_transition)

            current = self._scores_from_buckets(current_buckets, evidence_logits)
            alpha_by_pos.append(current)
            previous = current

        alpha = torch.stack(alpha_by_pos, dim=0)
        state_log_probs = torch.log_softmax(alpha, dim=-1)
        initiation_scores = self._scores_from_buckets(initiation_buckets, evidence_logits)
        termination_scores = self._scores_from_buckets(termination_buckets, evidence_logits)
        donor_scores = self._scores_from_buckets(donor_buckets, evidence_logits)
        acceptor_scores = self._scores_from_buckets(acceptor_buckets, evidence_logits)
        return SweepLayerOutput(
            state_log_probs=state_log_probs,
            state_probs=state_log_probs.exp(),
            initiation_log_probs=self._transition_log_probs(initiation_scores),
            termination_log_probs=self._transition_log_probs(termination_scores),
            donor_log_probs=self._transition_log_probs(donor_scores),
            acceptor_log_probs=self._transition_log_probs(acceptor_scores),
        )


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
        if dna_one_hot.ndim != 2 or dna_one_hot.shape[-1] != 4:
            raise ValueError("dna_one_hot must have shape L x 4")
        features = dna_one_hot
        if self.use_structure_tracks:
            if structure_tracks is None:
                raise ValueError("structure_tracks are required when use_structure_tracks=True")
            features = torch.cat([dna_one_hot, structure_tracks], dim=-1)
        return self.net(features.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)


class GenomeSweepPhaseModel(nn.Module):
    def __init__(self, *, hidden_dim: int, use_structure_tracks: bool):
        super().__init__()
        self.evidence = SweepEvidenceModel(hidden_dim=hidden_dim, use_structure_tracks=use_structure_tracks)
        self.phase_layer = GenomeSweepPhaseLayer()

    def forward(self, dna_one_hot: torch.Tensor, structure_tracks: torch.Tensor | None = None) -> tuple[SweepLayerOutput, torch.Tensor]:
        evidence_logits = self.evidence(dna_one_hot, structure_tracks)
        return self.phase_layer(evidence_logits), evidence_logits


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
    gene: SweepGene,
    *,
    start_loss_weight: float,
    stop_loss_weight: float,
    donor_loss_weight: float,
    acceptor_loss_weight: float,
    evidence_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    phase_loss = F.nll_loss(output.state_log_probs, target_states)
    start_loss = -output.initiation_log_probs[gene.start_codon_start]
    stop_loss = -output.termination_log_probs[gene.stop_codon_start]
    if gene.donor_positions:
        donor_positions = torch.tensor(gene.donor_positions, dtype=torch.long, device=target_states.device)
        donor_loss = -output.donor_log_probs[donor_positions].mean()
    else:
        donor_loss = phase_loss.new_zeros(())
    if gene.acceptor_positions:
        acceptor_positions = torch.tensor(gene.acceptor_positions, dtype=torch.long, device=target_states.device)
        acceptor_loss = -output.acceptor_log_probs[acceptor_positions].mean()
    else:
        acceptor_loss = phase_loss.new_zeros(())
    evidence_targets = motif_targets(gene, target_states.device)
    evidence_loss = F.binary_cross_entropy_with_logits(evidence_logits, evidence_targets)
    total = (
        phase_loss
        + start_loss_weight * start_loss
        + stop_loss_weight * stop_loss
        + donor_loss_weight * donor_loss
        + acceptor_loss_weight * acceptor_loss
        + evidence_loss_weight * evidence_loss
    )
    return total, {
        "total": float(total.detach().item()),
        "phase": float(phase_loss.detach().item()),
        "start": float(start_loss.detach().item()),
        "stop": float(stop_loss.detach().item()),
        "donor": float(donor_loss.detach().item()),
        "acceptor": float(acceptor_loss.detach().item()),
        "evidence": float(evidence_loss.detach().item()),
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
def evaluate(model: GenomeSweepPhaseModel, args: argparse.Namespace, *, device: torch.device, seed: int, examples: int) -> dict[str, object]:
    rng = random.Random(seed)
    model.eval()
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
    for _ in range(examples):
        gene = make_gene(args, rng)
        dna_one_hot, structure_tracks, target = tensors_to_device(gene, device)
        output, evidence_logits = model(dna_one_hot, structure_tracks if args.use_structure_tracks else None)
        _loss, parts = compute_loss(
            output,
            evidence_logits,
            target,
            gene,
            start_loss_weight=args.start_loss_weight,
            stop_loss_weight=args.stop_loss_weight,
            donor_loss_weight=args.donor_loss_weight,
            acceptor_loss_weight=args.acceptor_loss_weight,
            evidence_loss_weight=args.evidence_loss_weight,
        )
        predicted = output.state_log_probs.argmax(dim=-1)
        matches = predicted == target
        for key, value in parts.items():
            sums[key] += value
        bases += int(target.numel())
        correct += int(matches.sum().item())
        exact += int(bool(matches.all().item()))
        start_exact += int(output.initiation_log_probs.argmax().item() == gene.start_codon_start)
        stop_exact += int(output.termination_log_probs.argmax().item() == gene.stop_codon_start)
        if gene.donor_positions:
            donor_exact += int(output.donor_log_probs.argmax().item() in gene.donor_positions)
        else:
            donor_exact += 1
        if gene.acceptor_positions:
            acceptor_exact += int(output.acceptor_log_probs.argmax().item() in gene.acceptor_positions)
        else:
            acceptor_exact += 1
        length_sum += len(gene.dna)
        intron_sum += len(gene.intron_lengths)
        state_acc = split_state_accuracy(predicted, target)
        for key, value in state_acc.items():
            if value == value:
                state_group_sums[key] += value
    return {
        "loss": sums["total"] / examples,
        "phase_loss": sums["phase"] / examples,
        "start_loss": sums["start"] / examples,
        "stop_loss": sums["stop"] / examples,
        "donor_loss": sums["donor"] / examples,
        "acceptor_loss": sums["acceptor"] / examples,
        "evidence_loss": sums["evidence"] / examples,
        "base_accuracy": correct / bases,
        "gene_exact": exact / examples,
        "start_exact": start_exact / examples,
        "stop_exact": stop_exact / examples,
        "donor_exact": donor_exact / examples,
        "acceptor_exact": acceptor_exact / examples,
        "mean_length": length_sum / examples,
        "mean_introns": intron_sum / examples,
        "state_groups": {key: value / examples for key, value in state_group_sums.items()},
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
    train_rng = random.Random(args.seed + 101)
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
        step_loss = None
        loss_parts = {key: 0.0 for key in ("total", "phase", "start", "stop", "donor", "acceptor", "evidence")}
        train_counts = {"bases": 0, "correct": 0, "exact": 0, "start": 0, "stop": 0, "donor": 0, "acceptor": 0}
        length_sum = 0
        intron_sum = 0
        group_sums = {"U5": 0.0, "C": 0.0, "I": 0.0, "U3": 0.0}

        for _ in range(args.examples_per_step):
            gene = make_gene(args, train_rng)
            dna_one_hot, structure_tracks, target = tensors_to_device(gene, device)
            output, evidence_logits = model(dna_one_hot, structure_tracks if args.use_structure_tracks else None)
            loss, parts = compute_loss(
                output,
                evidence_logits,
                target,
                gene,
                start_loss_weight=args.start_loss_weight,
                stop_loss_weight=args.stop_loss_weight,
                donor_loss_weight=args.donor_loss_weight,
                acceptor_loss_weight=args.acceptor_loss_weight,
                evidence_loss_weight=args.evidence_loss_weight,
            )
            scaled = loss / args.examples_per_step
            step_loss = scaled if step_loss is None else step_loss + scaled
            with torch.no_grad():
                predicted = output.state_log_probs.argmax(dim=-1)
                matches = predicted == target
                train_counts["bases"] += int(target.numel())
                train_counts["correct"] += int(matches.sum().item())
                train_counts["exact"] += int(bool(matches.all().item()))
                train_counts["start"] += int(output.initiation_log_probs.argmax().item() == gene.start_codon_start)
                train_counts["stop"] += int(output.termination_log_probs.argmax().item() == gene.stop_codon_start)
                train_counts["donor"] += int(output.donor_log_probs.argmax().item() in gene.donor_positions) if gene.donor_positions else 1
                train_counts["acceptor"] += (
                    int(output.acceptor_log_probs.argmax().item() in gene.acceptor_positions) if gene.acceptor_positions else 1
                )
                state_acc = split_state_accuracy(predicted, target)
                for key, value in state_acc.items():
                    if value == value:
                        group_sums[key] += value
                length_sum += len(gene.dna)
                intron_sum += len(gene.intron_lengths)
            for key, value in parts.items():
                loss_parts[key] += value / args.examples_per_step

        assert step_loss is not None
        step_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        should_print = step == 1 or step % args.print_every == 0 or step == args.steps
        should_validate = step == 1 or step == args.steps or (args.validate_every > 0 and step % args.validate_every == 0)
        should_save = step % args.checkpoint_every == 0 or step == args.steps
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
