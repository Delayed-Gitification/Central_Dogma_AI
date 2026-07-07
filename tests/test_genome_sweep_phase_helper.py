import inspect
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_genome_sweep_phase_helper as sweep  # noqa: E402


def test_state_index_helpers_are_24_unique_states():
    indices = [sweep.idx_U5(g) for g in range(3)]
    indices += [sweep.idx_C(g, p) for g in range(3) for p in range(3)]
    indices += [sweep.idx_I(g, p) for g in range(3) for p in range(3)]
    indices += [sweep.idx_U3(g) for g in range(3)]
    assert len(indices) == 24
    assert sorted(indices) == list(range(24))


def test_single_exon_labels_u5_coding_u3():
    gene = sweep.generate_sweep_gene(
        utr5_length=12,
        coding_codons=8,
        utr3_length=10,
        exon_count=1,
        seed=10,
    )
    assert not gene.donor_positions
    assert not gene.acceptor_positions
    assert gene.start_codon_start == 12
    assert gene.dna[gene.start_codon_start : gene.start_codon_start + 3] == "ATG"
    for pos in range(gene.start_codon_start):
        assert int(gene.target_states[pos]) == sweep.idx_U5(pos % 3)
    for offset in range((8 * 3)):
        pos = gene.start_codon_start + offset
        assert int(gene.target_states[pos]) == sweep.idx_C(pos % 3, offset % 3)
    for pos in range(gene.stop_codon_start + 3, len(gene.dna)):
        assert int(gene.target_states[pos]) == sweep.idx_U3(pos % 3)


def test_intron_carry_preserves_mature_phase_while_genomic_track_advances():
    gene = sweep.generate_sweep_gene(
        utr5_length=9,
        coding_codons=18,
        utr3_length=9,
        exon_count=2,
        min_exon_length=6,
        min_intron_length=5,
        max_intron_length=5,
        seed=22,
    )
    donor = gene.donor_positions[0]
    acceptor = gene.acceptor_positions[0]
    carried_state = int(gene.target_states[donor + 1])
    carried_phase = (carried_state - 12) % 3
    for pos in range(donor + 1, acceptor):
        assert int(gene.target_states[pos]) == sweep.idx_I(pos % 3, carried_phase)
    assert int(gene.target_states[acceptor]) == sweep.idx_C(acceptor % 3, carried_phase)


def test_non_three_intron_shifts_genomic_track_not_mature_phase():
    kwargs = dict(
        utr5_length=9,
        coding_codons=18,
        utr3_length=9,
        exon_count=2,
        min_exon_length=6,
        seed=33,
    )
    mod0 = sweep.generate_sweep_gene(min_intron_length=6, max_intron_length=6, **kwargs)
    mod2 = sweep.generate_sweep_gene(min_intron_length=5, max_intron_length=5, **kwargs)
    acceptor0 = mod0.acceptor_positions[0]
    acceptor2 = mod2.acceptor_positions[0]
    state0 = int(mod0.target_states[acceptor0])
    state2 = int(mod2.target_states[acceptor2])
    phase0 = (state0 - 3) % 3
    phase2 = (state2 - 3) % 3
    track0 = (state0 - 3) // 3
    track2 = (state2 - 3) // 3
    assert phase0 == phase2
    assert track0 != track2


def test_genome_sweep_layer_has_no_path_argument():
    signature = inspect.signature(sweep.GenomeSweepPhaseLayer.forward)
    assert list(signature.parameters) == ["self", "evidence_logits"]


def test_gradients_flow_through_evidence_model_and_logits():
    gene = sweep.generate_sweep_gene(
        utr5_length=12,
        coding_codons=10,
        utr3_length=12,
        exon_count=2,
        min_exon_length=6,
        min_intron_length=6,
        max_intron_length=6,
        seed=44,
    )
    model = sweep.GenomeSweepPhaseModel(hidden_dim=16, use_structure_tracks=True)
    output, logits = model(gene.dna_one_hot, gene.structure_tracks)
    logits.retain_grad()
    loss, _parts = sweep.compute_loss(
        output,
        logits,
        gene.target_states,
        gene,
        start_loss_weight=0.25,
        stop_loss_weight=0.25,
        donor_loss_weight=0.1,
        acceptor_loss_weight=0.1,
        evidence_loss_weight=0.1,
    )
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0
    param_grad = sum(float(param.grad.abs().sum()) for param in model.parameters() if param.grad is not None)
    assert param_grad > 0
