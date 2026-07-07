import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_dense_transition_phase_helper as dense  # noqa: E402


def test_all_allowed_rows_have_exit():
    edge = dense.EDGE_TYPE_MATRIX
    assert edge.shape == (24, 24)
    assert ((edge >= 0).sum(dim=-1) > 0).all()


def test_row_softmax_sums_to_one():
    layer = dense.DenseTransitionPhaseLayer()
    logits = torch.zeros(2, 5, len(dense.TRANSITION_TYPES))
    transition_log_probs = layer.transition_log_probs(logits)
    row_sums = transition_log_probs.exp().sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_coding_phase_cycle():
    edge = dense.EDGE_TYPE_MATRIX
    coding = dense.TYPE_TO_INDEX["coding"]
    for g in range(3):
        assert int(edge[dense.idx_C(g, 0), dense.idx_C(g, 1)]) == coding
        assert int(edge[dense.idx_C(g, 1), dense.idx_C(g, 2)]) == coding
        assert int(edge[dense.idx_C(g, 2), dense.idx_C(g, 0)]) == coding


def test_intron_freezes_codon_phase_and_advances_row():
    edge = dense.EDGE_TYPE_MATRIX
    intron = dense.TYPE_TO_INDEX["intron"]
    assert int(edge[dense.idx_I(1, 2), dense.idx_I(2, 2)]) == intron
    assert int(edge[dense.idx_I(2, 2), dense.idx_I(0, 2)]) == intron
    assert int(edge[dense.idx_I(0, 2), dense.idx_I(1, 2)]) == intron


def test_weak_start_splits_mass():
    layer = dense.DenseTransitionPhaseLayer()
    logits = torch.full((1, 1, len(dense.TRANSITION_TYPES)), -8.0)
    logits[:, :, dense.TYPE_TO_INDEX["u5"]] = 0.0
    logits[:, :, dense.TYPE_TO_INDEX["start"]] = 0.0
    transition = layer.transition_log_probs(logits).exp()[0, 0]
    u_to_u = transition[dense.idx_U(0), dense.idx_U(1)]
    u_to_c = transition[dense.idx_U(0), dense.idx_C(0, 0)]
    assert torch.allclose(u_to_u, torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(u_to_c, torch.tensor(0.5), atol=1e-6)


def test_dense_layer_gradients_flow():
    gene = dense.generate_dense_gene(
        utr5_length=12,
        coding_codons=10,
        utr3_length=12,
        exon_count=2,
        min_exon_length=6,
        min_intron_length=6,
        max_intron_length=6,
        seed=44,
    )
    model = dense.DenseTransitionPhaseModel(
        hidden_dim=16,
        conv_layers=2,
        use_splice_tracks=True,
        materialize_transitions=False,
    )
    output, logits = model(gene.dna_one_hot, gene.splice_tracks)
    logits.retain_grad()
    loss, _parts = dense.compute_loss(
        output,
        logits,
        gene.target_states,
        gene.evidence_targets,
        gene,
        evidence_loss_weight=0.1,
        start_loss_weight=0.25,
        stop_loss_weight=0.25,
        donor_loss_weight=0.1,
        acceptor_loss_weight=0.1,
    )
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0
    param_grad = sum(float(param.grad.abs().sum()) for param in model.parameters() if param.grad is not None)
    assert param_grad > 0
