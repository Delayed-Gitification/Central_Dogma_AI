import torch

from central_dogma_ai.structured_phase import (
    C0_STATE,
    C1_STATE,
    C2_STATE,
    N_STATE,
    T_STATE,
    StructuredTranslationPhaseModel,
    generate_single_exon_phase_gene,
    phase_nll_loss,
)


def test_single_exon_synthetic_truth_has_utr_cds_and_post_stop_states():
    gene = generate_single_exon_phase_gene(
        utr5_length=9,
        coding_codons=5,
        utr3_length=7,
        seed=7,
    )

    assert gene.dna[gene.start_codon_start : gene.start_codon_start + 3] == "ATG"
    assert gene.dna[gene.stop_codon_start : gene.stop_codon_start + 3] in {"TAA", "TAG", "TGA"}
    assert gene.target_states[: gene.start_codon_start].tolist() == [N_STATE] * gene.start_codon_start
    assert gene.target_states[gene.start_codon_start : gene.start_codon_start + 6].tolist() == [
        C0_STATE,
        C1_STATE,
        C2_STATE,
        C0_STATE,
        C1_STATE,
        C2_STATE,
    ]
    assert gene.target_states[gene.stop_codon_start : gene.stop_codon_start + 3].tolist() == [
        C0_STATE,
        C1_STATE,
        C2_STATE,
    ]
    assert gene.target_states[gene.stop_codon_start + 3 :].tolist() == [T_STATE] * gene.utr3_length


def test_textbook_initialized_model_recovers_single_exon_phase_and_codons():
    gene = generate_single_exon_phase_gene(
        utr5_length=11,
        coding_codons=6,
        utr3_length=8,
        seed=13,
    )
    model = StructuredTranslationPhaseModel()
    model.feature_extractor.initialize_textbook_motifs(strength=6.0, bias=-12.0)

    output = model(gene.dna_one_hot, gene.paths)
    predicted = output.state_log_probs.argmax(dim=-1)

    assert predicted.tolist() == gene.target_states.tolist()
    assert int(output.initiation_log_probs.argmax().item()) == gene.start_codon_start
    assert int(output.termination_log_probs.argmax().item()) == gene.stop_codon_start
    assert output.state_probs.shape == (len(gene.dna), 5)
    assert torch.allclose(output.state_probs.sum(dim=-1), torch.ones(len(gene.dna)), atol=1e-5)


def test_phase_loss_backpropagates_to_dna_feature_extractor():
    gene = generate_single_exon_phase_gene(
        utr5_length=10,
        coding_codons=5,
        utr3_length=6,
        seed=21,
    )
    model = StructuredTranslationPhaseModel()
    output = model(gene.dna_one_hot, gene.paths)
    loss = phase_nll_loss(output, gene.target_states)
    loss.backward()

    assert model.feature_extractor.start_weight.grad is not None
    assert model.feature_extractor.stop_weight.grad is not None
    assert torch.isfinite(model.feature_extractor.start_weight.grad).all()
    assert torch.isfinite(model.feature_extractor.stop_weight.grad).all()
    assert model.feature_extractor.start_weight.grad.abs().sum().item() > 0
    assert model.feature_extractor.stop_weight.grad.abs().sum().item() > 0
