import torch

from central_dogma_ai.biology import translate_sequence
from central_dogma_ai.structured_phase import (
    C0_STATE,
    C1_STATE,
    C2_STATE,
    N_STATE,
    T_STATE,
    SplicePath,
    StructuredPhaseLayer,
    StructuredTranslationPhaseModel,
    generate_multiexon_phase_gene,
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


def test_synthetic_generator_builds_protein_then_aligned_dna_phase_matrix():
    gene = generate_multiexon_phase_gene(
        utr5_length=13,
        coding_codons=8,
        utr3_length=9,
        exon_lengths=(9, 11, 26),
        min_intron_length=10,
        max_intron_length=10,
        seed=17,
    )
    path = gene.paths[0].genomic_indices.tolist()
    path_set = set(path)
    spliced = "".join(gene.dna[index] for index in path)
    spliced_labels = gene.target_states[path].tolist()
    cds = spliced[gene.utr5_length : gene.utr5_length + 8 * 3]

    assert cds[:3] == "ATG"
    assert cds[-3:] in {"TAA", "TAG", "TGA"}
    assert translate_sequence(cds).startswith("M")
    assert translate_sequence(cds).endswith("*")
    assert spliced_labels[: gene.utr5_length] == [N_STATE] * gene.utr5_length
    assert spliced_labels[gene.utr5_length : gene.utr5_length + 6] == [
        C0_STATE,
        C1_STATE,
        C2_STATE,
        C0_STATE,
        C1_STATE,
        C2_STATE,
    ]
    assert spliced_labels[gene.utr5_length + 8 * 3 :] == [T_STATE] * gene.utr3_length

    intron_positions = [index for index in range(len(gene.dna)) if index not in path_set]
    assert intron_positions
    assert all(gene.target_states[index].item() == N_STATE for index in intron_positions)
    for intron_start in [index for index in intron_positions if index - 1 in path_set]:
        intron = gene.dna[intron_start : intron_start + 10]
        assert intron.startswith("GT")
        assert intron.endswith("AG")


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


def test_multiexon_synthetic_truth_skips_introns_and_preserves_phase():
    gene = generate_multiexon_phase_gene(
        utr5_length=6,
        coding_codons=6,
        utr3_length=5,
        exon_lengths=(10, 7, 12),
        min_intron_length=8,
        max_intron_length=8,
        seed=31,
    )
    path_indices = gene.paths[0].genomic_indices.tolist()
    path_set = set(path_indices)

    assert gene.dna[gene.start_codon_start : gene.start_codon_start + 3] == "ATG"
    assert gene.dna[gene.stop_codon_start : gene.stop_codon_start + 3] in {"TAA", "TAG", "TGA"}
    assert gene.exon_lengths == (10, 7, 12)
    assert gene.intron_lengths == (8, 8)
    assert all(gene.target_states[index].item() == N_STATE for index in range(len(gene.dna)) if index not in path_set)

    start_offset = gene.utr5_length
    for transcript_offset, genomic_index in enumerate(path_indices):
        if transcript_offset < start_offset:
            assert gene.target_states[genomic_index].item() == N_STATE
        elif transcript_offset <= gene.utr5_length + 6 * 3 - 1:
            assert gene.target_states[genomic_index].item() == C0_STATE + ((transcript_offset - start_offset) % 3)
        else:
            assert gene.target_states[genomic_index].item() == T_STATE


def test_textbook_initialized_model_recovers_multiexon_phase_across_introns():
    gene = generate_multiexon_phase_gene(
        utr5_length=6,
        coding_codons=6,
        utr3_length=5,
        exon_lengths=(10, 7, 12),
        min_intron_length=8,
        max_intron_length=8,
        seed=41,
    )
    model = StructuredTranslationPhaseModel()
    model.feature_extractor.initialize_textbook_motifs(strength=6.0, bias=-12.0)

    output = model(gene.dna_one_hot, gene.paths)
    predicted = output.state_log_probs.argmax(dim=-1)

    assert predicted.tolist() == gene.target_states.tolist()
    assert int(output.initiation_log_probs.argmax().item()) == gene.start_codon_start
    assert int(output.termination_log_probs.argmax().item()) == gene.stop_codon_start


def test_path_aware_model_recovers_split_start_and_stop_codons():
    gene = generate_multiexon_phase_gene(
        utr5_length=7,
        coding_codons=5,
        utr3_length=6,
        exon_lengths=(8, 12, 8),
        min_intron_length=8,
        max_intron_length=8,
        seed=51,
    )
    path = gene.paths[0].genomic_indices.tolist()
    assert path[gene.utr5_length] == gene.start_codon_start
    assert gene.dna[gene.start_codon_start : gene.start_codon_start + 3] != "ATG"
    assert gene.dna[gene.stop_codon_start : gene.stop_codon_start + 3] not in {"TAA", "TAG", "TGA"}

    model = StructuredTranslationPhaseModel(path_aware_codons=True)
    model.feature_extractor.initialize_textbook_motifs(strength=6.0, bias=-12.0)

    output = model(gene.dna_one_hot, gene.paths)
    predicted = output.state_log_probs.argmax(dim=-1)

    assert predicted.tolist() == gene.target_states.tolist()
    assert int(output.initiation_log_probs.argmax().item()) == gene.start_codon_start
    assert int(output.termination_log_probs.argmax().item()) == gene.stop_codon_start


def test_splice_path_log_weights_receive_gradients_through_logsumexp_merge():
    edge_logits = torch.zeros(2, requires_grad=True)
    paths = (
        SplicePath(torch.arange(9, dtype=torch.long), log_weight=edge_logits[0]),
        SplicePath(torch.tensor([0, 1, 2, 6, 7, 8], dtype=torch.long), log_weight=edge_logits[1]),
    )
    start_logits = torch.zeros(9)
    stop_logits = torch.zeros(9)

    output = StructuredPhaseLayer()(start_logits=start_logits, stop_logits=stop_logits, paths=paths)
    output.log_partition.backward()

    assert edge_logits.grad is not None
    assert torch.isfinite(edge_logits.grad).all()
    assert (edge_logits.grad > 0).all()


def test_variable_numbers_of_exons_and_introns_have_valid_targets():
    for exon_count in range(1, 6):
        gene = generate_multiexon_phase_gene(
            utr5_length=7,
            coding_codons=7,
            utr3_length=6,
            exon_count=exon_count,
            min_exon_length=3,
            min_intron_length=6,
            max_intron_length=11,
            seed=100 + exon_count,
        )
        path_indices = gene.paths[0].genomic_indices
        path_set = set(path_indices.tolist())

        assert len(gene.exon_lengths) == exon_count
        assert len(gene.intron_lengths) == max(0, exon_count - 1)
        assert int(path_indices.numel()) == sum(gene.exon_lengths)
        assert gene.target_states.shape == (len(gene.dna),)
        assert all(gene.target_states[index].item() == N_STATE for index in range(len(gene.dna)) if index not in path_set)

        model = StructuredTranslationPhaseModel()
        output = model(gene.dna_one_hot, gene.paths)
        assert torch.isfinite(output.state_log_probs[path_indices]).any()
        assert torch.allclose(output.state_probs.sum(dim=-1), torch.ones(len(gene.dna)), atol=1e-5)
