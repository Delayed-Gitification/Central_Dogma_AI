from central_dogma_ai.biology import translate_sequence
from central_dogma_ai.synthetic import SyntheticGeneConfig, generate_examples, make_random_splice_program
from central_dogma_ai.splicing import assemble_transcript, translate_isoform


def test_productive_synthetic_isoform_is_exact_coding_sequence():
    config = SyntheticGeneConfig(amino_acid_codons=12, min_exons=3, max_exons=3)
    program = make_random_splice_program(config=config, name="example")

    transcript = assemble_transcript(program, "productive")
    result = translate_isoform(program, "productive")

    assert len(transcript) == 39
    assert result.frame_valid
    assert result.has_terminal_stop
    assert not result.has_premature_stop
    assert result.protein == translate_sequence(transcript)


def test_training_example_shapes_are_consistent():
    config = SyntheticGeneConfig(amino_acid_codons=10, min_exons=2, max_exons=2)
    examples = generate_examples(3, config=config, seed=11)

    for example in examples:
        assert len(example.dna_one_hot) == len(example.program.dna)
        assert len(example.exon_mask) == len(example.program.dna)
        assert len(example.splice_tracks) == len(example.program.dna)
        assert len(example.genomic_path) == sum(example.exon_mask)
        assert sum(example.exon_mask) == len(example.protein) * 3
        assert len(example.amino_acid_one_hot) == len(example.protein)
        assert example.frame_valid