import pytest

torch = pytest.importorskip("torch")

from central_dogma_ai.biology import AA_TO_INDEX
from central_dogma_ai.synthetic import SyntheticGeneConfig, generate_examples
from central_dogma_ai.torch_model import (
    TypedSpliceTransducer,
    fixed_translate_codons,
    gather_transcript_bases,
    path_indices_from_exon_mask,
    project_spliced_codon_bases,
    protein_to_target,
)


def _padded_batch(examples):
    max_length = max(len(example.dna_one_hot) for example in examples)
    dna_rows = []
    mask_rows = []
    track_rows = []
    for example in examples:
        padding = max_length - len(example.dna_one_hot)
        dna_rows.append(example.dna_one_hot + [[0.0, 0.0, 0.0, 0.0]] * padding)
        mask_rows.append(example.exon_mask + [0] * padding)
        track_rows.append(example.splice_tracks + [[0.0, 0.0, 0.0]] * padding)
    return (
        torch.tensor(dna_rows, dtype=torch.float32),
        torch.tensor(mask_rows, dtype=torch.bool),
        torch.tensor(track_rows, dtype=torch.float32),
    )


def test_path_indices_gather_spliced_transcript_order():
    dna = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ]
    )
    mask = torch.tensor([[1, 1, 0, 0, 1, 1]], dtype=torch.bool)

    path_indices, path_mask = path_indices_from_exon_mask(mask)
    gathered = gather_transcript_bases(dna, path_indices, path_mask)

    assert path_indices.tolist() == [[0, 1, 4, 5, 0, 0]]
    assert path_mask.tolist() == [[True, True, True, True, False, False]]
    assert gathered[0, :4].argmax(dim=-1).tolist() == [0, 1, 0, 1]


def test_fixed_codon_table_translates_one_hot_codons():
    codon_bases = torch.zeros((1, 3, 3, 4))
    codon_bases[0, 0, 0, 0] = 1.0
    codon_bases[0, 0, 1, 3] = 1.0
    codon_bases[0, 0, 2, 2] = 1.0
    codon_bases[0, 1, 0, 2] = 1.0
    codon_bases[0, 1, 1, 0] = 1.0
    codon_bases[0, 1, 2, 0] = 1.0
    codon_bases[0, 2, 0, 3] = 1.0
    codon_bases[0, 2, 1, 0] = 1.0
    codon_bases[0, 2, 2, 0] = 1.0

    amino_acid_probs = fixed_translate_codons(codon_bases)

    assert amino_acid_probs.argmax(dim=-1).tolist() == [[AA_TO_INDEX["M"], AA_TO_INDEX["E"], AA_TO_INDEX["*"]]]


def test_typed_transducer_outputs_exact_amino_acids_from_given_path():
    config = SyntheticGeneConfig(amino_acid_codons=8, min_exons=3, max_exons=3)
    examples = generate_examples(2, config=config, seed=3)
    dna, mask, tracks = _padded_batch(examples)
    model = TypedSpliceTransducer.build(hidden_dim=32, layers=1)

    output = model(dna, splice_tracks=tracks, exon_mask=mask)
    predictions = output["amino_acid_probs"].argmax(dim=-1)
    target = protein_to_target([example.protein for example in examples])

    assert torch.equal(predictions, target)
    assert output["consequences"]["terminal_stop"].round().tolist() == [1.0, 1.0]
    assert output["consequences"]["premature_stop"].round().tolist() == [0.0, 0.0]


def test_project_spliced_codon_bases_preserves_batch_shape():
    config = SyntheticGeneConfig(amino_acid_codons=6, min_exons=2, max_exons=2)
    examples = generate_examples(3, config=config, seed=9)
    dna, mask, _tracks = _padded_batch(examples)

    codon_bases, codon_mask = project_spliced_codon_bases(dna, mask)

    assert codon_bases.shape == (3, 7, 3, 4)
    assert codon_mask.all()