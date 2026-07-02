from central_dogma_ai.splicing import Exon, Isoform, SpliceProgram, exon_mask, exon_phases, translate_isoform


def test_split_codon_across_exon_boundary_translates_exactly():
    program = SpliceProgram(
        dna="ATGGAACTGTAA",
        exons=(Exon("E1", 0, 4), Exon("E2", 4, 12)),
        isoforms=(Isoform("productive", ("E1", "E2")),),
    )

    result = translate_isoform(program, "productive")

    assert result.codons == ("ATG", "GAA", "CTG", "TAA")
    assert result.protein == "MEL*"
    assert result.frame_valid
    assert result.has_terminal_stop
    assert not result.has_premature_stop


def test_exon_phases_track_frame_across_boundaries():
    program = SpliceProgram(
        dna="ATGGAACTGTAA",
        exons=(Exon("E1", 0, 4), Exon("E2", 4, 12)),
        isoforms=(Isoform("productive", ("E1", "E2")),),
    )

    phases = exon_phases(program, "productive")

    assert [(phase.phase_before, phase.phase_after) for phase in phases] == [(0, 1), (1, 0)]


def test_premature_stop_and_nmd_like_risk_are_flagged():
    dna = "ATGTAA" + "GCT" * 30
    program = SpliceProgram(
        dna=dna,
        exons=(Exon("E1", 0, 66), Exon("E2", 66, len(dna))),
        isoforms=(Isoform("ptc", ("E1", "E2")),),
    )

    result = translate_isoform(program, "ptc")

    assert result.protein.startswith("M*")
    assert result.has_premature_stop
    assert result.nmd_risk


def test_exon_mask_marks_included_genomic_positions():
    program = SpliceProgram(
        dna="AAACCCGGGTTT",
        exons=(Exon("E1", 0, 3), Exon("E2", 6, 9)),
        isoforms=(Isoform("spliced", ("E1", "E2")),),
    )

    assert exon_mask(program, "spliced") == [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]