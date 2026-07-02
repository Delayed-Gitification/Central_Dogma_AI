"""Command-line demo for synthetic splice-aware translation examples."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from central_dogma_ai.synthetic import SyntheticGeneConfig, make_random_splice_program
from central_dogma_ai.splicing import translate_all


def _program_to_dict(program):
    translations = translate_all(program)
    return {
        "name": program.name,
        "genomic_length": len(program.dna),
        "exons": [
            {"name": exon.name, "start": exon.start, "end": exon.end, "length": exon.length}
            for exon in program.exons
        ],
        "isoforms": [
            {
                "name": isoform.name,
                "exon_path": list(isoform.exon_names),
                "intended": isoform.intended,
                "protein": translations[isoform.name].protein,
                "frame_valid": translations[isoform.name].frame_valid,
                "first_stop_codon_index": translations[isoform.name].first_stop_codon_index,
                "has_premature_stop": translations[isoform.name].has_premature_stop,
                "nmd_risk": translations[isoform.name].nmd_risk,
                "exon_phases": [asdict(phase) for phase in translations[isoform.name].exon_phases],
            }
            for isoform in program.isoforms
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amino-acid-codons", type=int, default=24)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a compact text summary")
    args = parser.parse_args(argv)

    import random

    rng = random.Random(args.seed)
    config = SyntheticGeneConfig(amino_acid_codons=args.amino_acid_codons)
    payloads = []
    for index in range(args.examples):
        program = make_random_splice_program(config=config, rng=rng, name=f"synthetic_{index}")
        payloads.append(_program_to_dict(program))

    if args.json:
        print(json.dumps(payloads, indent=2))
        return 0

    for payload in payloads:
        print(f"{payload['name']}: genomic_length={payload['genomic_length']}")
        print("  exons:")
        for exon in payload["exons"]:
            print(f"    {exon['name']}: {exon['start']}-{exon['end']} ({exon['length']} nt)")
        print("  isoforms:")
        for isoform in payload["isoforms"]:
            print(
                "    "
                f"{isoform['name']}: path={'-'.join(isoform['exon_path'])}, "
                f"protein_len={len(isoform['protein'])}, "
                f"frame_valid={isoform['frame_valid']}, "
                f"premature_stop={isoform['has_premature_stop']}, "
                f"nmd_risk={isoform['nmd_risk']}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())