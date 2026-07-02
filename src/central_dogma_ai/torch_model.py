"""Optional PyTorch components for the typed splice-transducer benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

from central_dogma_ai.biology import AMINO_ACIDS, AA_TO_INDEX, CODON_TABLE, CODON_VOCAB

if TYPE_CHECKING:
    import torch


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the optional ML dependencies with: python -m pip install -e '.[ml]'") from exc
    return torch


def _round_up_to_codon_length(length: int) -> int:
    remainder = length % 3
    return length if remainder == 0 else length + (3 - remainder)


def path_indices_from_exon_mask(exon_mask: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Convert B x L exon masks into padded transcript-order genomic pointers."""

    torch = require_torch()
    if exon_mask.ndim != 2:
        raise ValueError("exon_mask must have shape B x L")
    path_lengths = exon_mask.bool().sum(dim=1)
    padded_length = _round_up_to_codon_length(int(path_lengths.max().item()))
    path_indices = torch.zeros((exon_mask.shape[0], padded_length), dtype=torch.long, device=exon_mask.device)
    path_mask = torch.zeros((exon_mask.shape[0], padded_length), dtype=torch.bool, device=exon_mask.device)
    for batch_index in range(exon_mask.shape[0]):
        selected_indices = torch.nonzero(exon_mask[batch_index].bool(), as_tuple=False).flatten()
        selected_length = selected_indices.numel()
        path_indices[batch_index, :selected_length] = selected_indices
        path_mask[batch_index, :selected_length] = True
    return path_indices, path_mask


def gather_transcript_bases(
    dna_one_hot: "torch.Tensor",
    path_indices: "torch.Tensor",
    path_mask: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Gather genomic one-hots into transcript order using pointer indices."""

    if dna_one_hot.ndim != 3 or dna_one_hot.shape[-1] != 4:
        raise ValueError("dna_one_hot must have shape B x L x 4")
    if path_indices.ndim != 2 or path_indices.shape[0] != dna_one_hot.shape[0]:
        raise ValueError("path_indices must have shape B x T")
    expanded_indices = path_indices.unsqueeze(-1).expand(-1, -1, dna_one_hot.shape[-1])
    transcript_bases = dna_one_hot.gather(dim=1, index=expanded_indices)
    if path_mask is not None:
        transcript_bases = transcript_bases * path_mask.unsqueeze(-1).to(transcript_bases.dtype)
    return transcript_bases


def codon_buffer(
    transcript_bases: "torch.Tensor",
    transcript_mask: "torch.Tensor | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Group transcript bases into codons while carrying phase across junctions."""

    torch = require_torch()
    if transcript_bases.ndim != 3 or transcript_bases.shape[-1] != 4:
        raise ValueError("transcript_bases must have shape B x T x 4")
    if transcript_bases.shape[1] % 3 != 0:
        raise ValueError("Transcript length must be padded to a multiple of three")
    codon_count = transcript_bases.shape[1] // 3
    codon_bases = transcript_bases.reshape(transcript_bases.shape[0], codon_count, 3, 4)
    if transcript_mask is None:
        codon_mask = torch.ones(
            (transcript_bases.shape[0], codon_count),
            dtype=torch.bool,
            device=transcript_bases.device,
        )
    else:
        if transcript_mask.shape != transcript_bases.shape[:2]:
            raise ValueError("transcript_mask must have shape B x T")
        codon_mask = transcript_mask.reshape(transcript_mask.shape[0], codon_count, 3).all(dim=-1)
    return codon_bases, codon_mask


def codon_distribution(codon_bases: "torch.Tensor") -> "torch.Tensor":
    """Turn three base-probability rows into a 64-codon distribution."""

    if codon_bases.ndim != 4 or codon_bases.shape[-2:] != (3, 4):
        raise ValueError("codon_bases must have shape B x C x 3 x 4")
    first_base = codon_bases[:, :, 0, :]
    second_base = codon_bases[:, :, 1, :]
    third_base = codon_bases[:, :, 2, :]
    distribution = (
        first_base[:, :, :, None, None]
        * second_base[:, :, None, :, None]
        * third_base[:, :, None, None, :]
    )
    return distribution.reshape(codon_bases.shape[0], codon_bases.shape[1], len(CODON_VOCAB))


def codon_to_amino_acid_matrix(device=None, dtype=None):
    """Return a fixed 64 x 21 codon-table matrix."""

    torch = require_torch()
    matrix = torch.zeros((len(CODON_VOCAB), len(AMINO_ACIDS)), device=device, dtype=dtype or torch.float32)
    for codon_index, codon in enumerate(CODON_VOCAB):
        amino_acid = CODON_TABLE[codon]
        matrix[codon_index, AA_TO_INDEX[amino_acid]] = 1.0
    return matrix


def fixed_translate_codons(codon_bases: "torch.Tensor") -> "torch.Tensor":
    """Translate codon base probabilities through the frozen genetic code."""

    codon_probs = codon_distribution(codon_bases)
    table = codon_to_amino_acid_matrix(device=codon_bases.device, dtype=codon_bases.dtype)
    return codon_probs @ table


def project_spliced_codons(dna_one_hot: "torch.Tensor", exon_mask: "torch.Tensor") -> "torch.Tensor":
    """Project fixed-mask genomic one-hots into spliced codon one-hots.

    Args:
        dna_one_hot: Float tensor with shape B x L x 4.
        exon_mask: Boolean/int/float tensor with shape B x L. Nonzero bases are
            included in the transcript path.

    Returns:
        Float tensor with shape B x C x 12, where each row is three concatenated
        base one-hots. All batch items must have the same number of included
        bases and that count must be divisible by three.
    """

    torch = require_torch()
    if dna_one_hot.ndim != 3 or dna_one_hot.shape[-1] != 4:
        raise ValueError("dna_one_hot must have shape B x L x 4")
    if exon_mask.shape != dna_one_hot.shape[:2]:
        raise ValueError("exon_mask must have shape B x L")

    codon_rows = []
    codon_count: int | None = None
    for batch_index in range(dna_one_hot.shape[0]):
        selected = dna_one_hot[batch_index][exon_mask[batch_index].bool()]
        if selected.shape[0] % 3 != 0:
            raise ValueError("Included transcript length must be divisible by three")
        current_codon_count = selected.shape[0] // 3
        if codon_count is None:
            codon_count = current_codon_count
        elif current_codon_count != codon_count:
            raise ValueError("All batch items must have the same spliced codon count")
        codon_rows.append(selected.reshape(current_codon_count, 12))
    return torch.stack(codon_rows, dim=0)


def project_spliced_codon_bases(
    dna_one_hot: "torch.Tensor",
    exon_mask: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Project masks to B x C x 3 x 4 codon bases plus a codon-valid mask."""

    path_indices, path_mask = path_indices_from_exon_mask(exon_mask)
    transcript_bases = gather_transcript_bases(dna_one_hot, path_indices, path_mask=path_mask)
    return codon_buffer(transcript_bases, transcript_mask=path_mask)


def protein_to_target(proteins: list[str]) -> "torch.Tensor":
    """Encode equal-length protein strings as class indices."""

    torch = require_torch()
    if not proteins:
        raise ValueError("proteins must not be empty")
    length = len(proteins[0])
    if any(len(protein) != length for protein in proteins):
        raise ValueError("All protein strings must have equal length")
    rows = []
    for protein in proteins:
        rows.append([AA_TO_INDEX[amino_acid] for amino_acid in protein])
    return torch.tensor(rows, dtype=torch.long)


class CodonAminoAcidHead:
    """Factory for a small codon-to-amino-acid classifier.

    Defined as a factory rather than importing torch at module import time, so
    the non-ML package remains dependency-free.
    """

    @staticmethod
    def build(hidden_dim: int = 64):
        torch = require_torch()
        nn = torch.nn

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(12, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, len(AMINO_ACIDS)),
                )

            def forward(self, codon_one_hot):
                return self.net(codon_one_hot)

        return _Head()


class FixedCodonTableLayer:
    """Factory for the frozen 64-codon to amino-acid layer."""

    @staticmethod
    def build():
        torch = require_torch()
        nn = torch.nn

        class _FixedCodonTable(nn.Module):
            def forward(self, codon_bases):
                return fixed_translate_codons(codon_bases)

        return _FixedCodonTable()


class CodonBufferLayer:
    """Factory for the explicit phase/codon bottleneck."""

    @staticmethod
    def build():
        torch = require_torch()
        nn = torch.nn

        class _CodonBuffer(nn.Module):
            def forward(self, transcript_bases, transcript_mask=None):
                return codon_buffer(transcript_bases, transcript_mask=transcript_mask)

        return _CodonBuffer()


class IsoformConsequenceLayer:
    """Factory for differentiable stop/PTC/NMD-style consequence summaries."""

    @staticmethod
    def build(nmd_distance_codons: int = 18):
        torch = require_torch()
        nn = torch.nn
        stop_index = AA_TO_INDEX["*"]

        class _ConsequenceLayer(nn.Module):
            def forward(self, amino_acid_probs, codon_mask=None, last_junction_codon_index=None):
                if amino_acid_probs.ndim != 3 or amino_acid_probs.shape[-1] != len(AMINO_ACIDS):
                    raise ValueError("amino_acid_probs must have shape B x C x 21")
                if codon_mask is None:
                    codon_mask = torch.ones(
                        amino_acid_probs.shape[:2],
                        dtype=torch.bool,
                        device=amino_acid_probs.device,
                    )
                stop_probs = amino_acid_probs[:, :, stop_index] * codon_mask.to(amino_acid_probs.dtype)
                has_stop = 1.0 - torch.prod(1.0 - stop_probs, dim=1)
                codon_lengths = codon_mask.long().sum(dim=1).clamp(min=1)
                codon_positions = torch.arange(amino_acid_probs.shape[1], device=amino_acid_probs.device)
                terminal_mask = codon_positions.unsqueeze(0) == (codon_lengths - 1).unsqueeze(1)
                premature_mask = codon_positions.unsqueeze(0) < (codon_lengths - 1).unsqueeze(1)
                terminal_stop = (stop_probs * terminal_mask.to(stop_probs.dtype)).sum(dim=1)
                premature_stop = 1.0 - torch.prod(1.0 - stop_probs * premature_mask.to(stop_probs.dtype), dim=1)
                if last_junction_codon_index is None:
                    nmd_risk = torch.zeros_like(has_stop)
                else:
                    nmd_mask = codon_positions.unsqueeze(0) + nmd_distance_codons <= last_junction_codon_index.unsqueeze(1)
                    nmd_risk = 1.0 - torch.prod(1.0 - stop_probs * nmd_mask.to(stop_probs.dtype), dim=1)
                return {
                    "has_stop": has_stop,
                    "terminal_stop": terminal_stop,
                    "premature_stop": premature_stop,
                    "nmd_risk": nmd_risk,
                }

        return _ConsequenceLayer()


class SelectiveScanEncoder:
    """Factory for a small Mamba-inspired selective scan encoder."""

    @staticmethod
    def build(input_dim: int = 7, hidden_dim: int = 128, layers: int = 2):
        torch = require_torch()
        nn = torch.nn

        class _SelectiveScanBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.forward_candidate = nn.Linear(hidden_dim, hidden_dim)
                self.forward_gate = nn.Linear(hidden_dim, hidden_dim)
                self.backward_candidate = nn.Linear(hidden_dim, hidden_dim)
                self.backward_gate = nn.Linear(hidden_dim, hidden_dim)
                self.mix = nn.Linear(hidden_dim * 2, hidden_dim)
                self.norm = nn.LayerNorm(hidden_dim)

            def _scan(self, encoded, candidate_layer, gate_layer, reverse: bool = False):
                positions = range(encoded.shape[1] - 1, -1, -1) if reverse else range(encoded.shape[1])
                state = torch.zeros(encoded.shape[0], encoded.shape[2], dtype=encoded.dtype, device=encoded.device)
                states = []
                for position in positions:
                    position_features = encoded[:, position, :]
                    candidate_state = torch.tanh(candidate_layer(position_features))
                    carry_gate = torch.sigmoid(gate_layer(position_features))
                    state = carry_gate * state + (1.0 - carry_gate) * candidate_state
                    states.append(state)
                if reverse:
                    states.reverse()
                return torch.stack(states, dim=1)

            def forward(self, encoded):
                forward_states = self._scan(encoded, self.forward_candidate, self.forward_gate)
                backward_states = self._scan(encoded, self.backward_candidate, self.backward_gate, reverse=True)
                mixed = self.mix(torch.cat([forward_states, backward_states], dim=-1))
                return self.norm(encoded + mixed)

        class _Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_projection = nn.Linear(input_dim, hidden_dim)
                self.blocks = nn.ModuleList(_SelectiveScanBlock() for _ in range(layers))
                self.norm = nn.LayerNorm(hidden_dim)

            def forward(self, features):
                encoded = self.input_projection(features)
                for block in self.blocks:
                    encoded = block(encoded)
                return self.norm(encoded)

        return _Encoder()


class SplicePointerDecoder:
    """Factory for genomic-position pointer logits over splice-path structure."""

    @staticmethod
    def build(hidden_dim: int = 128):
        torch = require_torch()
        nn = torch.nn

        class _PointerDecoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.include_head = nn.Linear(hidden_dim, 1)
                self.acceptor_head = nn.Linear(hidden_dim, 1)
                self.donor_head = nn.Linear(hidden_dim, 1)

            def forward(self, encoded_positions):
                return {
                    "include_logits": self.include_head(encoded_positions).squeeze(-1),
                    "acceptor_logits": self.acceptor_head(encoded_positions).squeeze(-1),
                    "donor_logits": self.donor_head(encoded_positions).squeeze(-1),
                }

        return _PointerDecoder()


class TypedSpliceTranslationModel:
    """Factory for the first neural prototype.

    The exact layer turns `DNA one-hot + exon mask` into codon one-hots. The
    learned layer predicts amino-acid tokens from those typed codon states.
    """

    @staticmethod
    def build(hidden_dim: int = 64):
        torch = require_torch()
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.codon_head = CodonAminoAcidHead.build(hidden_dim=hidden_dim)

            def forward(self, dna_one_hot, exon_mask):
                codons = project_spliced_codons(dna_one_hot, exon_mask)
                return self.codon_head(codons)

        return _Model()


class TypedSpliceTransducer:
    """Factory for the proposed DNA/splice-track to isoform-product model.

    The neural branch reads DNA plus splice tracks and emits pointer logits. The
    exact branch uses transcript-order genomic pointers, a codon buffer, and the
    frozen codon table to produce amino-acid probabilities and consequences.
    """

    @staticmethod
    def build(input_dim: int = 7, hidden_dim: int = 128, layers: int = 2):
        torch = require_torch()
        nn = torch.nn

        class _Transducer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = SelectiveScanEncoder.build(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers)
                self.pointer_decoder = SplicePointerDecoder.build(hidden_dim=hidden_dim)
                self.codon_buffer = CodonBufferLayer.build()
                self.codon_table = FixedCodonTableLayer.build()
                self.consequence_layer = IsoformConsequenceLayer.build()

            def forward(self, dna_one_hot, splice_tracks=None, exon_mask=None, path_indices=None, path_mask=None):
                if splice_tracks is None:
                    splice_tracks = torch.zeros(
                        (*dna_one_hot.shape[:2], input_dim - dna_one_hot.shape[-1]),
                        dtype=dna_one_hot.dtype,
                        device=dna_one_hot.device,
                    )
                features = torch.cat([dna_one_hot, splice_tracks], dim=-1)
                encoded_positions = self.encoder(features)
                pointer_logits = self.pointer_decoder(encoded_positions)
                if path_indices is None:
                    if exon_mask is None:
                        raise ValueError("Either path_indices or exon_mask must be provided")
                    path_indices, path_mask = path_indices_from_exon_mask(exon_mask)
                elif path_mask is None:
                    path_mask = torch.ones(path_indices.shape, dtype=torch.bool, device=path_indices.device)

                transcript_bases = gather_transcript_bases(dna_one_hot, path_indices, path_mask=path_mask)
                codon_bases, codon_mask = self.codon_buffer(transcript_bases, transcript_mask=path_mask)
                amino_acid_probs = self.codon_table(codon_bases)
                consequences = self.consequence_layer(amino_acid_probs, codon_mask=codon_mask)
                return {
                    "encoded_positions": encoded_positions,
                    "pointer_logits": pointer_logits,
                    "path_indices": path_indices,
                    "path_mask": path_mask,
                    "transcript_bases": transcript_bases,
                    "codon_bases": codon_bases,
                    "codon_mask": codon_mask,
                    "amino_acid_probs": amino_acid_probs,
                    "amino_acid_log_probs": torch.log(amino_acid_probs.clamp_min(1e-8)),
                    "consequences": consequences,
                }

        return _Transducer()