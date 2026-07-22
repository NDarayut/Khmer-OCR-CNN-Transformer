from pathlib import Path

from .tokenizer import Tokenizer
from .tokenization.dispatch import MultiScriptSegmenter


class ClusterTokenizer(Tokenizer):
    """Tokenizer using grapheme-cluster segmentation (Khmer Character Clusters
    for Khmer text, per-character for Latin/digits/other -- see
    ``netra_ocr.recognition.tokenization``).

    ``decode()`` is inherited unchanged from ``Tokenizer``: decoding is purely
    mechanical id->string regardless of whether a vocab entry is one codepoint
    or several, so no override is needed there.

    Pointed at a vocab file with no multi-codepoint entries (e.g. the old
    176-token ``char2idx_new.json``), this degrades to exactly the old
    per-codepoint behavior, so it can safely replace ``Tokenizer`` at every
    encode call site.
    """

    def __init__(self, char2idx_path: str | Path, segmenter: MultiScriptSegmenter | None = None):
        super().__init__(char2idx_path)
        self.segmenter = segmenter if segmenter is not None else MultiScriptSegmenter()

    def encode(self, text: str) -> list[int]:
        ids = [self.sos_idx]
        for piece in self.segmenter.segment(text):
            ids.extend(self._encode_piece(piece))
        ids.append(self.eos_idx)
        return ids

    def _encode_piece(self, piece: str) -> list[int]:
        if piece in self.char2idx:
            return [self.char2idx[piece]]
        # OOV cluster: decompose to codepoints, look up each individually.
        # <unk> only for codepoints absent from the base vocab entirely
        # (i.e. genuinely foreign/unregistered scripts).
        return [self.char2idx.get(ch, self.unk_idx) for ch in piece]
