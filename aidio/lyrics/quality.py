"""Quality comparison against reference lyrics.

Connects to: cli.py — used when --reference is given to score output
against ground truth. The handoff doc's baseline was ~38-46% word overlap;
our target is >= 90%.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class QualityReport:
    """Metrics from comparing transcribed lyrics against a reference."""

    line_recall: float  # fraction of reference lines found in output
    word_overlap: float  # fraction of reference words found in output
    char_similarity: float  # overall sequence similarity of normalized text

    def summary(self) -> str:
        """Human-readable one-line summary for CLI output."""
        return (
            f"line recall: {self.line_recall:.1%} | "
            f"word overlap: {self.word_overlap:.1%} | "
            f"char similarity: {self.char_similarity:.1%}"
        )


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation/parens for fair comparison."""
    text = re.sub(r"[({\[][^)}\]]*[)}\]]", " ", text)
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def compare_with_reference(transcript: str, reference: str) -> QualityReport:
    """Compare a transcript against reference lyrics.

    Line recall uses fuzzy matching (SequenceMatcher >= 0.75) instead of
    the handoff script's substring matching, so paraphrased-but-correct
    lines still count — a fairer quality measure.
    """
    ref_lines = [line.strip() for line in reference.splitlines() if line.strip()]
    out_norm = _normalize(transcript)

    matched = 0
    for ref_line in ref_lines:
        norm = _normalize(ref_line)
        if not norm:
            continue
        # Fuzzy containment: is this reference line (mostly) present?
        if norm in out_norm or SequenceMatcher(None, norm, out_norm).ratio() >= 0.75:
            matched += 1
            continue
        # Sliding window over output lines for line-level fuzzy match.
        out_lines = [_normalize(line) for line in transcript.splitlines() if line.strip()]
        if any(SequenceMatcher(None, norm, ol).ratio() >= 0.75 for ol in out_lines):
            matched += 1

    ref_words = set(_normalize(reference).split())
    out_words = set(_normalize(transcript).split())
    word_overlap = len(ref_words & out_words) / len(ref_words) if ref_words else 0.0
    char_sim = SequenceMatcher(None, _normalize(reference), _normalize(transcript)).ratio()

    return QualityReport(
        line_recall=matched / len(ref_lines) if ref_lines else 0.0,
        word_overlap=word_overlap,
        char_similarity=char_sim,
    )
