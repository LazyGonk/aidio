"""Segment-to-lyrics post-processing.

Connects to: asr.py (consumes Segment lists) and output.py (produces the
final plain-text lyric block that gets written/embedded).

Raw Whisper segments are often mid-line fragments. Songs have regular
phrasing, so we merge segments into lines using pause boundaries and
collapse repeated lines (choruses) that Whisper transcribes with slight
variations.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from itertools import pairwise

from aidio.lyrics.asr import Segment

logger = logging.getLogger(__name__)

# Pause (seconds) between segments above which we consider a line break.
LINE_BREAK_PAUSE = 0.35
# Similarity threshold at which two consecutive lines are considered the
# same chorus line transcribed slightly differently.
DUP_SIMILARITY = 0.85
# Max words per merged line — sung phrases rarely exceed this; longer runs
# indicate Whisper merged multiple lyric lines into one segment.
MAX_LINE_WORDS = 9
# Whisper hallucination phrases to drop (common on instrumental sections).
HALLUCINATION_PATTERNS = (
    re.compile(r"^\s*(all right|okay|thank you|thanks for watching)\b", re.IGNORECASE),
    re.compile(r"\b(subscribe|amara\.org)\b", re.IGNORECASE),
)
# Trailing hallucination fragments to strip from line ends (e.g. a lone
# "You" Whisper appends when the music fades).
TRAILING_FRAGMENTS = re.compile(r"\s+\b(you|yeah|oh|hey|mm+|hmm+)\b[\s.]*$", re.IGNORECASE)
# Interjected backing-vocal shouts that Whisper inserts mid-line
# (e.g. "Call the hell in!" echoes between lead lines).
INTERJECTION_PATTERNS = (re.compile(r"\.\s+(?:Call|Come|Sing|Shout)\b[^.]*!\s*", re.IGNORECASE),)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/parens — used only for similarity checks."""
    text = re.sub(r"[({\[][^)}\]]*[)}\]]", " ", text)  # drop (backing vocals)
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def segments_to_lines(segments: list[Segment]) -> list[str]:
    """Merge segments into lyric lines.

    Strategy: consecutive segments are joined into one line when the gap
    between them is small AND the merged line stays under MAX_LINE_WORDS
    (Whisper often emits one segment containing several lyric lines; the
    word cap splits those back apart). A pause longer than LINE_BREAK_PAUSE
    starts a new line. This mirrors how lyrics are conventionally
    line-broken and matches the reference file's structure.
    """
    if not segments:
        return []
    lines: list[str] = []
    current = segments[0].text
    for prev, seg in pairwise(segments):
        gap = seg.start - prev.end
        merged_words = len(current.split()) + len(seg.text.split())
        if gap >= LINE_BREAK_PAUSE or merged_words > MAX_LINE_WORDS:
            lines.append(current.strip())
            current = seg.text
        else:
            current = f"{current} {seg.text}"
    lines.append(current.strip())
    # Drop hallucinated non-lyric lines and strip trailing fragments.
    cleaned = []
    for line in lines:
        if not line or any(p.search(line) for p in HALLUCINATION_PATTERNS):
            continue
        line = TRAILING_FRAGMENTS.sub("", line).strip()
        for pattern in INTERJECTION_PATTERNS:
            line = pattern.sub(" ", line).strip()
        if line:
            cleaned.append(line)
    return cleaned


def dedupe_lines(lines: list[str]) -> list[str]:
    """Collapse near-duplicate consecutive lines (chorus repetition artifacts).

    Whisper often renders the same sung line twice with minor word errors;
    the reference lyrics keep one line per repetition, so we keep the first
    occurrence of near-identical consecutive lines.
    """
    result: list[str] = []
    for line in lines:
        if result:
            norm_a, norm_b = _normalize(result[-1]), _normalize(line)
            if (
                norm_a
                and norm_b
                and SequenceMatcher(None, norm_a, norm_b).ratio() >= DUP_SIMILARITY
            ):
                logger.debug("Dropping near-duplicate line: %r", line)
                continue
        result.append(line)
    return result


def build_lyrics_text(segments: list[Segment], timestamps: bool = False) -> str:
    """Full pipeline: segments -> lines -> dedupe -> text block.

    With timestamps=True, each line is prefixed with [MM:SS.xx] in LRC style.
    Connects to: output.py — this is the text that gets saved/embedded.
    """
    lines = dedupe_lines(segments_to_lines(segments))
    if not timestamps:
        return "\n".join(lines) + "\n"
    # Attach the start time of the first segment that formed each line.
    stamped: list[str] = []
    seg_idx = 0
    for line in lines:
        word_count = len(line.split())
        start = segments[min(seg_idx, len(segments) - 1)].start
        minutes, seconds = divmod(start, 60)
        stamped.append(f"[{int(minutes):02d}:{seconds:05.2f}] {line}")
        # Advance segment index past the segments consumed by this line.
        consumed = 0
        while seg_idx < len(segments) and consumed < max(word_count, 1):
            consumed += len(segments[seg_idx].text.split())
            seg_idx += 1
    return "\n".join(stamped) + "\n"
