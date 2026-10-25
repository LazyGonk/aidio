"""Analyzer orchestration.

Coordinates preprocessing and API calls: loads an audio file, splits
it into chunks, sends each chunk to the LLM, and aggregates the
results into a single analysis report.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audio import preprocess_audio
from .client import AnalysisClient
from .config import get_settings

DEFAULT_PROMPT = """Analyze this audio recording in detail. Describe:
- Instruments and sounds you hear
- Genre and style
- Vocal characteristics (if any)
- Tempo, mood, and energy
- Any notable production qualities
- Overall impression

Be specific and thorough."""


@dataclass
class AnalysisResult:
    """Aggregated result from analyzing one or more audio chunks."""

    full_text: str
    chunk_count: int
    total_duration_seconds: float


def analyze_audio(
    path: str,
    *,
    prompt: str = DEFAULT_PROMPT,
    client: AnalysisClient | None = None,
    model: str | None = None,
    target_sr: int | None = None,
    max_chunk_seconds: float | None = None,
) -> AnalysisResult:
    """Preprocess an audio file and send it to the LLM for analysis.

    Parameters
    ----------
    path : str
        Path to the audio file.
    prompt : str
        Text prompt guiding the analysis.
    client : AnalysisClient, optional
        Pre-built client.  If ``None``, a new one is created from env.
    model : str, optional
        Analysis model override for the client created here. Ignored when
        ``client`` is supplied.
    target_sr : int, optional
        Target sample rate for preprocessing; defaults to ``MODEL_SAMPLE_RATE``.
    max_chunk_seconds : float, optional
        Maximum chunk duration; defaults to ``MAX_CHUNK_SECONDS``.

    Returns
    -------
    AnalysisResult
        Aggregated analysis text plus metadata.
    """
    if target_sr is None or max_chunk_seconds is None:
        settings = get_settings()
        if target_sr is None:
            target_sr = settings.sample_rate
        if max_chunk_seconds is None:
            max_chunk_seconds = settings.max_chunk_seconds

    chunks = preprocess_audio(path, target_sr=target_sr, max_chunk_seconds=max_chunk_seconds)

    if client is None:
        client = AnalysisClient(model=model)

    parts: list[str] = []
    for i, chunk in enumerate(chunks):
        parts.append(f"--- Chunk {i + 1}/{len(chunks)} ({chunk.duration_seconds:.1f}s) ---\n")
        parts.append(client.analyze(prompt, chunk.base64_wav))

    return AnalysisResult(
        full_text="\n\n".join(parts),
        chunk_count=len(chunks),
        total_duration_seconds=sum(c.duration_seconds for c in chunks),
    )
