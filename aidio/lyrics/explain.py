"""LLM-based lyrics explanation via OpenAI-compatible endpoint.

Connects to: cli.py — invoked when --explain is given. Consumes the final
lyrics text and returns a structured analysis (themes, motifs, characters,
narrative arc) plus a quality flag for suspicious lines.

Uses the openai Python SDK so it works with any OpenAI-compatible server
(LM Studio, Ollama, vLLM, etc.).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)

EXPLAIN_SYSTEM_PROMPT = """\
You are a literary analyst specializing in fantasy poetry and song lyrics.
You will receive transcribed song lyrics that may contain minor errors from
automatic speech recognition.

Analyze the lyrics and return a JSON object with these fields:
- "title": inferred title or best short label
- "themes": list of 2-5 major themes (e.g. "oath-bound loyalty", "war memorial")
- "motifs": list of recurring symbols/images (e.g. "blade", "bone", "fire")
- "characters": list of named or referenced characters/entities
- "narrative_arc": brief summary of the story in 2-3 sentences
- "worldbuilding_notes": observations about the fictional world, lore, or
  setting that can be extracted from the lyrics
- "suspicious_lines": list of objects with {"line": str, "reason": str} for
  lines that seem like ASR errors (nonsense words, missing words, etc.)
- "tone": one word describing the overall mood (e.g. "grim", "triumphant",
  "elegiac", "defiant")

If the lyrics are too short or unclear to analyze, set all fields to empty
and add a note in "narrative_arc".

Return ONLY valid JSON — no markdown, no explanation text outside the JSON.
"""


@dataclass
class ExplainResult:
    """Structured analysis of transcribed lyrics."""

    title: str = ""
    themes: list[str] = field(default_factory=list)
    motifs: list[str] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)
    narrative_arc: str = ""
    worldbuilding_notes: list[str] = field(default_factory=list)
    suspicious_lines: list[dict] = field(default_factory=list)
    tone: str = ""

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        lines: list[str] = []
        lines.append(f"\n{'=' * 60}")
        lines.append(f"Analysis: {self.title or 'Untitled'}")
        lines.append(f"{'=' * 60}")
        if self.tone:
            lines.append(f"Tone: {self.tone}")
        if self.themes:
            lines.append(f"Themes: {', '.join(self.themes)}")
        if self.motifs:
            lines.append(f"Motifs: {', '.join(self.motifs)}")
        if self.characters:
            lines.append(f"Characters: {', '.join(self.characters)}")
        if self.narrative_arc:
            lines.append(f"\nNarrative: {self.narrative_arc}")
        if self.worldbuilding_notes:
            lines.append("\nWorldbuilding:")
            for note in self.worldbuilding_notes:
                lines.append(f"  - {note}")
        if self.suspicious_lines:
            lines.append("\nSuspicious lines (possible ASR errors):")
            for sl in self.suspicious_lines:
                lines.append(f"  '{sl['line']}' — {sl['reason']}")
        lines.append("=" * 60)
        return "\n".join(lines)


def explain_lyrics(
    lyrics: str,
    endpoint: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> ExplainResult:
    """Send lyrics to an OpenAI-compatible LLM for analysis.

    Args:
        lyrics: The transcribed lyrics text.
        endpoint: OpenAI-compatible API base URL.
            Defaults to OPENAI_BASE_URL env var, falls back to
            http://localhost:1234/v1 (LM Studio default).
        model: Model name. Defaults to OPENAI_MODEL env var, falls back to
            a small capable model like "gemma-3-12b" or "llama3.1-8b".
        api_key: API key. Defaults to OPENAI_API_KEY env var.

    Returns:
        ExplainResult with structured analysis.
    """
    base_url = endpoint or os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
    model_name = model or os.environ.get("OPENAI_MODEL", "gemma-3-12b")
    api_key_val = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")

    client = OpenAI(base_url=base_url, api_key=api_key_val)

    logger.info("Explaining lyrics via %s (model: %s)", base_url, model_name)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": lyrics},
            ],
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
    except Exception:
        logger.exception("LLM explain failed")
        return ExplainResult(
            title="Analysis unavailable",
            narrative_arc="LLM endpoint returned an error.",
        )

    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON, parsing best-effort")
        data = _parse_fallback(content)

    return _build_result(data)


def _parse_fallback(raw: str) -> dict:
    """Best-effort JSON extraction from LLM output that wrapped JSON in
    markdown code blocks or added trailing text."""
    import json

    # Strip markdown code fences if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    # Try to find the first { and last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _build_result(data: dict) -> ExplainResult:
    """Convert raw LLM JSON into an ExplainResult."""
    return ExplainResult(
        title=data.get("title", ""),
        themes=data.get("themes", []),
        motifs=data.get("motifs", []),
        characters=data.get("characters", []),
        narrative_arc=data.get("narrative_arc", ""),
        worldbuilding_notes=data.get("worldbuilding_notes", []),
        suspicious_lines=data.get("suspicious_lines", []),
        tone=data.get("tone", ""),
    )
