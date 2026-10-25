"""LLM-based post-hoc correction of Whisper transcription output.

Connects to: pipeline.py — invoked only when `llm_correct=True`. Never a
required step: Whisper output alone is complete.

Uses the openai Python SDK against any OpenAI-compatible endpoint.
The built-in prompt fixes ASR artifacts without paraphrasing and preserves
timestamps; AIDIO_CORRECTION_PROMPT replaces it for power users, while the
web UI's correction notes are appended as extra user instructions.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

CORRECT_SYSTEM_PROMPT = """\
You are a transcription editor for song lyrics produced by automatic speech
recognition. Return ONLY the corrected lyrics text. No commentary, no
markdown, no code fences, no preamble.

Fix only:
- punctuation and capitalization
- clearly misheard words, using context
- line splits where one line clearly contains two lyric lines
- hallucinated filler lines

Preserve exactly:
- the original language
- line order and line count
- leading timestamps such as [01:23.45] or LRC timestamps - never edit,
  add, remove, or reflow them
- proper nouns and words you are unsure about (leave unchanged)

Do not paraphrase, summarize, translate, explain, or reorder lines.
If unsure, keep the original text.
"""

DEFAULT_CORRECTION_MODEL = "gemma-3-12b"


def correct_lyrics(
    lyrics: str,
    context: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Send transcribed lyrics to an LLM for punctuation/word correction.

    Args:
        lyrics: Raw Whisper transcription (one line per lyric line).
        context: Optional extra instructions from the user (artist names,
            spelling preferences, profanity policy) passed with the lyrics.
            Kept separate from CORRECT_SYSTEM_PROMPT so the structural
            safeguards cannot be overridden by accident.
        endpoint: OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL
            env var, falling back to http://localhost:1234/v1.
        model: Model name; defaults to OPENAI_MODEL or gemma-3-12b.
        api_key: API key; defaults to OPENAI_API_KEY (any value works on
            local endpoints).

    Returns:
        Corrected lyrics text. On endpoint failure the original lyrics
        are returned unchanged (correction must never be destructive).
    """
    # Web/CLI entry points may not have loaded .env yet; real env vars win.
    load_dotenv()
    base_url = endpoint or os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1")
    model_name = model or os.environ.get("OPENAI_MODEL", DEFAULT_CORRECTION_MODEL)
    key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    # Power-user escape hatch: replaces the whole system prompt (env only).
    system_prompt = os.environ.get("AIDIO_CORRECTION_PROMPT") or CORRECT_SYSTEM_PROMPT

    logger.info("Correcting lyrics via %s (model: %s)", base_url, model_name)

    user_content = lyrics
    if context:
        user_content = (
            f"Additional instructions from the user:\n{context}\n\nTranscribed lyrics:\n{lyrics}"
        )

    try:
        client = OpenAI(base_url=base_url, api_key=key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        corrected = (response.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("LLM correction failed; returning original lyrics")
        return lyrics

    return corrected if corrected else lyrics
