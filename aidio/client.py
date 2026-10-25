"""OpenAI-compatible API client.

Wraps ``httpx`` to talk to any OpenAI-compatible endpoint (LM Studio,
llama.cpp, LiteLLM gateway, …).  Sends audio as base64 WAV in the
``audio_url`` field of a multimodal message.
"""

from __future__ import annotations

import httpx

from .config import get_settings


class AnalysisClient:
    """Thin wrapper around the OpenAI chat-completions API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        cfg = get_settings()
        self.base_url = base_url or cfg.base_url
        self.api_key = api_key or cfg.api_key
        self.model = model or cfg.model
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def analyze(self, prompt: str, audio_b64_wav: str) -> str:
        """Send one audio chunk + a text prompt to the model.

        Parameters
        ----------
        prompt : str
            Text instructions for what to analyze.
        audio_b64_wav : str
            Base64-encoded WAV data.

        Returns
        -------
        str
            The model's text response.
        """
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64_wav,
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 32768,
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        # Check if response was truncated
        finish_reason = data["choices"][0].get("finish_reason", "")
        if finish_reason == "length":
            content += "\n\n[Note: Response was truncated due to token limit]"
        return content
