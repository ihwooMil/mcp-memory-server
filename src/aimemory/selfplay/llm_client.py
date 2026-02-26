"""Ollama LLM client wrapper with retry, timeout, and Korean enforcement."""

from __future__ import annotations

import logging
import re
import time
from typing import Iterator

import ollama

from aimemory.config import OllamaConfig

logger = logging.getLogger(__name__)

# Korean character ranges
_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]")

# Minimum ratio of Korean characters among all non-ASCII, non-whitespace, non-punct chars
_MIN_KOREAN_RATIO = 0.4
# Minimum number of Korean characters in a valid response
_MIN_KOREAN_CHARS = 5


def is_korean_text(text: str) -> bool:
    """Check if text is primarily Korean (not Chinese/English-only)."""
    if not text or not text.strip():
        return False
    korean_count = len(_HANGUL_RE.findall(text))
    chinese_count = len(_CJK_RE.findall(text))

    # Must have enough Korean characters
    if korean_count < _MIN_KOREAN_CHARS:
        return False

    # If Chinese chars are present, Korean must dominate
    if chinese_count > 0 and korean_count < chinese_count:
        return False

    # Check ratio: Korean chars / total non-space non-ascii
    non_space = re.sub(r"[\s\x00-\x7F]", "", text)
    if len(non_space) == 0:
        return False

    korean_ratio = korean_count / len(non_space)
    return korean_ratio >= _MIN_KOREAN_RATIO


_KOREAN_REMINDER = (
    "\n\n[중요: 반드시 한국어로만 답변하세요. "
    "영어나 중국어를 사용하지 마세요. 코드 예시도 최소화하세요.]"
)


class LLMClient:
    """Synchronous Ollama chat wrapper with retry, timeout, and Korean enforcement."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self._client = ollama.Client(host=self.config.base_url)

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        enforce_korean: bool = True,
    ) -> str:
        """Send chat messages and return the assistant response content.

        If enforce_korean is True (default), retries with a Korean reminder
        when the response is not primarily Korean. After max retries, returns
        the best Korean response found or the last response.
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.chat(
                    model=self.config.model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "top_p": self.config.top_p,
                        "num_predict": max_tokens,
                    },
                )
                content = response.message.content.strip()

                if not enforce_korean or is_korean_text(content):
                    return content

                # Non-Korean response detected: retry with reminder
                logger.warning(
                    "Non-Korean response detected (attempt %d/%d), retrying with reminder",
                    attempt + 1,
                    self.config.max_retries,
                )
                # Add Korean enforcement reminder to system message
                messages = _inject_korean_reminder(messages)
                continue

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)

        if last_error:
            raise RuntimeError(
                f"LLM call failed after {self.config.max_retries} attempts: {last_error}"
            )
        # All retries exhausted due to non-Korean: return last content anyway
        logger.warning("Could not get Korean response after retries, returning last response")
        response = self._client.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": min(temperature + 0.2, 1.0),
                "top_p": self.config.top_p,
                "num_predict": max_tokens,
            },
        )
        return response.message.content.strip()

    def stream(
        self,
        messages: list[dict],
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Stream chat response tokens."""
        temperature = temperature if temperature is not None else self.config.temperature
        response = self._client.chat(
            model=self.config.model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            },
        )
        for chunk in response:
            if chunk.message and chunk.message.content:
                yield chunk.message.content

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            self._client.list()
            return True
        except Exception:
            return False


def _inject_korean_reminder(messages: list[dict]) -> list[dict]:
    """Add Korean language reminder to the system message."""
    messages = [m.copy() for m in messages]
    for m in messages:
        if m["role"] == "system" and _KOREAN_REMINDER not in m["content"]:
            m["content"] += _KOREAN_REMINDER
            break
    return messages
