from functools import lru_cache
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from backend.app.core.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
)


LENGTH_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_completion_tokens",
}


def get_finish_reason(message: Any) -> Optional[str]:
    """Return a provider completion reason without logging response content."""

    for metadata_name in ("response_metadata", "additional_kwargs"):
        metadata = getattr(message, metadata_name, None)
        if not isinstance(metadata, dict):
            continue
        for key in ("finish_reason", "stop_reason"):
            reason = metadata.get(key)
            if isinstance(reason, str) and reason.strip():
                return reason.strip().lower()
    return None


def is_length_limited(message: Any) -> bool:
    return get_finish_reason(message) in LENGTH_FINISH_REASONS


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=60,
        max_retries=2,
        max_completion_tokens=LLM_MAX_TOKENS,
    )
