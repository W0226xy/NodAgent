from functools import lru_cache

from langchain_openai import ChatOpenAI

from backend.app.core.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=60,
        max_retries=2,
    )
