from langchain_openai import OpenAIEmbeddings

from backend.app.core.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)


embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL,
    dimensions=EMBEDDING_DIMENSION,
    # Some OpenAI-compatible embedding APIs accept
    # only an array of strings, not token ID arrays.
    check_embedding_ctx_length=False,
)
