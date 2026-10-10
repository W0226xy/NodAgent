import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(
    __file__
).resolve().parents[3]

load_dotenv(
    BASE_DIR / ".env"
)


POSTGRES_USER = os.getenv(
    "POSTGRES_USER"
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD"
)

POSTGRES_DB = os.getenv(
    "POSTGRES_DB"
)

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = os.getenv(
    "POSTGRES_PORT",
    "15432",
)


if not POSTGRES_USER:
    raise RuntimeError(
        "POSTGRES_USER is not configured"
    )

if not POSTGRES_PASSWORD:
    raise RuntimeError(
        "POSTGRES_PASSWORD is not configured"
    )

if not POSTGRES_DB:
    raise RuntimeError(
        "POSTGRES_DB is not configured"
    )


DATABASE_URL = (
    "postgresql+psycopg://"
    f"{POSTGRES_USER}:"
    f"{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:"
    f"{POSTGRES_PORT}/"
    f"{POSTGRES_DB}"
)


CHECKPOINT_DATABASE_URL = (
    "postgresql://"
    f"{POSTGRES_USER}:"
    f"{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:"
    f"{POSTGRES_PORT}/"
    f"{POSTGRES_DB}"
)


EMBEDDING_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL"
)


EMBEDDING_API_KEY = os.getenv(
    "EMBEDDING_API_KEY"
)


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL"
)


EMBEDDING_DIMENSION = int(
    os.getenv(
        "EMBEDDING_DIMENSION",
        "0",
    )
)


if not EMBEDDING_BASE_URL:
    raise RuntimeError(
        "EMBEDDING_BASE_URL is not configured"
    )


if not EMBEDDING_API_KEY:
    raise RuntimeError(
        "EMBEDDING_API_KEY is not configured"
    )


if not EMBEDDING_MODEL:
    raise RuntimeError(
        "EMBEDDING_MODEL is not configured"
    )


if EMBEDDING_DIMENSION <= 0:
    raise RuntimeError(
        "EMBEDDING_DIMENSION must be a positive integer"
    )


LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL"
)


LLM_API_KEY = os.getenv(
    "LLM_API_KEY"
)


LLM_MODEL = os.getenv(
    "LLM_MODEL"
)


LLM_MAX_TOKENS = int(
    os.getenv(
        "LLM_MAX_TOKENS",
        "4096",
    )
)


# Optional: web search returns a safe unavailable response when either value
# is absent. Do not validate them at application startup.
TAVILY_MCP_URL = os.getenv(
    "TAVILY_MCP_URL"
)


TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)


if not LLM_BASE_URL:
    raise RuntimeError(
        "LLM_BASE_URL is not configured"
    )


if not LLM_API_KEY:
    raise RuntimeError(
        "LLM_API_KEY is not configured"
    )


if not LLM_MODEL:
    raise RuntimeError(
        "LLM_MODEL is not configured"
    )


if LLM_MAX_TOKENS <= 0:
    raise RuntimeError(
        "LLM_MAX_TOKENS must be a positive integer"
    )


LANGGRAPH_STRICT_MSGPACK = os.getenv(
    "LANGGRAPH_STRICT_MSGPACK",
    "true",
)


os.environ[
    "LANGGRAPH_STRICT_MSGPACK"
] = LANGGRAPH_STRICT_MSGPACK


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:16379/0",
)


CELERY_BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    REDIS_URL,
)


CELERY_RESULT_BACKEND = os.getenv(
    "CELERY_RESULT_BACKEND",
    "redis://localhost:16379/1",
)


# GitHub MCP is optional.  Do not fail application startup when it is absent:
# the external agent can still answer weather/time and web-search questions.
GITHUB_MCP_URL = os.getenv(
    "GITHUB_MCP_URL",
    "http://127.0.0.1:8082",
).strip()

GITHUB_MCP_TIMEOUT_SECONDS = float(
    os.getenv(
        "GITHUB_MCP_TIMEOUT_SECONDS",
        "20",
    )
)

GITHUB_MCP_INIT_TIMEOUT_SECONDS = float(
    os.getenv(
        "GITHUB_MCP_INIT_TIMEOUT_SECONDS",
        "10",
    )
)


if GITHUB_MCP_TIMEOUT_SECONDS <= 0:
    raise RuntimeError(
        "GITHUB_MCP_TIMEOUT_SECONDS must be positive"
    )


if GITHUB_MCP_INIT_TIMEOUT_SECONDS <= 0:
    raise RuntimeError(
        "GITHUB_MCP_INIT_TIMEOUT_SECONDS must be positive"
    )
