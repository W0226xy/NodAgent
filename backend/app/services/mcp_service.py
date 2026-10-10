import asyncio
import logging
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import (
    StreamableHttpTransport,
)

from backend.app.core.config import (
    GITHUB_MCP_INIT_TIMEOUT_SECONDS,
    GITHUB_MCP_TIMEOUT_SECONDS,
    GITHUB_MCP_URL,
    TAVILY_API_KEY,
    TAVILY_MCP_URL,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


load_dotenv(
    PROJECT_ROOT / ".env"
)


EXTERNAL_INFO_SERVER_PATH = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "mcp_servers"
    / "external_info_server.py"
)


logger = logging.getLogger(
    __name__
)


PLACEHOLDER_GITHUB_TOKENS = {
    "your_github_personal_access_token",
    "your_token_here",
    "changeme",
    "change_me",
    "replace_me",
}


class GitHubMCPConfigurationError(
    RuntimeError
):
    """A local configuration error which should not fail a chat request."""


def is_usable_github_token(
    token: str | None,
) -> bool:
    if not token:
        return False

    normalized = token.strip()

    if not normalized:
        return False

    return (
        normalized.lower()
        not in PLACEHOLDER_GITHUB_TOKENS
    )


def get_github_mcp_diagnostic() -> str | None:
    """Return a safe configuration problem, without ever returning a token."""

    if not GITHUB_MCP_URL:
        return "GITHUB_MCP_URL is not configured"

    parsed = urlparse(GITHUB_MCP_URL)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "GITHUB_MCP_URL must be an absolute http(s) URL"

    try:
        parsed.port
    except ValueError:
        return "GITHUB_MCP_URL contains an invalid port"

    if not is_usable_github_token(
        os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    ):
        return (
            "GITHUB_PERSONAL_ACCESS_TOKEN is missing or still uses "
            "an example placeholder"
        )

    return None


async def log_github_mcp_startup_diagnostics(
) -> None:
    """Log a bounded, token-safe connectivity diagnostic during API startup."""

    problem = get_github_mcp_diagnostic()

    if problem:
        logger.warning(
            "GitHub MCP unavailable at startup: %s. GitHub requests "
            "will return a readable fallback; other external tools remain available.",
            problem,
        )
        return

    parsed = urlparse(GITHUB_MCP_URL)
    port = parsed.port or (
        443 if parsed.scheme == "https" else 80
    )

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                parsed.hostname,
                port,
            ),
            timeout=GITHUB_MCP_INIT_TIMEOUT_SECONDS,
        )
        del reader
        writer.close()
        await writer.wait_closed()
    except Exception as exc:
        logger.warning(
            "GitHub MCP TCP probe failed at startup url=%s error_type=%s. "
            "GitHub requests will degrade gracefully.",
            GITHUB_MCP_URL,
            type(exc).__name__,
        )
        return

    logger.info(
        "GitHub MCP startup probe succeeded url=%s transport=streamable_http "
        "timeout_seconds=%s",
        GITHUB_MCP_URL,
        GITHUB_MCP_TIMEOUT_SECONDS,
    )


def create_external_info_client() -> Client:
    """
    NodAgent local MCP server.

    Development transport:
    stdio.
    """

    return Client(
        EXTERNAL_INFO_SERVER_PATH
    )


def create_web_search_client() -> Client:
    """Connect to Tavily's remote Streamable HTTP MCP server."""

    if not TAVILY_MCP_URL or not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_MCP_URL and TAVILY_API_KEY must be configured."
        )

    separator = "&" if "?" in TAVILY_MCP_URL else "?"
    connection_url = (
        f"{TAVILY_MCP_URL}{separator}"
        f"{urlencode({'tavilyApiKey': TAVILY_API_KEY})}"
    )

    transport = StreamableHttpTransport(
        url=connection_url
    )

    return Client(
        transport=transport
    )


def create_github_mcp_client() -> Client:
    """
    GitHub official MCP server.

    Transport:
    Streamable HTTP.

    GitHub HTTP MCP requires the client
    to send its GitHub access token in
    the Authorization header.
    """

    problem = get_github_mcp_diagnostic()

    if problem:
        raise GitHubMCPConfigurationError(
            problem
        )

    github_token = os.environ[
        "GITHUB_PERSONAL_ACCESS_TOKEN"
    ].strip()

    logger.info(
        "GitHub MCP connection configured url=%s transport=streamable_http",
        GITHUB_MCP_URL,
    )

    transport = (
        StreamableHttpTransport(
            url=GITHUB_MCP_URL,
            headers={
                "Authorization": (
                    f"Bearer "
                    f"{github_token}"
                ),

                "X-MCP-Readonly": (
                    "true"
                ),

                "X-MCP-Toolsets": (
                    "repos,"
                    "issues,"
                    "pull_requests"
                ),
            },
        )
    )

    return Client(
        transport=transport,
        timeout=GITHUB_MCP_TIMEOUT_SECONDS,
        init_timeout=GITHUB_MCP_INIT_TIMEOUT_SECONDS,
    )
