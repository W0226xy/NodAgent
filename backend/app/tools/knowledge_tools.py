import asyncio
import logging
from typing import (
    Any,
    Dict,
    Tuple,
)

from langchain.tools import (
    ToolRuntime,
)
from langchain_core.tools import (
    tool,
)

from backend.app.agents.context import (
    AgentContext,
)
from backend.app.services import (
    citation_service,
    retrieval_service,
)


logger = logging.getLogger(__name__)


@tool(
    "search_knowledge_base",
    response_format=(
        "content_and_artifact"
    ),
)
async def search_knowledge_base(
    query: str,
    runtime: ToolRuntime[
        AgentContext
    ],
) -> Tuple[
    str,
    Dict[str, Any],
]:
    """
    Search the current workspace knowledge base.

    Use this tool when the user's question
    depends on uploaded documents or knowledge
    stored in the current workspace.

    The query should contain enough semantic
    context to retrieve the relevant document
    chunks.
    """

    workspace_id = (
        runtime.context[
            "workspace_id"
        ]
    )

    top_k = (
        runtime.context[
            "top_k"
        ]
    )

    try:
        results = (
            await retrieval_service
            .search_workspace_async(
                workspace_id=(
                    workspace_id
                ),
                query=query,
                top_k=top_k,
            )
        )
    except Exception:
        logger.exception(
            "Knowledge retrieval failed: workspace_id=%s",
            workspace_id,
        )
        return (
            "当前知识库检索暂时不可用，请稍后重试。",
            {"sources": []},
        )

    logger.info(
        "Knowledge search completed: workspace_id=%s result_count=%s",
        workspace_id,
        len(results),
    )

    if not results:
        reason = retrieval_service.get_empty_search_reason(
            workspace_id=workspace_id,
        )
        return (
            "当前知识库没有找到可用信息（"
            f"{reason}）。",
            {
                "sources": [],
            },
        )

    sources = await asyncio.to_thread(
        citation_service
        .build_citation_sources,
        results,
    )

    tool_content = (
        citation_service
        .build_citation_context(
            sources
        )
    )

    artifact = {
        "sources": sources,
    }

    return (
        tool_content,
        artifact,
    )
