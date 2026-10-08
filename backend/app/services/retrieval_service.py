import logging
from dataclasses import dataclass
from typing import (
    List,
    Sequence,
    Tuple,
)

from langchain_core.documents import (
    Document as LangChainDocument,
)
from sqlalchemy.orm import Session

from backend.app.core.document_status import (
    DocumentProcessingStatus,
)
from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.document import (
    Document,
)
from backend.app.models.document_chunk import (
    DocumentChunk,
)
from backend.app.models.workspace import (
    Workspace,
)
from backend.app.schemas.search import (
    SearchResult,
)
from backend.app.services.vector_store_service import (
    vector_store,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalReadiness:
    document_ids: List[int]
    empty_reason: str | None = None


def get_retrieval_readiness(
    db: Session,
    workspace_id: int,
) -> RetrievalReadiness:
    """Check the database invariants required before vector retrieval."""

    workspace = (
        db.query(Workspace)
        .filter(Workspace.id == workspace_id)
        .first()
    )

    if workspace is None:
        return RetrievalReadiness(
            document_ids=[],
            empty_reason="workspace 不匹配",
        )

    workspace_document_count = (
        db.query(Document)
        .filter(Document.workspace_id == workspace_id)
        .count()
    )

    if workspace_document_count == 0:
        return RetrievalReadiness(
            document_ids=[],
            empty_reason="无文档",
        )

    document_ids = get_completed_document_ids(
        db=db,
        workspace_id=workspace_id,
    )

    if not document_ids:
        return RetrievalReadiness(
            document_ids=[],
            empty_reason="无已完成文档",
        )

    all_chunk_count = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id.in_(document_ids))
        .count()
    )

    workspace_chunk_count = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id.in_(document_ids),
            DocumentChunk.workspace_id == workspace_id,
        )
        .count()
    )

    if all_chunk_count and not workspace_chunk_count:
        return RetrievalReadiness(
            document_ids=[],
            empty_reason="workspace 不匹配",
        )

    if workspace_chunk_count == 0:
        return RetrievalReadiness(
            document_ids=[],
            empty_reason="无 Chunk",
        )

    embedded_chunk_count = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.document_id.in_(document_ids),
            DocumentChunk.workspace_id == workspace_id,
            DocumentChunk.embedding.is_not(None),
        )
        .count()
    )

    if embedded_chunk_count != workspace_chunk_count:
        return RetrievalReadiness(
            document_ids=[],
            empty_reason="无 embedding",
        )

    return RetrievalReadiness(document_ids=document_ids)


def get_empty_search_reason(workspace_id: int) -> str:
    """Return a safe diagnostic for a user-visible empty retrieval result."""

    with SessionLocal() as db:
        readiness = get_retrieval_readiness(
            db=db,
            workspace_id=workspace_id,
        )

    return readiness.empty_reason or "向量未命中"


def log_empty_retrieval(
    workspace_id: int,
    reason: str,
) -> None:
    logger.info(
        "Knowledge retrieval empty: workspace_id=%s reason=%s",
        workspace_id,
        reason,
    )


def get_completed_document_ids(
    db: Session,
    workspace_id: int,
) -> List[int]:
    """
    获取当前 workspace 中已经完成解析和入库的文档 ID。
    """

    rows = (
        db.query(Document.id)
        .filter(
            Document.workspace_id
            == workspace_id,
            Document.processing_status
            == (
                DocumentProcessingStatus
                .COMPLETED
                .value
            ),
        )
        .all()
    )

    return [
        row[0]
        for row in rows
    ]


def build_vector_filter(
    workspace_id: int,
    document_ids: List[int],
):
    """
    限制向量检索范围：

    1. 当前 workspace
    2. 已完成处理的 documents
    """

    return {
        "$and": [
            {
                "workspace_id":
                    workspace_id
            },
            {
                "document_id": {
                    "$in":
                        document_ids
                }
            },
        ]
    }


def convert_vector_results(
    documents_with_scores: Sequence[
        Tuple[
            LangChainDocument,
            float,
        ]
    ],
) -> List[SearchResult]:
    """
    将 PGVector 返回结果转换成 NodAgent SearchResult。
    """

    results: List[
        SearchResult
    ] = []

    for rank, (
        document,
        distance,
    ) in enumerate(
        documents_with_scores,
        start=1,
    ):
        metadata = dict(
            document.metadata
        )

        document_id = metadata.pop(
            "document_id"
        )

        chunk_index = metadata.pop(
            "chunk_index"
        )

        metadata.pop(
            "workspace_id",
            None,
        )

        # cosine distance:
        # similarity = 1 - distance
        similarity = (
            1.0
            - float(distance)
        )

        result = SearchResult(
            chunk_id=int(
                document.id
            ),
            document_id=int(
                document_id
            ),
            chunk_index=int(
                chunk_index
            ),
            content=(
                document.page_content
            ),
            metadata_json=metadata,

            distance=float(
                distance
            ),

            similarity=similarity,

            # 已取消 Keyword Retrieval
            keyword_score=None,

            # Vector 排名
            vector_rank=rank,

            keyword_rank=None,

            # 已取消 RRF
            rrf_score=0.0,
        )

        results.append(
            result
        )

    return results


def search_workspace(
    db: Session,
    workspace_id: int,
    query: str,
    top_k: int = 5,
) -> List[SearchResult]:
    """
    同步 Vector-only RAG Retrieval。

    qwen3-embedding
        ->
    pgvector
        ->
    HNSW
        ->
    cosine similarity
        ->
    Top-K
    """

    readiness = get_retrieval_readiness(
        db=db,
        workspace_id=workspace_id,
    )

    if readiness.empty_reason:
        log_empty_retrieval(
            workspace_id,
            readiness.empty_reason,
        )
        return []

    vector_filter = (
        build_vector_filter(
            workspace_id=workspace_id,
            document_ids=(
                readiness.document_ids
            ),
        )
    )

    documents_with_scores = (
        vector_store
        .similarity_search_with_score(
            query=query,

            # Vector-only 直接取最终 Top-K
            k=top_k,

            filter=vector_filter,
        )
    )

    results = convert_vector_results(
        documents_with_scores
    )

    if not results:
        log_empty_retrieval(workspace_id, "向量未命中")

    return results


async def search_workspace_async(
    workspace_id: int,
    query: str,
    top_k: int = 5,
) -> List[SearchResult]:
    """
    异步 Vector-only RAG Retrieval。
    """

    with SessionLocal() as db:

        readiness = get_retrieval_readiness(
            db=db,
            workspace_id=workspace_id,
        )

        if readiness.empty_reason:
            log_empty_retrieval(
                workspace_id,
                readiness.empty_reason,
            )
            return []

    vector_filter = (
        build_vector_filter(
            workspace_id=workspace_id,
            document_ids=(
                readiness.document_ids
            ),
        )
    )

    documents_with_scores = (
        await vector_store
        .asimilarity_search_with_score(
            query=query,

            # Vector-only 直接取最终 Top-K
            k=top_k,

            filter=vector_filter,
        )
    )

    results = convert_vector_results(
        documents_with_scores
    )

    if not results:
        log_empty_retrieval(workspace_id, "向量未命中")

    return results
