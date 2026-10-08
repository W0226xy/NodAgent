from typing import (
    Any,
    Dict,
    List,
    TypedDict,
)

from backend.app.services.document_parser import (
    parse_and_split_document,
)
from backend.app.services.embedding_service import (
    embedding_model,
)
from backend.app.core.config import (
    EMBEDDING_DIMENSION,
)


class DocumentProcessingError(RuntimeError):
    """A safe, user-facing failure raised while preparing document chunks."""


class PreparedChunk(TypedDict):
    chunk_index: int

    content: str

    metadata_json: Dict[
        str,
        Any,
    ]

    embedding: List[float]


def prepare_document_chunks(
    file_path: str,
) -> List[PreparedChunk]:
    chunks = (
        parse_and_split_document(
            file_path=file_path
        )
    )

    chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk.page_content, str)
        and chunk.page_content.strip()
    ]

    if not chunks:
        raise DocumentProcessingError(
            "文档未解析出可用文本，无法建立知识库索引。"
        )

    texts = [
        chunk.page_content.strip()
        for chunk in chunks
    ]

    try:
        embeddings = (
            embedding_model.embed_documents(
                texts
            )
        )
    except Exception as exc:
        raise DocumentProcessingError(
            "调用嵌入服务失败，文档尚未写入知识库。"
        ) from exc

    if len(embeddings) != len(chunks):
        raise DocumentProcessingError(
            "嵌入返回数量与文本分块数量不一致，"
            "文档尚未写入知识库。"
        )

    prepared_chunks: List[
        PreparedChunk
    ] = []

    for index, (
        chunk,
        embedding,
    ) in enumerate(
        zip(
            chunks,
            embeddings,
        )
    ):
        if not isinstance(embedding, list):
            raise DocumentProcessingError(
                "第 "
                f"{index} "
                "个分块的嵌入数据格式无效。"
            )

        if len(embedding) != EMBEDDING_DIMENSION:
            raise DocumentProcessingError(
                "第 "
                f"{index} "
                "个分块的嵌入向量维度无效："
                f"期望 {EMBEDDING_DIMENSION} 维，"
                f"实际 {len(embedding)} 维。"
            )

        prepared_chunks.append(
            PreparedChunk(
                chunk_index=index,
                content=(
                    texts[index]
                ),
                metadata_json=dict(
                    chunk.metadata
                ),
                embedding=embedding,
            )
        )

    return prepared_chunks
