import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
)


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citation_numbers(
    answer: str,
) -> List[int]:
    """Return unique citation numbers in their first appearance order."""

    citation_numbers: List[int] = []
    seen_numbers = set()

    for match in CITATION_PATTERN.finditer(answer):
        source_number = int(match.group(1))

        if source_number in seen_numbers:
            continue

        seen_numbers.add(source_number)
        citation_numbers.append(source_number)

    return citation_numbers


def filter_cited_sources(
    answer: str,
    retrieved_sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep only retrieved sources explicitly cited by the final answer.

    Source numbering is assigned before the agent sees the retrieval context,
    so the original source_number/citation values must never be renumbered.
    """

    sources_by_number: Dict[int, Dict[str, Any]] = {}

    for source in retrieved_sources:
        try:
            source_number = int(source["source_number"])
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        sources_by_number.setdefault(source_number, source)

    cited_sources: List[Dict[str, Any]] = []
    seen_chunks = set()

    for source_number in extract_citation_numbers(answer):
        source = sources_by_number.get(source_number)

        if source is None:
            continue

        chunk_key = (
            source.get("document_id"),
            source.get("chunk_id"),
        )

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)
        cited_sources.append(source)

    return cited_sources

from backend.app.db.database import (
    SessionLocal,
)
from backend.app.models.document import (
    Document,
)
from backend.app.schemas.search import (
    SearchResult,
)


def get_page_number(
    metadata: Dict[str, Any],
) -> Optional[int]:
    page_label = metadata.get(
        "page_label"
    )

    if page_label is not None:
        try:
            return int(page_label)
        except (
            TypeError,
            ValueError,
        ):
            pass

    page = metadata.get(
        "page"
    )

    if isinstance(page, int):
        return page + 1

    return None


def get_document_filenames(
    document_ids: List[int],
) -> Dict[int, str]:
    if not document_ids:
        return {}

    unique_ids = list(
        dict.fromkeys(
            document_ids
        )
    )

    with SessionLocal() as db:
        rows = (
            db.query(
                Document.id,
                Document.filename,
            )
            .filter(
                Document.id.in_(
                    unique_ids
                )
            )
            .all()
        )

    return {
        int(document_id): filename
        for document_id, filename in rows
    }


def build_citation_sources(
    results: List[SearchResult],
) -> List[Dict[str, Any]]:
    filenames = (
        get_document_filenames(
            [
                result.document_id
                for result in results
            ]
        )
    )

    sources: List[
        Dict[str, Any]
    ] = []

    for source_number, result in enumerate(
        results,
        start=1,
    ):
        metadata = dict(
            result.metadata_json
            or {}
        )

        filename = filenames.get(
            result.document_id,
            f"document_{result.document_id}",
        )

        page_number = get_page_number(
            metadata
        )

        sources.append(
            {
                "source_number": (
                    source_number
                ),
                "citation": (
                    f"[{source_number}]"
                ),
                "chunk_id": (
                    result.chunk_id
                ),
                "document_id": (
                    result.document_id
                ),
                "filename": filename,
                "chunk_index": (
                    result.chunk_index
                ),
                "page_number": (
                    page_number
                ),
                "content": (
                    result.content
                ),
                "metadata_json": (
                    metadata
                ),
                "distance": (
                    result.distance
                ),
                "similarity": (
                    result.similarity
                ),
                "keyword_score": (
                    result.keyword_score
                ),
                "vector_rank": (
                    result.vector_rank
                ),
                "keyword_rank": (
                    result.keyword_rank
                ),
                "rrf_score": (
                    result.rrf_score
                ),
            }
        )

    return sources


def build_citation_context(
    sources: List[Dict[str, Any]],
) -> str:
    context_parts: List[str] = []

    for source in sources:
        citation = source[
            "citation"
        ]

        filename = source[
            "filename"
        ]

        page_number = source.get(
            "page_number"
        )

        if page_number is None:
            source_title = (
                f"{citation} "
                f"{filename}"
            )
        else:
            source_title = (
                f"{citation} "
                f"{filename} "
                f"(page {page_number})"
            )

        context_parts.append(
            (
                f"{source_title}\n"
                f"{source['content']}"
            )
        )

    return "\n\n".join(
        context_parts
    )
