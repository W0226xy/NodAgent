import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
)
from urllib.parse import (
    urlparse,
)


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def get_safe_web_url(
    value: Any,
) -> Optional[str]:
    """Only expose absolute public HTTP(S) URLs in answers or the UI."""

    if not isinstance(value, str):
        return None

    url = value.strip()

    if not url or any(
        character.isspace()
        for character in url
    ):
        return None

    parsed = urlparse(url)

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None

    return url


def get_web_source_url(
    source: Dict[str, Any],
) -> Optional[str]:
    if source.get("source_type") != "web":
        return None

    return get_safe_web_url(
        source.get("url")
    )


def markdown_link_text(
    value: str,
) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("\n", " ")
        .strip()
    )


def render_web_citations(
    answer: str,
    retrieved_sources: List[Dict[str, Any]],
) -> str:
    """Replace web-only [n] markers with one original Markdown URL per page.

    Document citations intentionally remain numeric because they point at local
    chunks rather than publicly addressable pages.
    """

    sources_by_number: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for source in retrieved_sources:
        try:
            source_number = int(
                source["source_number"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if get_web_source_url(source):
            sources_by_number.setdefault(
                source_number,
                source,
            )

    seen_urls = {
        url
        for source in sources_by_number.values()
        if (url := get_web_source_url(source))
        and url in answer
    }

    def replace(
        match: re.Match[str],
    ) -> str:
        source = sources_by_number.get(
            int(match.group(1))
        )

        if source is None:
            return match.group(0)

        url = get_web_source_url(source)

        if url is None:
            return ""

        if url in seen_urls:
            return ""

        seen_urls.add(url)
        title = source.get("title") or source.get("filename")

        if isinstance(title, str) and title.strip():
            return f"[{markdown_link_text(title)}]({url})"

        return f"<{url}>"

    return CITATION_PATTERN.sub(
        replace,
        answer,
    )


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

        if (
            source.get("source_type") == "web"
            and not get_web_source_url(source)
        ):
            continue

        chunk_key = (
            source.get("document_id"),
            source.get("chunk_id"),
        )

        if chunk_key in seen_chunks:
            continue

        seen_chunks.add(chunk_key)
        cited_sources.append(source)

    # A web-aware model may already use Markdown links instead of [n]. Keep
    # their structured source cards when the original, validated URL appears.
    for source in retrieved_sources:
        url = get_web_source_url(source)

        if not url or url not in answer:
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
                "source_type": "document",
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
