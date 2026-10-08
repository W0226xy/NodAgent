import unittest

from backend.app.services.citation_service import (
    extract_citation_numbers,
    filter_cited_sources,
)


def source(
    source_number: int,
    document_id: int | None = None,
    chunk_id: int | None = None,
):
    return {
        "source_number": source_number,
        "citation": f"[{source_number}]",
        "document_id": document_id or source_number,
        "chunk_id": chunk_id or source_number,
        "filename": f"document-{source_number}.txt",
    }


class CitationServiceTests(unittest.TestCase):
    def setUp(self):
        self.retrieved_sources = [source(number) for number in range(1, 6)]

    def test_only_explicit_citation_is_returned(self):
        sources = filter_cited_sources(
            "答案依据 [1]。",
            self.retrieved_sources,
        )
        self.assertEqual([item["source_number"] for item in sources], [1])

    def test_multiple_citations_keep_answer_order(self):
        sources = filter_cited_sources(
            "答案依据 [1] 和 [3]。",
            self.retrieved_sources,
        )
        self.assertEqual([item["source_number"] for item in sources], [1, 3])

    def test_missing_citation_returns_no_sources(self):
        self.assertEqual(
            filter_cited_sources("没有引用编号。", self.retrieved_sources),
            [],
        )

    def test_duplicate_citation_is_returned_once(self):
        sources = filter_cited_sources(
            "同一来源 [1][1]。",
            self.retrieved_sources,
        )
        self.assertEqual([item["source_number"] for item in sources], [1])

    def test_duplicate_document_chunk_is_returned_once(self):
        duplicated = [
            source(1, document_id=7, chunk_id=9),
            source(3, document_id=7, chunk_id=9),
        ]
        sources = filter_cited_sources("[1][3]", duplicated)
        self.assertEqual([item["source_number"] for item in sources], [1])

    def test_extract_citation_numbers(self):
        self.assertEqual(extract_citation_numbers("[1][3][1]"), [1, 3])


if __name__ == "__main__":
    unittest.main()
