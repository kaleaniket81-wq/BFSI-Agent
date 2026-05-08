"""tests/unit/test_chunker.py — Unit tests for TextChunker."""
import pytest
from ingestion.src.chunker.text_chunker import TextChunker


class TestTextChunker:

    def setup_method(self):
        self.chunker = TextChunker(chunk_size=10, overlap=2)

    def test_basic_chunking(self):
        text   = " ".join([f"word{i}" for i in range(25)])
        chunks = self.chunker.chunk_page(text, page_number=1)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        text   = " ".join([f"w{i}" for i in range(50)])
        chunks = self.chunker.chunk_page(text, page_number=1)
        for c in chunks:
            assert c["token_count"] <= 10

    def test_chunk_metadata(self):
        chunks = self.chunker.chunk_page("hello world test document", page_number=3)
        assert chunks[0]["page_number"]  == 3
        assert chunks[0]["chunk_index"]  == 0
        assert "token_count" in chunks[0]
        assert "text"        in chunks[0]

    def test_empty_text_returns_empty(self):
        assert self.chunker.chunk_page("", page_number=1) == []

    def test_short_text_single_chunk(self):
        chunks = self.chunker.chunk_page("short text", page_number=1)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "short text"

    def test_overlap_creates_shared_words(self):
        # With overlap=2, adjacent chunks share last 2 words of prior chunk
        text   = " ".join([f"w{i}" for i in range(30)])
        chunks = self.chunker.chunk_page(text, page_number=1)
        if len(chunks) > 1:
            last_words_of_first = chunks[0]["text"].split()[-2:]
            first_words_of_second = chunks[1]["text"].split()[:2]
            assert last_words_of_first == first_words_of_second

    def test_chunk_document_multiple_pages(self):
        parsed = {
            "doc_type": "loan_agreement",
            "pages": [
                {"page_number": 1, "text": " ".join([f"w{i}" for i in range(20)])},
                {"page_number": 2, "text": " ".join([f"x{i}" for i in range(20)])},
            ]
        }
        chunker = TextChunker(chunk_size=8, overlap=1)
        chunks  = chunker.chunk_document(parsed)
        assert len(chunks) > 2
        # Each chunk should have doc_type
        assert all(c["doc_type"] == "loan_agreement" for c in chunks)
