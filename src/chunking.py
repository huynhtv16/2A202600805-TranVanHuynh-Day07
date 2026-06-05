from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split on sentence boundaries and preserve punctuation.
        sentences = re.split(r'(?<=[.!?])(?:\s+|\n+)', text.strip())
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class ParagraphChunker:
    """
    Split text by paragraph boundaries.

    This preserves semantic units when each paragraph already contains
    a coherent idea. Paragraph chunking is useful for retrieval when the
    source document is already formatted in meaningful paragraphs.
    """

    def __init__(self) -> None:
        pass

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        paragraphs = re.split(r"\n\s*\n+", text.strip())
        return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


class SlidingWindowChunker:
    """
    Split text into overlapping sentence chunks.

    This helps preserve context across sentence boundaries. Each chunk is a
    sliding window of consecutive sentences, so relevant information that spans
    multiple sentences is less likely to be split apart.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        sentences = re.split(r'(?<=[.!?])(?:\s+|\n+)', text.strip())
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        chunks: list[str] = []
        stride = max(1, self.max_sentences_per_chunk - 1)
        if len(sentences) <= self.max_sentences_per_chunk:
            return [" ".join(sentences)]

        for i in range(0, len(sentences) - self.max_sentences_per_chunk + 1, stride):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, list(self.separators))

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []

        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        parts = current_text.split(separator)
        if len(parts) == 1:
            return self._split(current_text, next_separators)

        chunks: list[str] = []
        current_part = ""

        for part in parts:
            candidate = separator.join([current_part, part]) if current_part else part
            if len(candidate) <= self.chunk_size:
                current_part = candidate
                continue

            if current_part:
                chunks.extend(self._split(current_part, next_separators))
            if len(part) <= self.chunk_size:
                current_part = part
            else:
                chunks.extend(self._split(part, next_separators))
                current_part = ""

        if current_part:
            chunks.extend(self._split(current_part, next_separators))

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(value * value for value in vec_a))
    norm_b = math.sqrt(sum(value * value for value in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text)
        sentence_chunks = SentenceChunker(max_sentences_per_chunk=max(1, chunk_size // 100)).chunk(text)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)
        paragraph_chunks = ParagraphChunker().chunk(text)
        sliding_chunks = SlidingWindowChunker(max_sentences_per_chunk=max(1, chunk_size // 100)).chunk(text)

        def summarize(chunks: list[str]) -> dict:
            avg_length = sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0.0
            return {
                "count": len(chunks),
                "avg_length": avg_length,
                "chunks": chunks,
            }

        return {
            "fixed_size": summarize(fixed_chunks),
            "by_sentences": summarize(sentence_chunks),
            "recursive": summarize(recursive_chunks),
            "by_paragraph": summarize(paragraph_chunks),
            "sliding_window": summarize(sliding_chunks),
        }
