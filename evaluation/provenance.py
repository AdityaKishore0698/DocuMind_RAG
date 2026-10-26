"""Harness-side provenance: which PDF page(s) each System A chunk came from.

System A concatenates page texts and chunks the result without keeping page
numbers. To score page-level retrieval and evidence recall we rebuild the page
boundaries exactly as ``baseline_a.extract_text`` does (empty pages are skipped,
every non-empty page is followed by "\\n") and locate every chunk inside the
concatenated text. This is used for *evaluation only*.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import PyPDF2


@dataclass(frozen=True)
class PageSpan:
    page_number: int  # 1-based physical page index in the PDF
    start: int        # inclusive offset into the concatenated text
    end: int          # exclusive offset (excludes the trailing "\n")


@dataclass
class ChunkProvenance:
    ordinal: int
    text: str
    char_start: int | None
    char_end: int | None
    pages: list[int] = field(default_factory=list)

    @property
    def mapped(self) -> bool:
        return self.char_start is not None


def page_texts_from_pdf(data: bytes) -> list[str]:
    """Raw per-page text from PyPDF2, in page order (empty strings kept)."""
    reader = PyPDF2.PdfReader(io.BytesIO(data))
    return [page.extract_text() or "" for page in reader.pages]


def concat_with_spans(page_texts: list[str], preprocess=None) -> tuple[str, list[PageSpan]]:
    """Replicate extract_text's concatenation and remember each page's span.

    ``preprocess`` (e.g. preprocess.strip_nul) is applied to each included page;
    the inclusion decision still uses the raw text, exactly like production, so
    the result equals ``preprocess(extract_text(...))`` for any character-wise
    preprocessing."""
    text = ""
    spans: list[PageSpan] = []
    for index, extracted in enumerate(page_texts, start=1):
        if extracted:
            page = preprocess(extracted) if preprocess else extracted
            start = len(text)
            text += page + "\n"
            spans.append(PageSpan(page_number=index, start=start, end=start + len(page)))
    return text, spans


def pages_for_range(spans: list[PageSpan], start: int, end: int) -> list[int]:
    """Pages whose text overlaps the half-open range [start, end)."""
    return [s.page_number for s in spans if s.start < end and start < s.end]


def map_chunks(text: str, spans: list[PageSpan], chunks: list[str]) -> list[ChunkProvenance]:
    """Locate each chunk in ``text`` (chunks are stripped substrings, in order).

    Overlapping chunks start after their predecessor, so we search forward
    from the previous match first and fall back to a full search.
    """
    result: list[ChunkProvenance] = []
    cursor = 0
    for ordinal, chunk in enumerate(chunks):
        pos = text.find(chunk, cursor)
        if pos < 0:
            pos = text.find(chunk)
        if pos < 0:
            result.append(ChunkProvenance(ordinal, chunk, None, None, []))
            continue
        end = pos + len(chunk)
        result.append(ChunkProvenance(ordinal, chunk, pos, end, pages_for_range(spans, pos, end)))
        cursor = pos + 1
    return result
