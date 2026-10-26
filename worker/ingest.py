"""Page- and section-aware ingestion (plan §7.1). Pure functions — no database, storage or model calls.

* `extract_pages`   PDF → one record per page (PyPDF2, the existing extractor); TXT → one record with no
                    page number. NUL characters are removed: PostgreSQL text cannot store them and PyPDF2
                    emits them for some maths glyphs (docs/failure-analysis.md F-3).
* `find_headings`   heuristic detection of common research-paper section headings on their own line
                    ("3 Method", "IV. EXPERIMENTS", "Related Work", …). PDFs vary; misses are expected and
                    simply leave the previous section in force.
* `chunk_pages`     the existing RecursiveCharacterTextSplitter (1000 / 100) applied *within* each page and
                    each section span on it, so a chunk never crosses a page or a detected heading. Each
                    chunk keeps page_number, section_heading and a document-wide chunk_index.
* `paper_metadata`  title / authors / year with the source of each value recorded, so extracted metadata
                    is distinguishable from heuristics (plan §9).
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# canonical name -> accepted spellings (matched case-insensitively as the whole heading text)
SECTION_NAMES = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction"],
    "Related Work": ["related work", "related works", "prior work", "background and related work"],
    "Background": ["background", "preliminaries", "problem formulation", "problem statement"],
    "Method": ["method", "methods", "methodology", "approach", "our approach", "proposed method", "model",
               "architecture", "system overview"],
    "Experiments": ["experiments", "experimental setup", "experiment setup", "experimental results",
                    "experiments and results", "evaluation", "experimental evaluation"],
    "Results": ["results", "main results", "results and discussion"],
    "Discussion": ["discussion", "analysis", "discussion and future work", "discussion and limitations"],
    "Limitations": ["limitations", "limitations and future work", "limitation", "limitations and conclusion",
                    "limitations and conclusions"],
    "Conclusion": ["conclusion", "conclusions", "conclusion and future work", "conclusions and future work",
                   "future work", "conclusion and limitations", "conclusions and limitations"],
    "References": ["references", "bibliography"],
    "Acknowledgements": ["acknowledgements", "acknowledgments", "acknowledgement", "acknowledgment"],
    "Appendix": ["appendix", "appendices", "supplementary material"],
}
_LOOKUP = {spelling: canon for canon, spellings in SECTION_NAMES.items() for spelling in spellings}
# small caps sometimes lose the spaces between words ("RELATEDWORK"): also match with spaces removed
_LOOKUP_NOSPACE = {spelling.replace(" ", ""): canon for spelling, canon in _LOOKUP.items()}
_LOOKUP_NOSPACE.update({"experimentalevaluation": "Experiments", "experimentalresults": "Experiments"})
# optional numbering: "3", "3.", "3.1", "III.", "A." (letters/roman numerals need punctuation)
_NUMBERING = r"(?:(?:\d{1,2}(?:\.\d{1,2})*\.?|[IVX]{1,5}[.)]|[A-H][.)])\s+)?"
_HEADING = re.compile(rf"^\s*{_NUMBERING}([A-Za-z][A-Za-z &\-]{{2,48}}?)\s*:?\s*$")
# IEEE-style small caps come out of PyPDF2 split: "I. I NTRODUCTION", "II. R ELATED WORK"
_SMALL_CAPS = re.compile(r"\b([A-Z]) ([A-Z]{2,})\b")
# run-in heading after a sentence on the same line: "…trajectories.II. R ELATED WORK"
_RUN_IN = re.compile(r"[.!?:]\s*((?:\d{1,2}|[IVX]{1,5})\.\s+[A-Z][A-Z &\-]{2,48})\s*$")


@dataclass
class Page:
    page_number: int | None   # 1-based for PDFs, None for plain text
    text: str


@dataclass
class ChunkSpec:
    content: str
    page_number: int | None
    section_heading: str | None
    chunk_index: int


@dataclass
class Extracted:
    pages: list[Page]
    pdf_title: str | None = None
    pdf_author: str | None = None
    warnings: list[str] = field(default_factory=list)


def strip_nul(text: str) -> str:
    return text.replace("\x00", "")


def extract_pages(data: bytes, filename: str) -> Extracted:
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".pdf":
        return Extracted(pages=[Page(None, strip_nul(data.decode("utf-8", errors="replace")))])
    import PyPDF2

    reader = PyPDF2.PdfReader(io.BytesIO(data))
    pages, warnings = [], []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 — one bad page must not lose the paper
            warnings.append(f"page {number}: extraction failed ({type(exc).__name__})")
            text = ""
        pages.append(Page(number, strip_nul(text)))
    meta = reader.metadata or {}
    title = (meta.get("/Title") or "").strip() if hasattr(meta, "get") else ""
    author = (meta.get("/Author") or "").strip() if hasattr(meta, "get") else ""
    return Extracted(pages=pages, pdf_title=title or None, pdf_author=author or None, warnings=warnings)


def _join_small_caps(text: str) -> str:
    return _SMALL_CAPS.sub(lambda m: m.group(1) + m.group(2), text)


_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via", "with", "vs"}
_ROMAN_CAPS = re.compile(r"^\s*([IVX]{1,5})\.\s+([A-Z][A-Z0-9 &\-:]{2,56})\s*$")
_NUMBERED_TITLE = re.compile(r"^\s*(\d{1,2})\.?\s+([A-Z][A-Za-z0-9&\-: ]{2,56})\s*$")


def custom_heading(line: str) -> str | None:
    """A numbered top-level heading with a paper-specific title ("III. THE OCTO MODEL", "4 Octo Model").
    Only high-precision forms are accepted: Roman numeral + ALL CAPS, or a number + Title Case words."""
    line = _join_small_caps(line).strip()
    if len(line) > 60:
        return None
    match = _ROMAN_CAPS.match(line)
    if match:
        words = match.group(2).split()
    else:
        match = _NUMBERED_TITLE.match(line)
        if not match or int(match.group(1)) > 15:
            return None
        words = match.group(2).split()
        if not all(w[0].isupper() or w.lower() in _SMALL_WORDS or w[0].isdigit() for w in words):
            return None
    if not 1 <= len(words) <= 8:
        return None
    words = [x.lower() if x.isupper() else x for x in words]
    titled = [w if (i and w.lower() in _SMALL_WORDS) else w[0].upper() + w[1:] for i, w in enumerate(words)]
    text = " ".join(titled).strip(" :")
    return canonical_heading(text) or text


def canonical_heading(line: str) -> str | None:
    """Canonical section name if `line` looks like a heading on its own line, else None."""
    line = _join_small_caps(line)
    if len(line) > 60:
        return None
    match = _HEADING.match(line)
    if not match or not match.group(1)[0].isupper():   # headings are capitalised; wrapped prose rarely is
        return None
    name = re.sub(r"\s+", " ", match.group(1)).strip().lower()
    return _LOOKUP.get(name) or _LOOKUP_NOSPACE.get(name.replace(" ", ""))


def find_headings(text: str) -> list[tuple[int, str]]:
    """(character offset of the heading line, canonical name), in order of appearance."""
    found, offset = [], 0
    for line in text.splitlines(keepends=True):
        name = canonical_heading(line.strip()) or custom_heading(line)
        if name:
            found.append((offset, name))
        else:
            tail = _RUN_IN.search(line.rstrip("\n"))
            if tail:
                name = canonical_heading(tail.group(1))
                if name:
                    found.append((offset + tail.start(1), name))
        offset += len(line)
    return found


def _splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def chunk_pages(pages: list[Page], splitter=None) -> list[ChunkSpec]:
    splitter = splitter or _splitter()
    chunks: list[ChunkSpec] = []
    section: str | None = None
    for page in pages:
        if not page.text.strip():
            continue
        boundaries = find_headings(page.text)
        spans = []                      # (start, end, section in force)
        start = 0
        for offset, name in boundaries:
            spans.append((start, offset, section))
            section, start = name, offset
        spans.append((start, len(page.text), section))
        for begin, end, span_section in spans:
            segment = page.text[begin:end]
            if not segment.strip():
                continue
            for piece in splitter.split_text(segment):
                if piece.strip():
                    chunks.append(ChunkSpec(piece, page.page_number, span_section, len(chunks)))
    return chunks


_ARXIV = re.compile(r"arXiv:\s?(\d{2})(\d{2})\.\d{4,5}")
_JUNK_LINE = re.compile(r"(?i)https?://|www\.|\.github\.io|^preprint$|^\d{4}-\d{1,2}-\d{1,2}$|arxiv:|^under review")
_CONNECTIVES = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "the", "to", "via", "with"}


def _continues_title(line: str, nxt: str) -> bool:
    """Is `nxt` the second line of a wrapped title (rather than an author/affiliation line)?"""
    if not nxt or len(nxt) > 90 or re.search(r"[\d∗*†@]", nxt) or _JUNK_LINE.search(nxt):
        return False
    last = line.split()[-1].lower() if line.split() else ""
    if line.endswith(":") or last in _CONNECTIVES:   # the first line is visibly unfinished
        return True
    words = [w.lower().strip(".:") for w in nxt.split()]
    # author lists have commas and no connectives; a title's second line usually has one
    return "," not in nxt and (nxt[:1].islower() or any(w in _CONNECTIVES for w in words))


def first_page_title(text: str) -> str | None:
    """Title from the first page: skip URLs/dates/'Preprint', then join a title wrapped over two lines."""
    head = text.split("Abstract")[0] if "Abstract" in text[:3000] else text[:600]
    lines = [_join_small_caps(line.strip()) for line in head.splitlines() if line.strip()]
    lines = [line for line in lines if not _JUNK_LINE.search(line)]
    for i, line in enumerate(lines):
        if 5 <= len(line) <= 200 and not re.fullmatch(r"[\d\W]+", line):
            title = line
            if i + 1 < len(lines) and _continues_title(line, lines[i + 1]):
                title = f"{line} {lines[i + 1]}"
            return re.sub(r"\s+", " ", title).strip(" .")[:300]
    return None


def paper_metadata(extracted: Extracted, chunks: list[ChunkSpec]) -> dict:
    """title / authors / year plus the source of each, and the sections that were detected."""
    meta: dict = {"sections": sorted({c.section_heading for c in chunks if c.section_heading}),
                  "warnings": extracted.warnings}
    title = extracted.pdf_title
    if title and len(title) > 5 and not re.search(r"(?i)\.(pdf|docx?|tex)$|^untitled|^microsoft", title):
        meta["title"], meta["title_source"] = title[:300], "pdf_metadata"
    else:
        first = next((p.text for p in extracted.pages if p.text.strip()), "")
        title = first_page_title(first)
        if title:
            meta["title"], meta["title_source"] = title, "first_page_heuristic"
    if extracted.pdf_author:
        meta["authors"], meta["authors_source"] = extracted.pdf_author[:500], "pdf_metadata"
    first_text = "\n".join(p.text for p in extracted.pages[:2])
    match = _ARXIV.search(first_text)
    if match:
        meta["year"], meta["year_source"] = 2000 + int(match.group(1)), "arxiv_identifier"
        meta["arxiv_id"] = match.group(0).replace("arXiv:", "").strip()
    return meta
