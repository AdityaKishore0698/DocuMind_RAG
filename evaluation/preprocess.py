"""Evaluation-only text preprocessing. NOT part of production and NOT part of
the frozen System A copy in baseline_a.py.

Deviation: strip NUL (0x00) characters.
----------------------------------------
PyPDF2 extracts some mathematical glyphs as NUL bytes (e.g. the bracket in
"N(0, I)" or the minus in "exp(-m i)"). PostgreSQL text values cannot contain
NUL, so psycopg2 rejects the INSERT with
``ValueError: A string literal cannot contain NUL (0x00) characters``.

In the corpus this affects Octo (1 of 95 chunks), Diffusion Policy (2 of 114)
and ACT/ALOHA (3 of 92). Production stores chunks through the same driver, so
it would most likely fail to ingest those papers entirely; the harness instead
removes the NUL characters (nothing else) so the benchmark measures retrieval
and answer quality on the whole corpus. Every run records this deviation in its
metadata; the report states that production would not have ingested these papers.

The stripping is applied per page *after* deciding which pages to include,
using production's rule (a page is included iff PyPDF2 returned non-empty text),
so chunk text, stored text and page provenance stay consistent.
"""

from __future__ import annotations

NUL = "\x00"
DEVIATION = "strip_nul: evaluation-only removal of NUL (0x00) characters before chunking and storage"


def strip_nul(text: str) -> str:
    return text.replace(NUL, "")


def nul_count(text: str) -> int:
    return text.count(NUL)
