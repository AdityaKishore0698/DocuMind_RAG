"""Download the evaluation corpus from arXiv (PDFs are never committed).

    evaluation/.venv/bin/python evaluation/download_corpus.py            # enabled papers
    evaluation/.venv/bin/python evaluation/download_corpus.py --all      # include disabled extras
    evaluation/.venv/bin/python evaluation/download_corpus.py --verify   # hashes only, no network

* Verifies each paper's title (and that the requested version exists) through
  the arXiv API before downloading the version-pinned PDF.
* Writes evaluation/data/pdfs/<paper_id>.pdf (gitignored).
* Records sha256 / size / page count in evaluation/corpus.lock.json (committed)
  so later runs can detect a changed PDF.
* Pauses >= 3 s between arXiv requests, per arXiv's API etiquette.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from common import CORPUS_LOCK, PDF_DIR, load_corpus, setup_logging

USER_AGENT = "DocuMind-eval/0.1 (research benchmark; contact: repository owner)"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={ids}&max_results=100"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
PAUSE_S = 3.0

log = setup_logging()


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — fixed https hosts
        return response.read()


def check_pdf_complete(data: bytes, expected_length: int | None = None) -> None:
    """Raise ValueError unless ``data`` is a whole, parseable PDF.

    A proxy can close a stream early without an error, so the header alone is
    not enough: check the declared length, the trailing %%EOF and a real parse.
    """
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    if expected_length is not None and len(data) != expected_length:
        raise ValueError(f"truncated: got {len(data)} of {expected_length} bytes")
    if b"%%EOF" not in data[-2048:]:
        raise ValueError("truncated: no %%EOF marker at the end")
    try:
        page_count(data)
    except Exception as exc:  # noqa: BLE001 — any parser failure means an unusable file
        raise ValueError(f"unparseable PDF ({type(exc).__name__})") from exc


def _download(url: str, dest, attempts: int = 4, timeout: int = 90) -> bytes:
    """Stream to a temp file and rename atomically; retry stalled transfers."""
    tmp = dest.with_suffix(".part")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as fh:  # noqa: S310
                expected = response.headers.get("Content-Length")
                while chunk := response.read(1 << 16):
                    fh.write(chunk)
            data = tmp.read_bytes()
            check_pdf_complete(data, int(expected) if expected and expected.isdigit() else None)
            tmp.replace(dest)
            return data
        except (OSError, ValueError) as exc:  # timeouts, resets, truncated or bad payloads
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            wait = PAUSE_S * 2 ** attempt
            log.warning("attempt %d/%d failed (%s) — retrying in %.0fs", attempt, attempts, type(exc).__name__, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _norm_title(title: str) -> str:
    return " ".join(title.split()).casefold()


def fetch_titles(versioned_ids: list[str]) -> dict[str, str]:
    root = ET.fromstring(_get(ARXIV_API.format(ids=",".join(versioned_ids))))
    titles = {}
    for entry in root.findall("a:entry", ATOM):
        entry_id = entry.find("a:id", ATOM).text.rsplit("/", 1)[-1]
        titles[entry_id] = " ".join(entry.find("a:title", ATOM).text.split())
    return titles


def page_count(data: bytes) -> int:
    import PyPDF2

    return len(PyPDF2.PdfReader(io.BytesIO(data)).pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="include papers with enabled: false")
    parser.add_argument("--verify", action="store_true", help="only check local PDFs against corpus.lock.json")
    parser.add_argument("--force", action="store_true", help="re-download even if the PDF exists")
    args = parser.parse_args()

    papers = load_corpus(enabled_only=not args.all)
    lock = json.loads(CORPUS_LOCK.read_text()) if CORPUS_LOCK.exists() else {"papers": {}}
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        bad = 0
        for paper in papers:
            entry = lock["papers"].get(paper.id)
            if not paper.pdf_path.exists() or not entry:
                log.error("%s: missing PDF or lock entry", paper.id)
                bad += 1
                continue
            digest = hashlib.sha256(paper.pdf_path.read_bytes()).hexdigest()
            status = "ok" if digest == entry["sha256"] else "HASH MISMATCH"
            bad += status != "ok"
            log.info("%-18s %s", paper.id, status)
        return 1 if bad else 0

    titles = fetch_titles([p.versioned_id for p in papers])
    problems = []
    for paper in papers:
        api_title = titles.get(paper.versioned_id)
        if api_title is None:
            problems.append(f"{paper.id}: {paper.versioned_id} not found in arXiv API response")
        elif _norm_title(api_title) != _norm_title(paper.title):
            problems.append(f"{paper.id}: title mismatch — arXiv says {api_title!r}")
    if problems:
        for p in problems:
            log.error(p)
        return 1

    failed = []
    for paper in papers:
        data = paper.pdf_path.read_bytes() if paper.pdf_path.exists() and not args.force else None
        if data is not None:
            try:
                check_pdf_complete(data)
                log.info("%-18s already present (%d bytes)", paper.id, len(data))
            except ValueError as exc:
                log.warning("%-18s existing file is incomplete (%s) — re-downloading", paper.id, exc)
                paper.pdf_path.unlink()
                data = None
        if data is None:
            time.sleep(PAUSE_S)
            url = f"https://arxiv.org/pdf/{paper.versioned_id}"
            log.info("%-18s downloading %s", paper.id, url)
            try:
                data = _download(url, paper.pdf_path)
            except (OSError, ValueError) as exc:
                log.error("%s: download failed after retries (%s)", paper.id, type(exc).__name__)
                failed.append(paper.id)
                continue
        digest = hashlib.sha256(data).hexdigest()
        previous = lock["papers"].get(paper.id)
        if previous and previous["sha256"] == digest:
            continue  # unchanged — keep the existing lock entry (no churn)
        if previous:
            log.warning("%s: sha256 changed since the lock was written (was %s…)", paper.id, previous["sha256"][:12])
        lock["papers"][paper.id] = {
            "arxiv": paper.versioned_id,
            "title": paper.title,
            "license": paper.license,
            "sha256": digest,
            "bytes": len(data),
            "pages": page_count(data),
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _write_lock(lock)  # after every paper, so partial progress is never lost
    _write_lock(lock)
    log.info("wrote %s (%d papers)", CORPUS_LOCK.name, len(lock["papers"]))
    if failed:
        log.error("failed: %s — re-run to retry (already-downloaded papers are skipped)", ", ".join(failed))
        return 1
    return 0


def _write_lock(lock: dict) -> None:
    lock["papers"] = dict(sorted(lock["papers"].items()))
    CORPUS_LOCK.write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
