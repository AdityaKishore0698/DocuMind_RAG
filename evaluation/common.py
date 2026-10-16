"""Shared paths, corpus/question loading, schema validation and safe logging."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
DATA_DIR = EVAL_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
PAGES_DIR = DATA_DIR / "pages"
RESULTS_DIR = EVAL_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"            # gitignored: raw traces with paper excerpts
SUMMARIES_DIR = RESULTS_DIR / "summaries"  # committable: metrics only, no paper text
CORPUS_FILE = EVAL_DIR / "corpus.yaml"
CORPUS_LOCK = EVAL_DIR / "corpus.lock.json"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"

QUESTION_TYPES = {"lookup", "method", "dataset", "result", "limitation", "cross_paper", "negative"}
QUESTION_STATUSES = {"draft", "approved", "rejected"}
FACETS = {"method", "architecture", "dataset", "result", "metric", "limitation", "setup", "claim"}


# --- logging -----------------------------------------------------------------
class RedactSecrets(logging.Filter):
    """Replace any known secret value that leaks into a log record."""

    def __init__(self, names: tuple[str, ...] = ("GEMINI_API_KEY",)):
        super().__init__()
        self._secrets = [v for v in (os.environ.get(n) for n in names) if v and len(v) >= 8]

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            message = record.getMessage()
            for secret in self._secrets:
                message = message.replace(secret, "[REDACTED]")
            record.msg, record.args = message, ()
        return True


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactSecrets())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # The HTTP client logs full request URLs at INFO; keep it quiet.
    for noisy in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("evaluation")


# --- corpus ------------------------------------------------------------------
@dataclass(frozen=True)
class Paper:
    id: str
    arxiv_id: str
    version: str
    title: str
    year: int
    license: str
    enabled: bool

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}{self.version}"

    @property
    def pdf_path(self) -> Path:
        return PDF_DIR / f"{self.id}.pdf"

    @property
    def pages_path(self) -> Path:
        return PAGES_DIR / f"{self.id}.json"

    @property
    def filename(self) -> str:
        """Name stored in documents.filename (production keeps the upload name)."""
        return f"{self.id}.pdf"


def load_corpus(path: Path = CORPUS_FILE, enabled_only: bool = True) -> list[Paper]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    papers = [
        Paper(
            id=p["id"],
            arxiv_id=str(p["arxiv_id"]),
            version=str(p["version"]),
            title=p["title"],
            year=int(p["year"]),
            license=p["license"],
            enabled=bool(p.get("enabled", True)),
        )
        for p in raw["papers"]
    ]
    ids = [p.id for p in papers]
    if len(ids) != len(set(ids)):
        raise ValueError("corpus.yaml: duplicate paper ids")
    return [p for p in papers if p.enabled] if enabled_only else papers


def load_pages(paper: Paper) -> list[str]:
    """Per-page texts written by `run_eval.py extract` (index 0 = page 1)."""
    return json.loads(paper.pages_path.read_text(encoding="utf-8"))["pages"]


# --- questions -------------------------------------------------------------------
def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    questions = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"questions.jsonl line {line_no}: {exc}") from exc
    return questions


def validate_questions(questions: list[dict], corpus_ids: set[str]) -> list[str]:
    """Structural checks (no PDFs needed). Returns a list of error strings."""
    errors: list[str] = []
    seen: set[str] = set()
    for q in questions:
        qid = q.get("id", "<missing id>")
        if qid in seen:
            errors.append(f"{qid}: duplicate id")
        seen.add(qid)
        missing = [k for k in ("id", "status", "type", "papers", "question", "gold_answer", "evidence") if k not in q]
        if missing:
            errors.extend(f"{qid}: missing field '{k}'" for k in missing)
            continue
        if q["status"] not in QUESTION_STATUSES:
            errors.append(f"{qid}: unknown status {q['status']!r}")
        if q["type"] not in QUESTION_TYPES:
            errors.append(f"{qid}: unknown type {q['type']!r}")
        if not str(q["question"]).strip():
            errors.append(f"{qid}: empty question")
        if "scoring_rule" in q and not (isinstance(q["scoring_rule"], str) and q["scoring_rule"].strip()):
            errors.append(f"{qid}: scoring_rule must be a non-empty string when present")
        for pid in q["papers"]:
            if pid not in corpus_ids:
                errors.append(f"{qid}: paper {pid!r} not in corpus")
        if q["type"] == "negative":
            if q["evidence"] or q["papers"]:
                errors.append(f"{qid}: negative questions must have no papers/evidence")
            continue
        if not q["evidence"]:
            errors.append(f"{qid}: non-negative question without evidence")
        if q["type"] == "cross_paper" and len(set(q["papers"])) < 2:
            errors.append(f"{qid}: cross_paper question needs >= 2 papers")
        for i, ev in enumerate(q["evidence"]):
            for key in ("paper", "page", "quote", "facet"):
                if key not in ev:
                    errors.append(f"{qid}: evidence[{i}] missing '{key}'")
            if ev.get("paper") not in q["papers"]:
                errors.append(f"{qid}: evidence[{i}] paper {ev.get('paper')!r} not listed in papers")
            if not isinstance(ev.get("page"), int) or ev.get("page", 0) < 1:
                errors.append(f"{qid}: evidence[{i}] page must be a positive int")
            if ev.get("facet") not in FACETS:
                errors.append(f"{qid}: evidence[{i}] unknown facet {ev.get('facet')!r}")
            quotes = [str(ev.get("quote", ""))]
            for j, alt in enumerate(ev.get("alternatives", [])):
                if set(alt) != {"page", "quote"}:
                    errors.append(f"{qid}: evidence[{i}].alternatives[{j}] must have exactly 'page' and 'quote'")
                    continue
                if not isinstance(alt["page"], int) or alt["page"] < 1:
                    errors.append(f"{qid}: evidence[{i}].alternatives[{j}] page must be a positive int")
                quotes.append(str(alt["quote"]))
            for quote in quotes:
                if len(quote.split()) < 4:
                    errors.append(f"{qid}: evidence[{i}] quote too short to be unambiguous")
                if len(quote.split()) > 60:
                    errors.append(f"{qid}: evidence[{i}] quote too long (keep excerpts short)")
        covered = {ev.get("paper") for ev in q["evidence"]}
        for pid in q["papers"]:
            if pid not in covered:
                errors.append(f"{qid}: paper {pid!r} has no evidence item")
    return errors
