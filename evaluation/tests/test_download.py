import io

import pytest

from download_corpus import check_pdf_complete


def _minimal_pdf() -> bytes:
    """A tiny valid one-page PDF built with PyPDF2 (no network, no fixtures)."""
    import PyPDF2

    writer = PyPDF2.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_accepts_complete_pdf():
    data = _minimal_pdf()
    check_pdf_complete(data, len(data))


@pytest.mark.parametrize("mutate, message", [
    (lambda d: b"<html>proxy error</html>", "not a PDF"),
    (lambda d: d[: len(d) // 2], "truncated"),              # stream cut mid-file
])
def test_rejects_bad_payloads(mutate, message):
    with pytest.raises(ValueError, match=message):
        check_pdf_complete(mutate(_minimal_pdf()))


def test_rejects_length_mismatch():
    data = _minimal_pdf()
    with pytest.raises(ValueError, match="truncated"):
        check_pdf_complete(data, len(data) + 10)
