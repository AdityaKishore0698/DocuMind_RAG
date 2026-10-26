import baseline_a
from provenance import concat_with_spans, map_chunks, pages_for_range


def test_concat_matches_production_concatenation_rules():
    pages = ["Page one text.", "", "Page three text."]  # empty page skipped, like extract_text
    text, spans = concat_with_spans(pages)
    assert text == "Page one text.\nPage three text.\n"
    assert [s.page_number for s in spans] == [1, 3]
    assert text[spans[1].start:spans[1].end] == "Page three text."


def test_pages_for_range_overlaps():
    _, spans = concat_with_spans(["aaaa", "bbbb"])  # "aaaa\nbbbb\n"
    assert pages_for_range(spans, 0, 4) == [1]
    assert pages_for_range(spans, 3, 7) == [1, 2]   # straddles the page break
    assert pages_for_range(spans, 4, 5) == []       # only the separator newline


def test_real_production_splitter_chunks_map_back_to_pages():
    pages = [
        ("Methods. " + "We train a transformer policy on demonstrations. " * 30).strip(),
        ("Results. " + "The policy succeeds in most evaluation trials. " * 30).strip(),
        ("Limitations. " + "Performance drops on unseen embodiments. " * 30).strip(),
    ]
    text, spans = concat_with_spans(pages)
    chunks = baseline_a.split_text(text)
    assert len(chunks) > 3
    prov = map_chunks(text, spans, chunks)
    assert all(p.mapped for p in prov), "every production chunk must be locatable in the text"
    assert all(text[p.char_start:p.char_end] == p.text for p in prov)
    assert all(p.pages for p in prov)
    assert any("Limitations." in p.text and 3 in p.pages for p in prov)
    # ordinals preserve chunk order and offsets never go backwards
    assert [p.ordinal for p in prov] == list(range(len(chunks)))
    assert all(a.char_start <= b.char_start for a, b in zip(prov, prov[1:]))


def test_unmappable_chunk_is_reported_not_guessed():
    text, spans = concat_with_spans(["hello world"])
    prov = map_chunks(text, spans, ["not present"])
    assert not prov[0].mapped and prov[0].pages == []


def test_txt_branch_of_extract_text():
    assert baseline_a.extract_text("héllo".encode("utf-8"), "notes.txt") == "héllo"
    assert baseline_a.extract_text(b"\xff\xfe", "x.txt") == "��"
