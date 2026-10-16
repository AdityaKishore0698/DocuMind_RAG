# Human review — scoring guide

One row per (question, system). Read the **question**, **gold answer**, **gold evidence** and
`scoring_rule` (if any), then the **answer** and the **retrieved context** that system was given.
In blind files the `system` column is a letter (A–D) shuffled per question; do not open the `_KEY.json`
file until scoring is finished. Score from the row's own context and the gold material only.

`auto_*` columns are automated hints, **not grades**: `auto_gold_quote_in_context` = share of gold quotes
found in the context; `auto_citation_validity` = share of citation markers that point to a supplied source.
Keyword recall is never a correctness score.

| Field | Scale | Definition |
|---|---|---|
| `correctness` | 0 / 1 / 2 / NA | Are the answer's factual claims right, judged against the gold answer and evidence? **2** all material claims correct (wording may differ; extra correct detail is fine); **1** partly correct — a key fact missing or one material error; **0** wrong, contradicts the gold, or a wrong refusal on an answerable question. **NA** for negative (unanswerable) questions — score `refusal_appropriate` instead. Apply `scoring_rule` where present (e.g. `crossformer-01`). |
| `evidence_support` | 0 / 1 / 2 | Are the answer's claims supported by **the retrieved context shown in this row** (regardless of the gold)? **2** every claim is stated in the context; **1** some claims are not in the context; **0** mostly unsupported (claims from model memory or invented). A correct refusal that cites nothing scores 2. |
| `completeness` | 0 / 1 / 2 / NA | Does the answer address every part the question asks (all papers, all requested quantities)? **2** all parts; **1** some parts missing; **0** misses the point. **NA** for negative questions. |
| `citation_quality` | 0 / 1 / 2 / NA | Graph systems only (rows whose `citations` column lists sources; **NA** for one-shot rows). **2** claims are cited and each cited source actually states what it is cited for; **1** some claims uncited or some citations point to sources that do not support them; **0** citations mostly missing or misleading. |
| `refusal_appropriate` | 0 / 1 | **1** the system answered when the corpus contains the answer, or refused on a negative question; **0** it refused an answerable question or answered (guessed) a negative one. A partial "the context does not specify X, but …" with a substantive answer counts as answering. |
| `paper_attribution` | 0 / 1 / NA | Cross-paper questions only (`cross_paper = yes`, otherwise **NA**). **1** each fact is attributed to the correct paper; **0** any fact attributed to the wrong paper or papers conflated. |

**Citation validity ≠ evidence support.** A citation is *valid* if it points to a source that was supplied
(automated: `auto_citation_validity`). It is *supporting* only if that source actually states the claim —
that judgement is `citation_quality`, and only a human makes it.

Fill `reviewer` (initials) and use `notes` for anything the scales miss (e.g. "right number, wrong paper").
Leave a field blank if you have not scored it — blanks are excluded from means (not counted as 0).

## AI judging (used instead of manual scoring for the 44 priority rows)
`evaluation/llm_judge.py` prepares blinded judge inputs (rubric above copied verbatim), validates judge outputs
against these exact scales and NA rules, runs a scoped second pass, and finalises a judged CSV that the
`summarize --key` command below un-blinds. Results are labelled **AI-judged, not human-verified**.

## Workflow

```bash
PY=evaluation/.venv/bin/python
# 1. (already built) regenerate a new file if needed — never overwrites an existing one
$PY evaluation/human_review.py build --run L=20261015T161715Z_L --run BL=20261015T165903Z_BL \
    --run CL=20261015T171852Z_CL --run CLV=20261015T175700Z_CLV --blind --cross-paper-first
# 2. score evaluation/results/human_review/review_L-BL-CL-CLV_<stamp>_blind.csv in a spreadsheet (save as CSV)
# 3. summarise (validates values, un-blinds via the _KEY.json, writes *_summary_<stamp>.json/.md)
$PY evaluation/human_review.py summarize evaluation/results/human_review/review_L-BL-CL-CLV_<stamp>_blind.csv
# 4. optional: per-run human_scores.csv so `run_eval.py report <RUN_ID>` shows human correctness
$PY evaluation/human_review.py export-run-scores evaluation/results/human_review/review_L-BL-CL-CLV_<stamp>_blind.csv
```

The per-run `review.csv` files written by `run_eval.py report` are left untouched; this consolidated file
replaces them for cross-system scoring. Suggested order: the 28 cross-paper rows first (they lead the file),
then the rest. Scoring all 168 rows takes several hours; a first pass on cross-paper + negatives (44 rows)
already answers the open Phase 3 question.
