---
name: markitdown-pdf-validator
description: Validates a Markdown (.md) file that was generated from a PDF by Microsoft's MarkItDown tool, by comparing it against the source PDF and appending back any content MarkItDown dropped, mangled, or skipped (table rows, sidebar/callout text, multi-column bleed-through, FAQ/accordion lists, nested bullet lists inside table cells, decorative/stylised headings). Use this skill any time the user provides both a PDF and its MarkItDown-generated .md counterpart and asks to check, validate, verify, diff, compare, audit, or fix the conversion — or asks whether anything is "missing" from the .md. Also use it if the user asks to convert a PDF with MarkItDown and then verify the result. Trigger even if they only say "check this md against the pdf" without naming MarkItDown explicitly.
---

# MarkItDown PDF ↔ MD Validator

MarkItDown (and PDF-to-Markdown converters generally) can silently drop or
scramble content: table cells, nested lists inside cells, multi-column
layouts, sidebar/callout boxes, FAQ accordions, and stylised or
letter-spaced headings are the most common failure points. This skill
finds anything from the source PDF that isn't reasonably represented in
the .md file, and appends it back so nothing gets lost.

## Inputs required

You need exactly two files from the user:
1. The **source PDF**.
2. The **MarkItDown-generated .md file** for that same PDF.

If the user has only given you a PDF (no .md yet), first convert it with
MarkItDown yourself (see "If no .md exists yet" below), then run the
validation.

If the user has given you two PDFs, or two .md files, or files that
clearly don't correspond to each other (very different page counts,
unrelated titles), stop and ask them to confirm which file is the PDF
source and which is the MarkItDown output — don't guess.

## Workflow

1. **Locate the files.** Confirm the PDF path and the .md path (both should
   already be readable from wherever the user uploaded them, e.g. under
   `/mnt/user-data/uploads/`).

2. **Run the comparison + repair script** — this does the whole job in one
   step (extraction, comparison, and appending):

   ```bash
   python3 scripts/compare_and_append.py <input.pdf> <input.md> <output.md>
   ```

   - `<output.md>` can be the same path as `<input.md>` if the user wants
     to update it in place, or a new path (e.g. `<name>_validated.md`) if
     they want to keep the original untouched. Default to writing a new
     `*_validated.md` file unless the user says to edit in place.
   - The script prints a one-line summary: how many chunks were checked
     and how many were found missing.
   - If dependencies aren't installed yet, install them first:
     `pip install pdfplumber --break-system-packages -q`

3. **Report the result to the user in plain terms:**
   - If nothing was missing: say so plainly — the MD conversion is a
     faithful match, no changes made.
   - If content was missing: state how many items were recovered and from
     which pages, and mention that they've been appended under a
     `## ⚠️ Content recovered from source PDF` heading at the end of the
     file, grouped by page, with a note asking the user to review and
     merge them into the right place in the body (the script cannot know
     *where* in the document structure a dropped table row or bullet
     belongs — only that it's missing).

4. **Deliver the output file** the way this session supports (present the
   file, publish it, etc.) — never just describe the result in prose only.

## Tuning the sensitivity

The default similarity threshold (`--threshold 0.72`) works well for most
PDFs. If the user reports either:
- **false positives** (real content getting flagged as "missing" when it's
  actually there, just reformatted) → rerun with a lower threshold, e.g.
  `--threshold 0.6`, which is more lenient about what counts as "close
  enough."
- **false negatives** (the user knows something is missing but the script
  didn't catch it) → rerun with a higher threshold, e.g. `--threshold
  0.85`, which is stricter about what counts as "found."

## If no .md exists yet

If the user only supplied a PDF and wants a validated MarkItDown
conversion from scratch:

```bash
pip install markitdown pdfplumber --break-system-packages -q
python3 -m markitdown <input.pdf> > <input>.md
python3 scripts/compare_and_append.py <input.pdf> <input>.md <input>_validated.md
```

## Known limitations (mention these if relevant, don't over-explain unprompted)

- **Ordering, not placement.** Recovered content is grouped by source page
  and appended at the end — the script does not attempt to re-insert it
  into the correct spot in the document's existing structure. That's a
  human judgment call.
- **Some duplication is expected.** A dropped table cell can surface both
  as an individual recovered line and as part of a larger raw table-row
  block, since PDF table extraction is imperfect. This errs toward
  over-including rather than silently dropping something twice — better
  to review and delete a duplicate than miss content.
- **Not spell-check or prose-quality review.** This only checks for
  *presence* of content, not formatting fidelity, heading levels, or
  whether Markdown syntax rendered correctly.
- **Repeated running headers/footers are intentionally ignored** (page
  numbers, "page x/y", timestamps, and short lines that repeat on nearly
  every page) — these are treated as page furniture, not content, since
  Markdown conversions correctly drop them.

## Files in this skill

- `scripts/extract_pdf_chunks.py` — extracts the PDF's text and table
  content into an ordered, de-duplicated list of comparison chunks. Can
  be run standalone for debugging (`python3 scripts/extract_pdf_chunks.py
  <pdf> <out.json>`).
- `scripts/compare_and_append.py` — the main entry point; calls the
  extractor, fuzzy-matches every PDF chunk against the .md content, and
  appends whatever wasn't found.
- `references/approach.md` — implementation details on the chunking and
  fuzzy-matching strategy, for debugging or extending the skill.
