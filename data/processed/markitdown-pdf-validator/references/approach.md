# Implementation notes

## Why fuzzy matching instead of a plain diff

A naive `diff` between the PDF's raw text and the .md file produces almost
100% "differences," because:
- Markdown reformats whitespace, bullets, and heading markers.
- Table cells get reordered or flattened into different structures.
- Links become `[text](url)` instead of bare text.
- Line-wrapping differs completely (PDF wraps by pixel width, Markdown
  by paragraph).

So the script never diffs the raw documents directly. Instead it:

1. Breaks the **PDF** into short, atomic content units ("chunks") — one
   per visual line of text, plus one per table cell/row.
2. Breaks the **MD** into the same kind of atomic units.
3. Normalizes both (lowercase, strip Markdown punctuation, collapse
   whitespace, resolve `[text](url)` → `text`).
4. For every PDF chunk, checks whether *anything reasonably similar*
   exists anywhere in the MD's normalized units — not just at the
   "corresponding" position, since position doesn't carry over between a
   PDF layout and a Markdown document.

## Matching tiers (in `is_present()`)

1. **Exact match** against the full set of normalized MD units — cheap,
   catches the large majority of correctly-converted content.
2. **Substring match** — a PDF chunk counts as present if it's a substring
   of (or contains) some MD unit, but only once both strings are at least
   `MIN_SUBSTRING_LEN` (8) characters. This threshold matters: without it,
   short leftover tokens in the MD (page numbers, unit labels like "kg",
   chemical formulas like "CO", table remnants like "1", "2") trivially
   satisfy `x in y` for almost any longer string, causing false "already
   present" results. This was caught during testing — an earlier version
   without the length guard missed the majority of deliberately-removed
   test content because short numeric/unit fragments left behind in the
   MD were matching everything.
3. **Fuzzy match** via `difflib.SequenceMatcher`, restricted to MD units of
   comparable length (within `60% + 20 chars` of the chunk's length, for
   performance and to avoid comparing a one-word chunk against a
   full-paragraph unit). Default similarity cutoff: `0.72`.

## Page furniture filtering

Two kinds of PDF content are deliberately excluded before comparison,
because their absence from the .md is *correct* behavior, not a
conversion failure:

- **Page numbers / timestamps**: matched with regexes (`\d+/\d+`,
  date-time patterns) and dropped during extraction.
- **Repeated running headers/footers**: any short line (under 60 chars)
  that appears on nearly every page (`>= total_pages - 1`, with a floor of
  3) is treated as furniture and dropped. This is why the skill correctly
  ignores things like a document title repeated on every page, or a
  tracking URL appended to every page footer.

## Table handling

`pdfplumber.Page.extract_tables()` is used in addition to
`extract_text()`. Table rows are flattened into `cell1 | cell2 | ...`
strings and treated as their own chunk type (`table_row`), separately
from the page's plain-text chunks. On the MD side, any line starting with
`|` is split on `|` and each cell is *also* added as its own comparison
unit (in addition to the whole line), since MarkItDown sometimes merges,
reorders, or drops individual cells rather than whole rows.

Caveat: `pdfplumber` sometimes returns a table cell's text as a single
multi-line string (embedded `\n` characters) rather than splitting it
into separate rows. This means a dropped multi-item table cell (e.g. a
bulleted list of options crammed into one cell) can show up as both a
group of short individual-line chunks *and* one large multi-line blob
chunk. Both get flagged as missing if genuinely absent, which produces
some duplication in the appended output — treated as an acceptable
trade-off in favor of not missing content, per the skill's own
documented limitations.

## Bugs found and fixed during validation (keep in mind when modifying)

Three real bugs surfaced while testing this skill against an actual PDF +
MD pair, and are worth remembering if you touch the matching logic:

1. **Trivial substring matches.** Without a minimum-length guard, short
   leftover tokens in the MD (page numbers, unit labels like "kg",
   chemical formulas like "CO") trivially satisfy `x in y` for almost any
   longer PDF chunk, causing false "already present" results. Fixed with
   `MIN_SUBSTRING_LEN = 8`.
2. **Two different normalization functions.** The extractor's own `"norm"`
   field only lowercases and collapses whitespace — it does **not** strip
   punctuation. The comparison script has its own, stricter `normalize()`
   that also strips Markdown/bullet punctuation. Comparing a chunk's loose
   `"norm"` against MD units produced by the strict `normalize()` caused
   real content (e.g. "Option 2: X" in the PDF vs "Option 2 – X" in the
   MD) to be flagged as missing purely because of a colon-vs-dash
   difference. Fix: `compare_and_append.py` now always re-normalizes
   `c["raw"]` with the same `normalize()` used for the MD side before
   comparing — never compare the two sides using different normalization
   rules.
3. **Footer URLs with a page number baked in.** A PDF footer like
   `https://example.com/page?... 3/12` differs on every page (different
   trailing page number), so simple "repeats on every page" dedup never
   catches it. Fixed with `is_url_furniture()`, which drops any line
   that's mostly a bare URL (optionally with a trailing `N/M` page
   fragment) regardless of what page number is appended — including when
   the URL's leading characters were clipped by an overlapping-text-layer
   artifact (`ttps://...` instead of `https://...`, common in PDFs printed
   from web pages). This filter applies to both plain-text lines and table
   cells.

## Known remaining limitations (by design, not bugs)

- **Multi-column layout bleed-through.** When a PDF has several side-by-side
  cards/columns (e.g. four "application" cards in a row), `extract_text()`
  often interleaves their text row-by-row (e.g. "Cartridges Proven" /
  "system accuracy" where "Cartridges system" and "Proven accuracy" were
  two separate adjacent cards). A hand-cleaned MD that correctly separates
  these will not textually match the jumbled PDF extraction, so it gets
  flagged as "missing" even though the information is present in reworded
  form. This is inherent to line-based text extraction from multi-column
  layouts and isn't something fuzzy string matching can fully solve —
  flag it for human review rather than suppress it, since the alternative
  (suppressing it) risks hiding genuinely missing content too.
- **Decorative/stylised text extracted as doubled letters.** Some PDFs
  render letter-spaced or styled headings in a way that `pdfplumber`
  extracts with every letter doubled (e.g. "YYoouu ccaannnnoott..." for
  "You cannot..."). This is a real PDF-rendering artifact, not a bug in
  this skill, and will correctly be flagged as a mismatch against a clean
  MD rendering of the same text — again, a legitimate "textually
  different" flag even though the meaning is preserved.

## Extending this skill

- To support DOCX/PPTX sources instead of PDF, swap `extract_pdf_chunks.py`
  for an equivalent extractor and keep `compare_and_append.py` unchanged
  (it only depends on the chunk JSON schema: `{page, type, raw, norm}`).
- To reduce duplicate output, a de-duplication pass could collapse a
  `table_row` blob chunk against its already-flagged constituent line
  chunks before writing the report — not implemented here to keep the
  script simple and biased toward not silently dropping content.
