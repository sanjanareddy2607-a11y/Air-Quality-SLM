#!/usr/bin/env python3
"""
extract_pdf_chunks.py

Extracts the PDF's ground-truth content as a flat, ordered list of text
"chunks" (roughly paragraph/line granularity), plus any table content
rendered as pipe-delimited rows. This is the reference content that the
MarkItDown .md output will be checked against.

Usage:
    python3 extract_pdf_chunks.py <input.pdf> <output_chunks.json>
"""
import sys
import json
import re
import pdfplumber


def normalize_text(text: str) -> str:
    """Collapse whitespace for reliable matching; keep original for display."""
    return re.sub(r"\s+", " ", text).strip()


# Matches a URL even if its leading character(s) were clipped — a common
# artifact when a PDF has overlapping/duplicated text layers (e.g. PDFs
# printed from a web page), which can turn "https://..." into "ttps://...".
URL_RE = re.compile(r"(?:https?://|w?ttps?://|www\.)\S+", re.IGNORECASE)


def is_url_furniture(line: str) -> bool:
    """A line that is mostly a bare URL (optionally with a trailing page
    number like '... 3/12') is page-footer furniture, not content — this
    holds even when a page number is baked into the URL text itself, which
    defeats simple exact-repetition dedup across pages."""
    match = URL_RE.search(line)
    if not match:
        return False
    url_span = match.end() - match.start()
    remainder = (line[:match.start()] + line[match.end():]).strip()
    # Allow a trailing/leading page-number fragment like "3/12" alongside the URL
    if remainder and not re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", remainder):
        return False
    return url_span >= 0.5 * len(line)


def extract_chunks(pdf_path: str):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # --- Paragraph / line text ---
            text = page.extract_text() or ""
            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                # Skip obvious page-furniture (page numbers, bare URLs repeated
                # on every page, timestamp headers) — these are expected to be
                # dropped by converters and aren't "content".
                if re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", line):
                    continue
                if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}", line):
                    continue
                if is_url_furniture(line):
                    continue
                chunks.append({
                    "page": page_num,
                    "type": "text",
                    "raw": line,
                    "norm": normalize_text(line).lower(),
                })

            # --- Tables ---
            for table in page.extract_tables():
                for row in table:
                    cells = [c.strip() if c else "" for c in row]
                    if not any(cells):
                        continue
                    raw_row = " | ".join(cells)
                    if is_url_furniture(raw_row):
                        continue
                    chunks.append({
                        "page": page_num,
                        "type": "table_row",
                        "raw": raw_row,
                        "norm": normalize_text(raw_row).lower(),
                    })

    # De-duplicate consecutive identical chunks (headers/footers repeated per page)
    deduped = []
    seen_counts = {}
    for c in chunks:
        seen_counts[c["norm"]] = seen_counts.get(c["norm"], 0) + 1
    total_pages = None
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    for c in chunks:
        # If a short line repeats on (almost) every page, it's running
        # header/footer furniture, not content — drop it.
        if total_pages and total_pages > 2 and len(c["norm"]) < 60:
            if seen_counts[c["norm"]] >= max(3, total_pages - 1):
                continue
        deduped.append(c)

    return deduped


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 extract_pdf_chunks.py <input.pdf> <output_chunks.json>")
        sys.exit(1)

    pdf_path, out_path = sys.argv[1], sys.argv[2]
    chunks = extract_chunks(pdf_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(chunks)} content chunks from {chunks[-1]['page'] if chunks else 0} pages -> {out_path}")


if __name__ == "__main__":
    main()
