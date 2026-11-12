#!/usr/bin/env python3
"""
compare_and_append.py

Compares a MarkItDown-generated .md file against its source PDF (via
extract_pdf_chunks.py's output) and appends any PDF content that is
missing from the .md — content MarkItDown dropped, mangled, or skipped
(common with tables, sidebars, captions, multi-column layouts, and
image-based text).

Matching is fuzzy (difflib), not exact, because Markdown conversion
legitimately reformats whitespace, list markers, and punctuation. A PDF
line only counts as "missing" if nothing reasonably similar to it exists
anywhere in the .md.

Usage:
    python3 compare_and_append.py <input.pdf> <input.md> <output.md> [--threshold 0.72]
"""
import sys
import re
import json
import argparse
import subprocess
from pathlib import Path
from difflib import SequenceMatcher

SCRIPT_DIR = Path(__file__).parent


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[#*_`>:;\-–—•]+", " ", text)  # strip markdown/bullet/label punctuation
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url) -> text
    text = re.sub(r"\s+", " ", text).strip()
    return text


def md_to_norm_lines(md_text: str):
    """Break the MD file into normalized comparison units (lines + table cells)."""
    units = []
    for line in md_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Split markdown table rows into individual cells too, since MarkItDown
        # sometimes reorders or merges cells within a row.
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            for cell in cells:
                n = normalize(cell)
                if n:
                    units.append(n)
        n = normalize(line)
        if n:
            units.append(n)
    return units


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


MIN_SUBSTRING_LEN = 8  # guards against trivial matches like "1", "2", "co", "pm"


def is_present(chunk_norm: str, md_units, md_units_set, threshold: float = 0.72) -> bool:
    if not chunk_norm:
        return True
    # Fast path: exact match
    if chunk_norm in md_units_set:
        return True
    # Substring match — only meaningful once both sides are long enough that a
    # match isn't just a coincidental short token (units/numbers/labels).
    if len(chunk_norm) >= MIN_SUBSTRING_LEN:
        for unit in md_units:
            if len(unit) < MIN_SUBSTRING_LEN:
                continue
            if chunk_norm in unit or unit in chunk_norm:
                return True
    # Slow path: fuzzy match against similarly-sized units only (perf + accuracy).
    # Skip very short chunks here too — SequenceMatcher ratios on short strings
    # are noisy and prone to false positives.
    if len(chunk_norm) < MIN_SUBSTRING_LEN:
        return chunk_norm in " ".join(md_units)  # last-resort exact-ish check
    target_len = len(chunk_norm)
    for unit in md_units:
        if abs(len(unit) - target_len) > target_len * 0.6 + 20:
            continue
        if similar(chunk_norm, unit) >= threshold:
            return True
    return False


def group_missing_into_blocks(missing_chunks):
    """Group consecutive missing chunks (by page + order) into readable blocks."""
    blocks = []
    current = []
    last_page = None
    for c in missing_chunks:
        if current and c["page"] != last_page:
            blocks.append(current)
            current = []
        current.append(c)
        last_page = c["page"]
    if current:
        blocks.append(current)
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("md_path")
    parser.add_argument("output_md_path")
    parser.add_argument("--threshold", type=float, default=0.72)
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    md_path = Path(args.md_path)
    out_path = Path(args.output_md_path)

    # Step 1: extract PDF ground-truth chunks
    chunks_json = out_path.with_suffix(".pdfchunks.json")
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "extract_pdf_chunks.py"), str(pdf_path), str(chunks_json)],
        check=True,
    )
    chunks = json.loads(chunks_json.read_text(encoding="utf-8"))

    # Step 2: load MD content
    md_text = md_path.read_text(encoding="utf-8")
    md_units = md_to_norm_lines(md_text)
    md_units_set = set(md_units)

    # Step 3: find missing chunks
    # Re-normalize each PDF chunk with the SAME punctuation-stripping `normalize()`
    # used for the MD side (the extractor's own "norm" field only lowercases and
    # collapses whitespace — it does not strip punctuation). Comparing those two
    # different normalization levels against each other caused false "missing"
    # results whenever a chunk and its MD counterpart used different punctuation
    # for the same separator (e.g. "Option 2: X" vs "Option 2 – X").
    missing = []
    for c in chunks:
        if not c["raw"].strip():
            continue
        chunk_norm = normalize(c["raw"])
        if not chunk_norm:
            continue
        if not is_present(chunk_norm, md_units, md_units_set, threshold=args.threshold):
            missing.append(c)

    # Step 4: build report + appended section
    report_lines = [
        f"PDF chunks checked: {len(chunks)}",
        f"Missing from MD: {len(missing)}",
    ]

    if missing:
        blocks = group_missing_into_blocks(missing)
        append_lines = [
            "",
            "---",
            "",
            "## ⚠️ Content recovered from source PDF",
            "",
            (
                "_The section below was found in the original PDF but was missing "
                "or altered in the MarkItDown conversion above. Added automatically "
                "by the PDF↔MD validator skill — review and merge into the body "
                "text where it belongs, then delete this note._"
            ),
            "",
        ]
        for block in blocks:
            page = block[0]["page"]
            append_lines.append(f"### Missing content — page {page}")
            for item in block:
                if item["type"] == "table_row":
                    append_lines.append(f"- `{item['raw']}`")
                else:
                    append_lines.append(f"- {item['raw']}")
            append_lines.append("")

        new_md = md_text.rstrip("\n") + "\n" + "\n".join(append_lines) + "\n"
    else:
        new_md = md_text

    out_path.write_text(new_md, encoding="utf-8")
    chunks_json.unlink(missing_ok=True)

    print("\n".join(report_lines))
    print(f"Updated MD written to: {out_path}")


if __name__ == "__main__":
    main()
