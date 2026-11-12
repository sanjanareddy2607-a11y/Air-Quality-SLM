# MarkItDown PDF ↔ MD Validator — Claude Skill

A Claude Agent Skill that validates a Markdown file generated from a PDF by
Microsoft's **MarkItDown** tool. It compares the `.md` output against the
source PDF, finds anything MarkItDown dropped, mangled, or scrambled
(table rows, nested lists inside table cells, multi-column layouts, FAQ
accordions, sidebar/callout text), and **appends the missing content back**
into the `.md` file under a clearly-marked review section.

## What's in this folder

```
markitdown-pdf-validator/
├── SKILL.md                     ← the skill definition Claude reads
├── scripts/
│   ├── extract_pdf_chunks.py    ← pulls ground-truth text/tables out of the PDF
│   └── compare_and_append.py    ← compares + appends missing content to the MD
└── references/
    └── approach.md              ← implementation notes, known bugs fixed, limitations
```

## How to use this with Claude

1. Open this folder in Claude (claude.ai, Claude Code, or Cowork) as a
   **skill** — e.g. drop the whole `markitdown-pdf-validator/` folder
   wherever your skills are loaded from, or upload the `.zip` if your
   environment accepts packaged skills.
2. In your conversation, attach the **PDF** and its **MarkItDown-generated
   `.md` file**.
3. Ask Claude to validate / check / compare the two, or to append any
   content MarkItDown missed. Claude will run the scripts and hand you
   back an updated `.md` with a `## ⚠️ Content recovered from source PDF`
   section if anything was missing.

## Running it yourself (without Claude)

You can also run this directly from a terminal — handy for testing inside
VS Code:

```bash
pip install pdfplumber --break-system-packages   # one-time setup
python3 scripts/compare_and_append.py your_file.pdf your_file.md your_file_validated.md
```

Optional: tune sensitivity with `--threshold` (default `0.72`). Lower it
(e.g. `0.6`) if it's flagging content that's really just reformatted;
raise it (e.g. `0.85`) if it's missing things you know are genuinely gone.

## Opening this in VS Code

1. Unzip `markitdown-pdf-validator.zip` anywhere on your machine.
2. In VS Code: **File → Open Folder…** and select the extracted
   `markitdown-pdf-validator` folder.
3. The `scripts/*.py` files are plain Python — VS Code's Python extension
   will pick them up automatically for linting/running (`Run Python File`
   or `F5`), no extra configuration needed. A `.vscode/settings.json` is
   included with sensible defaults (format-on-save off, so the scripts'
   existing style is preserved).

## Known limitations (see `references/approach.md` for detail)

- Recovered content is appended at the end, grouped by PDF page — it is
  **not** re-inserted into the "correct" spot in the document structure.
  That's left as a human judgment call.
- Multi-column PDF layouts and decorative/letter-spaced headings can
  produce text that looks very different from a cleanly-reworded `.md`
  version of the same content, even though the information itself is
  present. These get flagged for review rather than silently ignored —
  better to double-check something that's actually fine than miss
  something that isn't.
