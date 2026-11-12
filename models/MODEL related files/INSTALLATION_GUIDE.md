# PDF-Markdown Validator Skill - Installation & Usage Guide

## 📦 What You've Downloaded

A complete, production-ready skill for validating PDF-to-Markdown conversions and reconciling missing content.

**File:** `pdf-md-validator.zip` (24 KB)

## 🚀 Installation Steps

### Step 1: Extract the ZIP File

**On Windows:**
1. Right-click `pdf-md-validator.zip`
2. Select "Extract All..."
3. Choose your workspace location
4. Click Extract

**On Mac/Linux:**
```bash
unzip pdf-md-validator.zip
cd pdf-md-validator
```

### Step 2: Open in VS Code

```bash
code pdf-md-validator
```

Or drag the folder into VS Code.

### Step 3: Install Dependencies

Open VS Code terminal (Ctrl+` or View → Terminal) and run:

```bash
npm install
```

This installs:
- `pdf-parse`: For reading PDF files
- `marked`: For parsing Markdown

**First time? This takes ~30 seconds**

### Step 4: Verify Installation

```bash
node scripts/compare.js --help
```

You should see usage information. ✅ Ready to go!

## 📋 What's Included

```
pdf-md-validator/
├── 📖 SKILL.md                  Complete documentation
├── 📖 README.md                 Full guide and reference
├── 📖 QUICKSTART.md             5-minute quick start
├── 🛠️  scripts/
│   ├── compare.js               Comparison engine
│   └── merger.js                Content merger
├── 📚 references/
│   └── best-practices.md        Usage guidelines
├── ⚙️  .vscode/
│   ├── settings.json            VS Code settings
│   ├── launch.json              Debug configs
│   └── extensions.json          Recommended extensions
├── 📄 package.json              Dependencies
├── 🔧 config.example.json       Configuration template
└── 📌 .gitignore                Git ignore rules
```

## 🎯 5-Minute Start

### Prepare Your Files

You need:
1. A PDF file (e.g., `document.pdf`)
2. Its Markdown conversion from MarkItDown (e.g., `document.md`)

### Run Comparison

```bash
node scripts/compare.js --pdf document.pdf --md document.md
```

**Output example:**
```
=== VALIDATION REPORT ===

Completeness: 87%

Missing Sections (3):
  - Advanced Features
  - References
  - Appendix

Recommendations:
  [HIGH] Missing 13% of text content
    → Review PDF and MD for significant gaps
```

### If Completeness < 90%: Merge Content

```bash
node scripts/merger.js --pdf document.pdf --md document.md \
  --output document.merged.md
```

This creates `document.merged.md` with all content from the PDF merged in!

### Verify Result

```bash
node scripts/compare.js --pdf document.pdf --md document.merged.md
```

Completeness should now be higher. ✅

## 🔧 Common Use Cases

### Use Case 1: Quality Check

After converting a PDF to Markdown with MarkItDown:

```bash
node scripts/compare.js --pdf article.pdf --md article.md
```

**Get:** Completeness percentage, what's missing, recommendations

### Use Case 2: Fix a Document

To make a Markdown complete:

```bash
# See what's missing
node scripts/compare.js --pdf article.pdf --md article.md --verbose

# Add missing content
node scripts/merger.js --pdf article.pdf --md article.md --output article.complete.md

# Verify it's complete
node scripts/compare.js --pdf article.pdf --md article.complete.md
```

### Use Case 3: Batch Processing

Handle multiple documents:

```bash
# Process all PDFs in current directory
for pdf in *.pdf; do
  md="${pdf%.pdf}.md"
  if [ -f "$md" ]; then
    echo "Validating: $pdf"
    node scripts/compare.js --pdf "$pdf" --md "$md" --verbose
    node scripts/merger.js --pdf "$pdf" --md "$md" --output "${pdf%.pdf}.merged.md"
    echo "Complete: ${pdf%.pdf}.merged.md"
  fi
done
```

### Use Case 4: Save Reports

Generate JSON reports for tracking:

```bash
node scripts/compare.js --pdf doc.pdf --md doc.md \
  --output report.json --verbose
```

**Output:** `report.json` with detailed metrics, missing sections, recommendations

## 💡 Advanced Features

### Comparison Modes

```bash
# Fuzzy matching (tolerant of formatting differences)
node scripts/compare.js --pdf doc.pdf --md doc.md --mode fuzzy

# Strict matching (exact content required)
node scripts/compare.js --pdf doc.pdf --md doc.md --mode strict
```

### Merge Options

```bash
# With validation
node scripts/merger.js --pdf doc.pdf --md doc.md --output out.md --validate

# Interactive (review before merge)
node scripts/merger.js --pdf doc.pdf --md doc.md --output out.md --interactive

# Without source markers
node scripts/merger.js --pdf doc.pdf --md doc.md --output out.md --no-markers
```

### Verbose Output

```bash
# See detailed processing steps
node scripts/compare.js --pdf doc.pdf --md doc.md --verbose
```

## 📚 Documentation

Inside the extracted folder:

- **QUICKSTART.md** - 5-minute introduction
- **README.md** - Complete guide with examples
- **SKILL.md** - Full feature documentation
- **references/best-practices.md** - Advanced usage, troubleshooting

## 🔌 Integration with VS Code

### Method 1: Direct Terminal

```bash
# In VS Code terminal
node scripts/compare.js --pdf input.pdf --md input.md
```

### Method 2: Debug Configurations

1. Open `.vscode/launch.json`
2. Edit file paths in launch configs
3. Press F5 or Run → Start Debugging

Configurations included:
- **Compare PDF-MD** - Run comparison
- **Merge Content** - Run content merger
- **Node: Current File** - Run any Node script

### Method 3: VS Code Tasks

Add to `.vscode/tasks.json` for custom tasks

## ⚡ Performance Tips

**For large PDFs (>50 pages):**
- Split PDF before processing
- Process one file at a time
- Monitor memory usage

**For batch processing:**
- Process 5-10 files in parallel
- Save reports for tracking
- Review incremental results

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
npm install
npm install pdf-parse marked
```

### Issue: Very low completeness despite good visual inspection
```bash
# Try different comparison mode
node scripts/compare.js --pdf doc.pdf --md doc.md --mode strict
```

### Issue: Merge isn't adding expected content
```bash
# Check what's actually missing
node scripts/compare.js --pdf doc.pdf --md doc.md --verbose --output report.json

# Review report.json for missing content types
```

### Issue: Large PDF processing slowly
- Split PDF: Use `--chunk-size 50`
- Or split manually before processing
- Increase Node memory: `node --max-old-space-size=4096 scripts/compare.js`

### Issue: "Cannot create output file"
- Check directory permissions
- Ensure output directory exists
- Try absolute path for output file

## 🔄 Integration with Git

Track your validation work:

```bash
# Initial setup
git init
git add .
git commit -m "Add pdf-md-validator skill"

# After validation/merge
git add *.merged.md report.json
git commit -m "docs: reconcile PDF content (87% -> 98%)"

# For team sharing
git push origin main
```

## 📊 Understanding Reports

### JSON Report Structure

```json
{
  "completeness": 87,
  "metrics": {
    "pdf": {
      "characters": 50000,
      "pages": 12,
      "sections": 8
    },
    "md": {
      "characters": 43500,
      "headings": 8,
      "paragraphs": 45
    }
  },
  "missingElements": {
    "sections": [...],
    "tables": [...],
    "textGaps": [...]
  },
  "recommendations": [...]
}
```

### Completeness Interpretation

| Score | Status | Action |
|-------|--------|--------|
| 95-100% | ✅ Excellent | Use as-is |
| 85-95% | 🟡 Good | Minor review |
| 70-85% | 🟠 Fair | Merge recommended |
| <70% | ❌ Poor | Re-convert PDF |

## 🚀 Next Steps

1. **Extract** the ZIP file
2. **Run** `npm install`
3. **Try** the quick start example
4. **Read** QUICKSTART.md for more examples
5. **Integrate** into your workflow

## 📞 Support Resources

Inside the extracted folder:
- `QUICKSTART.md` - Quick examples
- `README.md` - Complete reference
- `SKILL.md` - All features documented
- `references/best-practices.md` - Advanced usage

## 🎓 Example Workflow

```bash
# 1. Extract and install
unzip pdf-md-validator.zip
cd pdf-md-validator
npm install

# 2. Copy your files to the directory
cp ~/Downloads/report.pdf .
cp ~/Downloads/report.md .

# 3. Validate conversion
node scripts/compare.js --pdf report.pdf --md report.md

# Shows: Completeness: 87%

# 4. Fix the document
node scripts/merger.js --pdf report.pdf --md report.md --output report.merged.md

# 5. Verify it's complete
node scripts/compare.js --pdf report.pdf --md report.merged.md

# Shows: Completeness: 98%

# 6. Use report.merged.md in your documentation!
```

## ✨ You're All Set!

You now have a professional-grade tool for validating and reconciling PDF-to-Markdown conversions.

**Quick command reference:**
```bash
# Validate
node scripts/compare.js --pdf doc.pdf --md doc.md

# Fix
node scripts/merger.js --pdf doc.pdf --md doc.md --output doc.merged.md

# Report
node scripts/compare.js --pdf doc.pdf --md doc.md --output report.json
```

---

**Questions?** Check the documentation inside the skill folder, or enable `--verbose` for detailed output.

**Ready?** Extract the ZIP and run: `npm install` then try a validation!
