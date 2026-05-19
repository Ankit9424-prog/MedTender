# MedTender Extractor

AI-assisted tender requirement extraction system for Nepal medical procurement. Extracts critical clauses, deadlines, eligibility criteria, equipment lists, and financial requirements from government tender PDFs — with built-in verification to catch AI hallucinations.

## Features

- **PDF Text Extraction** — Page-by-page text extraction preserving block structure and page numbers using PyMuPDF
- **OCR Fallback** — Automatic Tesseract OCR for scanned pages with English + Nepali language support
- **Table Extraction** — Structured table parsing via pdfplumber with context-aware header detection
- **Rule-Based Clause Detection** — Pattern-matched extraction of important clauses (deadlines, bid security, eligibility, disqualification criteria) without any AI dependency
- **LLM Structured Extraction** — AWS Bedrock (Claude) powered extraction using tool_use for guaranteed structured JSON output
- **BOQ/Equipment Extraction** — Identifies Bill of Quantities tables and extracts medical equipment items with quantities, specifications, and delivery details
- **Three-State Verification** — Every AI-extracted item is verified against source text:
  - `exact_match` — text is a verified substring of source
  - `fuzzy_needs_review` — similar but requires human review
  - `failed` — likely hallucination, do not use
- **Cross-Page Detection** — Handles clauses split across page boundaries
- **Nepali/BS Date Support** — Preserves Bikram Sambat dates and Devanagari text without conversion
- **Audit Logging** — Every extraction run is logged with file metadata, model used, and verification results

## Architecture

```
app.py                  # Streamlit web interface
src/
├── config.py           # Settings, AWS credentials, temp directory management
├── models.py           # Pydantic models (PageText, CandidateClause, LLMExtractedItem, etc.)
├── pdf_parser.py       # PDF text + table extraction with PyMuPDF and pdfplumber
├── ocr.py              # Tesseract OCR fallback for scanned pages
├── clause_splitter.py  # Cross-page boundary detection
├── rule_extractor.py   # Pattern-based clause candidate extraction
├── boq_extractor.py    # Bill of Quantities / equipment list extraction
├── llm_extractor.py    # AWS Bedrock Claude extraction with tool_use
├── verifier.py         # Three-state verification (exact/fuzzy/failed)
├── audit.py            # Extraction run audit logging
└── logging_setup.py    # Structured logging configuration
tests/
├── test_clause_splitter.py
├── test_rule_extractor.py
├── test_boq_extractor.py
├── test_llm_extractor.py
└── test_verifier.py
```

## Requirements

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, for scanned PDFs)
- AWS account with Bedrock access (optional, for AI extraction)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/medtender-extractor.git
cd medtender-extractor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# AWS Bedrock credentials (required for AI extraction)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1

# Model configuration
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# Application limits
MAX_PDF_SIZE_MB=50
MAX_PAGES=200
LLM_BATCH_SIZE=15
LLM_MAX_RETRIES=3
```

### Optional: Install Tesseract OCR

For scanned PDF support:

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-nep

# macOS
brew install tesseract
brew install tesseract-lang  # includes Nepali

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH and install eng + nep language data
```

## Usage

### Run the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, then:

1. Upload a tender PDF
2. View extracted page text and tables
3. Review rule-based clause candidates (no AI needed)
4. Optionally enable AI extraction for structured details
5. Review verification results — only `exact_match` items are safe to use without manual checking
6. Download results as JSON or CSV

### Extraction Modes

| Mode | Requires AI | Description |
|------|------------|-------------|
| Rule-based | No | Pattern matching for important clauses. High recall, lower precision. |
| AI Structured | Yes (AWS Bedrock) | LLM extracts structured fields with exact text preservation. |
| BOQ Extraction | No | Identifies equipment/supply tables and extracts item details. |

## Running Tests

```bash
pytest
```

## How It Works

### Pipeline

```
PDF Upload
    │
    ├── Text Extraction (PyMuPDF blocks, sorted by position)
    │       └── OCR fallback for scanned pages
    │
    ├── Table Extraction (pdfplumber)
    │       └── BOQ identification and item extraction
    │
    ├── Rule-Based Extraction
    │       ├── Category classification (deadline, eligibility, financial, etc.)
    │       ├── Risk term detection (shall, must, disqualified, etc.)
    │       ├── Amount and date extraction
    │       └── Cross-page boundary handling
    │
    └── [Optional] LLM Extraction
            ├── Batched Bedrock API calls with tool_use
            ├── Dynamic batch sizing based on token estimates
            ├── Retry with exponential backoff
            └── Three-state verification against source text
```

### Verification Logic

The verifier ensures AI doesn't alter legally binding text:

- **Exact match**: Extracted text is a case-insensitive substring of the source clause
- **Fuzzy needs review**: >85% similarity + all critical legal terms preserved + amounts/dates intact
- **Failed**: Low similarity, missing critical terms, or altered amounts/dates

Critical terms tracked: `shall`, `must`, `may`, `and/or`, `minimum`, `maximum`, `unless`, `except`, `within`, `not less than`, `not more than`, etc.

## Important Disclaimer

This is an AI-assisted extraction tool. All extracted items **must** be verified against the original document by a qualified professional before use in bid preparation. AI extraction may contain errors — never rely solely on these results.

## License

MIT
