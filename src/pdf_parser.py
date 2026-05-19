from pathlib import Path

import fitz
import pdfplumber

from src.clause_splitter import detect_incomplete_boundary
from src.config import get_settings
from src.logging_setup import get_logger
from src.models import ExtractedTable, PageText
from src.ocr import ocr_fitz_page

logger = get_logger("pdf_parser")


def validate_pdf_size(pdf_path: Path) -> None:
    settings = get_settings()
    file_size = pdf_path.stat().st_size
    if file_size > settings.max_pdf_bytes:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"PDF file is {size_mb:.1f} MB, exceeds limit of {settings.MAX_PDF_SIZE_MB} MB."
        )


def extract_text_by_page(pdf_path: str | Path) -> list[PageText]:
    """
    Extracts text page by page preserving page numbers and block structure.
    Applies OCR fallback for scanned pages.
    """
    pdf_path = Path(pdf_path)
    validate_pdf_size(pdf_path)
    settings = get_settings()

    pages: list[PageText] = []

    with fitz.open(pdf_path) as doc:
        if len(doc) > settings.MAX_PAGES:
            logger.warning(
                "PDF has %d pages, exceeds MAX_PAGES=%d. Processing first %d only.",
                len(doc), settings.MAX_PAGES, settings.MAX_PAGES,
            )

        page_limit = min(len(doc), settings.MAX_PAGES)

        for page_index in range(page_limit):
            page = doc[page_index]
            page_number = page_index + 1

            blocks = page.get_text("blocks")
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

            block_texts = []
            for block in blocks:
                text = block[4].strip()
                if text:
                    block_texts.append(text)

            page_text = "\n\n".join(block_texts).strip()

            has_images = len(page.get_images()) > 0
            is_scanned = len(page_text) < 50 and has_images
            ocr_applied = False

            if is_scanned:
                logger.info("Page %d appears scanned, attempting OCR", page_number)
                ocr_text = ocr_fitz_page(page)
                if ocr_text:
                    page_text = ocr_text
                    ocr_applied = True
                    logger.info("OCR extracted %d chars from page %d", len(ocr_text), page_number)
                else:
                    logger.warning("OCR produced no text for page %d", page_number)

            pages.append(PageText(
                page_number=page_number,
                text=page_text,
                is_scanned=is_scanned,
                ocr_applied=ocr_applied,
            ))

    logger.info("Extracted text from %d pages", len(pages))
    return pages


def get_cross_page_boundaries(pages: list[PageText]) -> dict[int, str]:
    """
    Detect pages that end mid-sentence and return trailing fragments.
    Key is the page number whose trailing text should be prepended to the NEXT page.
    """
    boundaries: dict[int, str] = {}

    for page in pages:
        trailing = detect_incomplete_boundary(page.text)
        if trailing:
            boundaries[page.page_number] = trailing
            logger.info(
                "Page %d ends mid-sentence (%d chars trailing)",
                page.page_number, len(trailing),
            )

    return boundaries


def extract_tables(pdf_path: str | Path) -> list[ExtractedTable]:
    """
    Extracts tables with context (text immediately above each table).
    """
    pdf_path = Path(pdf_path)
    all_tables: list[ExtractedTable] = []

    with pdfplumber.open(pdf_path) as pdf:
        settings = get_settings()
        page_limit = min(len(pdf.pages), settings.MAX_PAGES)

        for page_index in range(page_limit):
            page = pdf.pages[page_index]
            page_number = page_index + 1

            tables = page.extract_tables() or []

            page_text = page.extract_text() or ""

            for table_index, table in enumerate(tables, start=1):
                context_before = _get_table_context(page, page_text, table_index)

                rows = []
                for row in table:
                    rows.append([cell if cell else None for cell in row])

                all_tables.append(ExtractedTable(
                    page_number=page_number,
                    table_number=table_index,
                    rows=rows,
                    context_before=context_before,
                ))

    logger.info("Extracted %d tables", len(all_tables))
    return all_tables


def _get_table_context(page, page_text: str, table_index: int) -> str | None:
    """Get up to 300 chars of text above the first table on a page."""
    if table_index != 1:
        return None

    try:
        tables_on_page = page.find_tables()
        if not tables_on_page:
            return None

        first_table = tables_on_page[0]
        table_top = first_table.bbox[1]

        words_above = [
            w for w in (page.extract_words() or [])
            if w["top"] < table_top
        ]

        if not words_above:
            return None

        words_above.sort(key=lambda w: (w["top"], w["x0"]))
        text_above = " ".join(w["text"] for w in words_above)

        return text_above[-300:] if len(text_above) > 300 else text_above

    except Exception:
        return None
