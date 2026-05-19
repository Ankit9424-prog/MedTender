"""
OCR fallback for scanned PDF pages.

Uses pytesseract with English + Nepali language support.
Gracefully degrades if Tesseract is not installed.
"""

from src.logging_setup import get_logger

logger = get_logger("ocr")

_tesseract_available: bool | None = None


def is_tesseract_available() -> bool:
    global _tesseract_available
    if _tesseract_available is not None:
        return _tesseract_available

    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _tesseract_available = True
        logger.info("Tesseract OCR available")
    except Exception:
        _tesseract_available = False
        logger.warning(
            "Tesseract not installed or not found in PATH. "
            "Scanned pages will not be OCR'd. "
            "Install Tesseract and eng+nep language packs for full functionality."
        )

    return _tesseract_available


def ocr_page_from_pixmap(pixmap_bytes: bytes, width: int, height: int) -> str:
    """
    Run OCR on a page rendered as a pixmap.

    Args:
        pixmap_bytes: Raw pixel data from PyMuPDF page.get_pixmap().samples
        width: Pixmap width
        height: Pixmap height

    Returns:
        Extracted text, or empty string if OCR unavailable.
    """
    if not is_tesseract_available():
        return ""

    try:
        import pytesseract
        from PIL import Image

        image = Image.frombytes("RGB", (width, height), pixmap_bytes)

        image = image.convert("L")

        langs = "eng"
        try:
            available_langs = pytesseract.get_languages()
            if "nep" in available_langs:
                langs = "eng+nep"
                logger.info("Using eng+nep language pack for OCR")
        except Exception:
            pass

        text = pytesseract.image_to_string(
            image,
            lang=langs,
            config="--psm 6",
        )

        return text.strip()

    except Exception as e:
        logger.error("OCR failed: %s", str(e))
        return ""


def ocr_fitz_page(page) -> str:
    """
    Run OCR on a PyMuPDF page object.

    Args:
        page: fitz.Page object

    Returns:
        Extracted text, or empty string if OCR unavailable.
    """
    if not is_tesseract_available():
        return ""

    try:
        pixmap = page.get_pixmap(dpi=300)
        return ocr_page_from_pixmap(pixmap.samples, pixmap.width, pixmap.height)
    except Exception as e:
        logger.error("Failed to render page for OCR: %s", str(e))
        return ""
