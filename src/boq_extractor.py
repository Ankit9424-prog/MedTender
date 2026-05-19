"""
BOQ (Bill of Quantities) / Schedule of Requirements extractor.

Identifies BOQ tables in extracted PDF tables, maps columns to standard fields,
and returns structured equipment/item data for medical supply tenders.
"""

import re

from src.logging_setup import get_logger
from src.models import ExtractedTable

logger = get_logger("boq_extractor")

BOQ_HEADER_PATTERNS: dict[str, re.Pattern] = {
    "sn": re.compile(
        r"^(s\.?\s*n\.?|sl\.?\s*no\.?|क्र\.?\s*सं\.?|सि\.?\s*नं\.?|#|no\.?)$",
        re.IGNORECASE,
    ),
    "item_name": re.compile(
        r"(item|description|particular|name|specification|सामग्री|विवरण|सामानको\s*नाम|goods)",
        re.IGNORECASE,
    ),
    "quantity": re.compile(
        r"^(qty\.?|quantity|मात्रा|परिमाण|संख्या|nos\.?|units?)$",
        re.IGNORECASE,
    ),
    "unit": re.compile(
        r"^(unit|इकाई|एकाइ|uom|unit\s*of\s*measure)$",
        re.IGNORECASE,
    ),
    "unit_price": re.compile(
        r"(unit\s*(?:rate|price|cost)|rate|दर|एकाइ\s*मूल्य|per\s*unit)",
        re.IGNORECASE,
    ),
    "total_price": re.compile(
        r"(total\s*(?:price|cost|amount)|amount|रकम|जम्मा|कुल\s*मूल्य)",
        re.IGNORECASE,
    ),
    "specifications": re.compile(
        r"(spec|technical\s*spec|विशिष्टता|technical\s*requirement|model|brand)",
        re.IGNORECASE,
    ),
    "lot": re.compile(
        r"^(lot|group|समूह|लट)$",
        re.IGNORECASE,
    ),
    "delivery_location": re.compile(
        r"(delivery\s*(?:location|place|point|site)|destination|स्थान|ठाउँ|सुपुर्दगी)",
        re.IGNORECASE,
    ),
    "delivery_period": re.compile(
        r"(delivery\s*(?:period|time|date|schedule)|lead\s*time|अवधि|समयावधि|सुपुर्दगी\s*अवधि)",
        re.IGNORECASE,
    ),
    "remarks": re.compile(
        r"^(remarks?|कैफियत|note)$",
        re.IGNORECASE,
    ),
}

BOQ_TABLE_INDICATORS = [
    re.compile(r"bill\s*of\s*quantit", re.IGNORECASE),
    re.compile(r"schedule\s*of\s*requirement", re.IGNORECASE),
    re.compile(r"list\s*of\s*(goods|items|equipment|materials)", re.IGNORECASE),
    re.compile(r"(सामग्री|सामानको)\s*(सूची|विवरण)", re.IGNORECASE),
    re.compile(r"BOQ", re.IGNORECASE),
    re.compile(r"price\s*schedule", re.IGNORECASE),
    re.compile(r"schedule\s*of\s*supply", re.IGNORECASE),
    re.compile(r"equipment\s*list", re.IGNORECASE),
    re.compile(r"item\s*list", re.IGNORECASE),
]

TOTAL_ROW_PATTERNS = re.compile(
    r"^(total|grand\s*total|sub\s*total|जम्मा|कुल|vat|tax|discount)$",
    re.IGNORECASE,
)


def is_boq_table(table: ExtractedTable) -> bool:
    if not table.rows or len(table.rows) < 2:
        return False

    header_row = table.rows[0]
    header_text = " ".join(str(cell or "") for cell in header_row).strip()

    if _check_context_indicators(table.context_before):
        if _has_minimum_boq_columns(header_row):
            return True

    if _has_minimum_boq_columns(header_row):
        if _check_header_indicators(header_text):
            return True
        matched_cols = _count_matched_columns(header_row)
        if matched_cols >= 3:
            return True

    return False


def _check_context_indicators(context: str | None) -> bool:
    if not context:
        return False
    for pattern in BOQ_TABLE_INDICATORS:
        if pattern.search(context):
            return True
    return False


def _check_header_indicators(header_text: str) -> bool:
    for pattern in BOQ_TABLE_INDICATORS:
        if pattern.search(header_text):
            return True
    return False


def _has_minimum_boq_columns(header_row: list[str | None]) -> bool:
    has_item = False
    has_qty_or_amount = False

    for cell in header_row:
        if not cell:
            continue
        cell_clean = cell.strip()
        if BOQ_HEADER_PATTERNS["item_name"].search(cell_clean):
            has_item = True
        if (BOQ_HEADER_PATTERNS["quantity"].search(cell_clean) or
                BOQ_HEADER_PATTERNS["unit_price"].search(cell_clean) or
                BOQ_HEADER_PATTERNS["total_price"].search(cell_clean)):
            has_qty_or_amount = True

    return has_item and has_qty_or_amount


def _count_matched_columns(header_row: list[str | None]) -> int:
    count = 0
    for cell in header_row:
        if not cell:
            continue
        cell_clean = cell.strip()
        for pattern in BOQ_HEADER_PATTERNS.values():
            if pattern.search(cell_clean):
                count += 1
                break
    return count


def map_columns(header_row: list[str | None]) -> dict[str, int]:
    """Map each column index to a standard field name."""
    column_map: dict[str, int] = {}

    for col_idx, cell in enumerate(header_row):
        if not cell:
            continue
        cell_clean = cell.strip()
        if not cell_clean:
            continue

        for field_name, pattern in BOQ_HEADER_PATTERNS.items():
            if field_name in column_map:
                continue
            if pattern.search(cell_clean):
                column_map[field_name] = col_idx
                break

    return column_map


def _is_total_row(row: list[str | None]) -> bool:
    for cell in row:
        if cell and TOTAL_ROW_PATTERNS.search(cell.strip()):
            return True
    return False


def _is_empty_row(row: list[str | None]) -> bool:
    return all(not cell or not cell.strip() for cell in row)


def extract_boq_items(table: ExtractedTable) -> list[dict]:
    """Extract structured BOQ items from a single identified BOQ table."""
    if not table.rows or len(table.rows) < 2:
        return []

    header_row = table.rows[0]
    column_map = map_columns(header_row)

    if "item_name" not in column_map:
        logger.warning(
            "BOQ table on page %d has no item_name column mapped",
            table.page_number,
        )
        return []

    items: list[dict] = []
    data_rows = table.rows[1:]

    for row_idx, row in enumerate(data_rows, start=1):
        if _is_empty_row(row) or _is_total_row(row):
            continue

        item_name_idx = column_map["item_name"]
        if item_name_idx >= len(row):
            continue

        item_name = (row[item_name_idx] or "").strip()
        if not item_name:
            continue

        item: dict = {
            "page_number": table.page_number,
            "table_number": table.table_number,
            "row_number": row_idx,
            "item_name": item_name,
        }

        for field_name, col_idx in column_map.items():
            if field_name == "item_name":
                continue
            if col_idx < len(row):
                value = (row[col_idx] or "").strip()
                item[field_name] = value if value else None
            else:
                item[field_name] = None

        items.append(item)

    logger.info(
        "Extracted %d BOQ items from page %d table %d",
        len(items), table.page_number, table.table_number,
    )
    return items


def extract_all_boq_items(tables: list[ExtractedTable]) -> list[dict]:
    """Process all tables and extract BOQ items from those identified as BOQ tables."""
    all_items: list[dict] = []
    boq_table_count = 0

    for table in tables:
        if is_boq_table(table):
            boq_table_count += 1
            items = extract_boq_items(table)
            all_items.extend(items)

    logger.info(
        "Found %d BOQ tables, extracted %d total items",
        boq_table_count, len(all_items),
    )
    return all_items
