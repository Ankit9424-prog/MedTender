"""
Structure-aware clause splitting for tender documents.

Design principle: KEEP THINGS TOGETHER. A requirement with sub-items
is ONE clause, not five fragments. Only split at major section boundaries.
Fragmented output is worse than slightly long clauses.

Handles:
- Major section headings as split boundaries
- Numbered sub-items kept with their parent text
- Short fragments merged into adjacent clauses
- Cross-page clause continuity
- Table content as clause text
"""

import re
from dataclasses import dataclass

from src.logging_setup import get_logger

logger = get_logger("clause_splitter")

# Major section headings — these are the ONLY places we split
MAJOR_SECTION_PATTERN = re.compile(
    r"^("
    r"\d+\.\d+\.\d+\.?\s+"  # 1.1.1
    r"|\d+\.\d+\.?\s+"  # 1.1
    r"|\d+\.?\s+[A-Z]"  # 1. Capital letter start (numbered heading)
    r"|Section\s+\d+"
    r"|SECTION\s+\d+"
    r"|Article\s+\d+"
    r"|Clause\s+\d+"
    r"|Schedule\s+\d+"
    r"|Annex\s+[A-Z\d]+"
    r"|भाग\s+\d+"
    r"|खण्ड\s+\d+"
    r"|दफा\s+\d+"
    r"|अनुसूची\s+\d+"
    r")",
    re.MULTILINE,
)

# Headings that are section titles (for labeling, not splitting)
SECTION_HEADING_PATTERN = re.compile(
    r"^("
    r"\d+\.\d+\.\d+\.?\s+"
    r"|\d+\.\d+\.?\s+"
    r"|\d+\.?\s+"
    r"|[A-Z]\.\s+"
    r"|\([a-z]\)\s+"
    r"|\([ivxlcdm]+\)\s+"
    r"|[a-z]\)\s+"
    r"|Section\s+\d+"
    r"|SECTION\s+\d+"
    r"|Article\s+\d+"
    r"|Clause\s+\d+"
    r"|Schedule\s+\d+"
    r"|Annex\s+[A-Z\d]+"
    r"|भाग\s+\d+"
    r"|खण्ड\s+\d+"
    r"|दफा\s+\d+"
    r"|अनुसूची\s+\d+"
    r")",
    re.MULTILINE | re.IGNORECASE,
)

ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z\s\-:]{5,}$", re.MULTILINE)

INCOMPLETE_ENDINGS = re.compile(
    r"(,\s*$|;\s*$|\band\s*$|\bor\s*$|\bthe\s*$|\bof\s*$|\bfor\s*$|\bto\s*$|\bin\s*$)"
)

MIN_CLAUSE_LENGTH = 15
MERGE_THRESHOLD = 80


@dataclass
class SplitClause:
    text: str
    section_heading: str | None = None
    is_cross_page: bool = False


def detect_section_heading(text: str) -> str | None:
    """Extract section heading from the beginning of text, if present."""
    lines = text.strip().split("\n")
    if not lines:
        return None

    first_line = lines[0].strip()

    if SECTION_HEADING_PATTERN.match(first_line):
        return first_line

    if ALL_CAPS_HEADING.match(first_line) and len(first_line) < 100:
        return first_line

    return None


def detect_incomplete_boundary(page_text: str) -> str | None:
    """
    Check if a page ends mid-sentence, suggesting the clause continues on the next page.
    Returns the trailing fragment that should be prepended to the next page's first clause.
    """
    if not page_text.strip():
        return None

    lines = page_text.strip().split("\n")
    last_line = lines[-1].strip()

    if not last_line:
        return None

    if INCOMPLETE_ENDINGS.search(last_line):
        paragraphs = re.split(r"\n\s*\n", page_text)
        last_para = paragraphs[-1].strip() if paragraphs else ""
        if last_para and len(last_para) >= MIN_CLAUSE_LENGTH:
            return last_para

    if last_line and last_line[-1] not in ".;:!?\"')":
        if not SECTION_HEADING_PATTERN.match(last_line):
            paragraphs = re.split(r"\n\s*\n", page_text)
            last_para = paragraphs[-1].strip() if paragraphs else ""
            if last_para and len(last_para) >= MIN_CLAUSE_LENGTH:
                return last_para

    return None


def _is_major_section_start(text: str) -> bool:
    """Check if text starts with a major section heading."""
    first_line = text.strip().split("\n")[0].strip() if text.strip() else ""
    if MAJOR_SECTION_PATTERN.match(first_line):
        return True
    if ALL_CAPS_HEADING.match(first_line) and len(first_line) < 100:
        return True
    return False


def split_into_clauses(
    page_text: str,
    previous_page_trailing: str | None = None,
) -> list[SplitClause]:
    """
    Split page text into clauses conservatively — prefer keeping content together.

    Strategy:
    1. Split at double-newlines (PDF paragraph boundaries)
    2. Merge consecutive short fragments into one clause
    3. Only start a new clause at major section headings
    4. Keep numbered sub-items with their parent paragraph
    """
    if not page_text.strip():
        return []

    paragraphs = re.split(r"\n\s*\n", page_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    # Group paragraphs into sections — only split at major headings
    sections: list[list[str]] = []
    current_section: list[str] = []

    for para in paragraphs:
        if _is_major_section_start(para) and current_section:
            sections.append(current_section)
            current_section = [para]
        else:
            current_section.append(para)

    if current_section:
        sections.append(current_section)

    # Build clauses — each section becomes one clause (preserving all sub-items together)
    clauses: list[SplitClause] = []

    for section_paras in sections:
        full_text = "\n\n".join(section_paras)

        if len(full_text) < MIN_CLAUSE_LENGTH:
            continue

        heading = detect_section_heading(full_text)

        # If section is very long (>2000 chars), try splitting at sub-sections
        # but keep each sub-section whole
        if len(full_text) > 2000 and len(section_paras) > 1:
            sub_clauses = _split_long_section(section_paras, heading)
            clauses.extend(sub_clauses)
        else:
            clauses.append(SplitClause(
                text=full_text,
                section_heading=heading,
            ))

    # Merge any remaining short clauses into adjacent ones
    clauses = _merge_short_clauses(clauses)

    # Handle cross-page continuation
    if previous_page_trailing and clauses:
        first_clause = clauses[0]
        merged_text = previous_page_trailing + "\n" + first_clause.text
        clauses[0] = SplitClause(
            text=merged_text,
            section_heading=first_clause.section_heading,
            is_cross_page=True,
        )
        logger.info("Merged cross-page clause: %d chars from previous page", len(previous_page_trailing))
    elif previous_page_trailing and not clauses:
        if len(previous_page_trailing) >= MIN_CLAUSE_LENGTH:
            clauses.append(SplitClause(
                text=previous_page_trailing,
                section_heading=None,
                is_cross_page=True,
            ))

    return clauses


def _split_long_section(paragraphs: list[str], parent_heading: str | None) -> list[SplitClause]:
    """
    For very long sections, split into sub-groups but keep related paragraphs together.
    A new sub-group starts only when a paragraph looks like a sub-heading.
    """
    SUB_HEADING = re.compile(
        r"^(\d+\.\d+\.?\s+|\([a-z]\)\s+|[a-z]\)\s+|\([ivxlcdm]+\)\s+)",
        re.IGNORECASE,
    )

    groups: list[list[str]] = []
    current: list[str] = []

    for para in paragraphs:
        if SUB_HEADING.match(para.strip()) and current and len("\n\n".join(current)) > 100:
            groups.append(current)
            current = [para]
        else:
            current.append(para)

    if current:
        groups.append(current)

    results: list[SplitClause] = []
    for group in groups:
        text = "\n\n".join(group)
        if len(text) >= MIN_CLAUSE_LENGTH:
            heading = detect_section_heading(text) or parent_heading
            results.append(SplitClause(text=text, section_heading=heading))

    return results


def _merge_short_clauses(clauses: list[SplitClause]) -> list[SplitClause]:
    """Merge clauses shorter than MERGE_THRESHOLD into the previous clause."""
    if not clauses:
        return clauses

    merged: list[SplitClause] = [clauses[0]]

    for clause in clauses[1:]:
        if len(clause.text) < MERGE_THRESHOLD and merged:
            prev = merged[-1]
            merged[-1] = SplitClause(
                text=prev.text + "\n\n" + clause.text,
                section_heading=prev.section_heading,
                is_cross_page=prev.is_cross_page,
            )
        else:
            merged.append(clause)

    return merged


def table_to_clause_text(rows: list[list[str | None]], context_before: str | None = None) -> str:
    """
    Convert a table's rows into readable text suitable for keyword matching.
    Preserves all cell content including specifications, quantities, etc.
    """
    parts: list[str] = []

    if context_before:
        parts.append(context_before.strip())

    header = rows[0] if rows else []
    header_text = [str(cell or "").strip() for cell in header]

    for row in rows[1:]:
        row_parts: list[str] = []
        for col_idx, cell in enumerate(row):
            cell_text = str(cell or "").strip()
            if not cell_text:
                continue
            col_name = header_text[col_idx] if col_idx < len(header_text) else ""
            if col_name:
                row_parts.append(f"{col_name}: {cell_text}")
            else:
                row_parts.append(cell_text)
        if row_parts:
            parts.append(" | ".join(row_parts))

    return "\n".join(parts)
