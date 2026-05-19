"""
Three-state verification for LLM-extracted tender items.

Verification states:
- exact_match: Extracted text is an exact substring of source. ONLY state that counts as verified.
- fuzzy_needs_review: High similarity but not exact. ALWAYS requires human review.
- failed: Low similarity or critical terms altered. Likely hallucination.

Critical design principle: In legal/tender text, a single word change
("shall" -> "should", "and" -> "or", "minimum" -> "maximum")
can completely invert the meaning. Fuzzy matching NEVER counts as verified.
"""

import re

from rapidfuzz import fuzz

from src.logging_setup import get_logger
from src.models import LLMExtractedItem, VerificationResult

logger = get_logger("verifier")

CRITICAL_LEGAL_TERMS = [
    "shall", "must", "may", "should", "will", "can",
    "and/or", "and", "or",
    "not less than", "not more than", "at least", "at most",
    "within", "before", "after", "prior to",
    "minimum", "maximum", "mandatory", "optional",
    "unless", "except", "notwithstanding", "subject to",
    "provided that", "on condition that",
]

# Amount pattern for preservation check
AMOUNT_PATTERN = re.compile(
    r"(NPR|NRs\.?|Rs\.?|रु\.?)\s?[\d,]+(\.\d+)?|[\d,]+(\.\d+)?\s?(NPR|rupees|only)",
    re.IGNORECASE,
)

# Date pattern for preservation check
DATE_PATTERN = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|20[78]\d[-/\.]\d{1,2}[-/\.]\d{1,2}",
)

FUZZY_THRESHOLD = 85


def normalize_for_comparison(text: str) -> str:
    """Collapse whitespace for substring comparison, preserve words."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_term_check(text: str) -> str:
    """Lowercase + collapse whitespace for term presence checking."""
    return re.sub(r"\s+", " ", text).strip().lower()


def find_critical_terms_in_text(text: str) -> list[str]:
    """Find which critical legal terms appear in the text."""
    lower = normalize_for_term_check(text)
    found = []

    # Check multi-word terms first (longer matches take priority)
    sorted_terms = sorted(CRITICAL_LEGAL_TERMS, key=len, reverse=True)

    for term in sorted_terms:
        # Use word boundary matching for single words, substring for multi-word
        if " " in term:
            if term in lower:
                found.append(term)
        else:
            if re.search(rf"\b{re.escape(term)}\b", lower):
                found.append(term)

    return found


def check_critical_terms_preserved(
    source_text: str,
    extracted_text: str,
) -> tuple[bool, list[str]]:
    """
    Check that critical legal terms in the source appear in the extraction.
    Returns (all_preserved, list_of_missing_terms).
    """
    source_terms = find_critical_terms_in_text(source_text)
    if not source_terms:
        return True, []

    extracted_lower = normalize_for_term_check(extracted_text)
    missing = []

    for term in source_terms:
        if " " in term:
            if term not in extracted_lower:
                missing.append(term)
        else:
            if not re.search(rf"\b{re.escape(term)}\b", extracted_lower):
                missing.append(term)

    return len(missing) == 0, missing


def check_amounts_preserved(source_text: str, extracted_text: str) -> bool:
    """Check that all amounts in source appear in extracted text."""
    source_amounts = AMOUNT_PATTERN.findall(source_text)
    if not source_amounts:
        return True

    source_amount_strings = [m[0] for m in AMOUNT_PATTERN.finditer(source_text)]
    norm_extracted = normalize_for_comparison(extracted_text)

    for amount_str in source_amount_strings:
        norm_amount = normalize_for_comparison(amount_str)
        if norm_amount not in norm_extracted:
            return False

    return True


def check_dates_preserved(source_text: str, extracted_text: str) -> bool:
    """Check that all dates in source appear in extracted text."""
    source_dates = DATE_PATTERN.findall(source_text)
    if not source_dates:
        return True

    norm_extracted = normalize_for_comparison(extracted_text)
    for date_str in source_dates:
        if date_str not in norm_extracted:
            return False

    return True


def verify_single_item(
    item: LLMExtractedItem,
    source_text: str,
) -> VerificationResult:
    """
    Verify a single extracted item against its source candidate text.
    """
    base_fields = item.model_dump(exclude={"human_review_required"})
    exact_text = item.exact_text or ""

    if not exact_text.strip():
        return VerificationResult(
            **base_fields,
            human_review_required=True,
            verification_status="failed",
            verification_score=0,
            critical_terms_preserved=False,
            verification_note="Extracted text is empty.",
        )

    norm_extracted = normalize_for_comparison(exact_text)
    norm_source = normalize_for_comparison(source_text)

    # Step 1: Exact substring check (case-insensitive for comparison)
    if norm_extracted.lower() in norm_source.lower():
        return VerificationResult(
            **base_fields,
            human_review_required=False,
            verification_status="exact_match",
            verification_score=100,
            critical_terms_preserved=True,
            verification_note="Exact text verified as substring of source.",
        )

    # Step 2: Not exact — compute fuzzy score and run checks
    fuzzy_score = fuzz.partial_ratio(
        norm_extracted.lower(),
        norm_source.lower(),
    )

    terms_preserved, missing_terms = check_critical_terms_preserved(
        source_text, exact_text
    )
    amounts_ok = check_amounts_preserved(source_text, exact_text)
    dates_ok = check_dates_preserved(source_text, exact_text)

    # Step 3: Determine status
    if fuzzy_score >= FUZZY_THRESHOLD and terms_preserved and amounts_ok and dates_ok:
        return VerificationResult(
            **base_fields,
            human_review_required=True,
            verification_status="fuzzy_needs_review",
            verification_score=fuzzy_score,
            critical_terms_preserved=True,
            amounts_preserved=amounts_ok,
            dates_preserved=dates_ok,
            verification_note=(
                "Text is similar but not an exact quote. "
                "HUMAN REVIEW REQUIRED — verify against original document."
            ),
        )
    else:
        notes = []
        if fuzzy_score < FUZZY_THRESHOLD:
            notes.append(f"Low similarity ({fuzzy_score}%).")
        if not terms_preserved:
            notes.append(f"Critical terms changed or missing: {missing_terms}.")
        if not amounts_ok:
            notes.append("Amount values differ from source.")
        if not dates_ok:
            notes.append("Date values differ from source.")

        return VerificationResult(
            **base_fields,
            human_review_required=True,
            verification_status="failed",
            verification_score=fuzzy_score,
            critical_terms_preserved=terms_preserved,
            missing_critical_terms=missing_terms,
            amounts_preserved=amounts_ok,
            dates_preserved=dates_ok,
            verification_note=" ".join(notes),
        )


def verify_extractions(
    extracted_items: list[dict],
    candidates: list[dict],
) -> list[dict]:
    """
    Verify all extracted items against their source candidates.
    Returns list of VerificationResult dicts.
    """
    candidate_map = {c["candidate_id"]: c for c in candidates}
    results: list[dict] = []

    for item_dict in extracted_items:
        candidate_id = item_dict.get("candidate_id")

        candidate = candidate_map.get(candidate_id)
        if not candidate:
            result = VerificationResult(
                candidate_id=candidate_id or "UNKNOWN",
                page_number=item_dict.get("page_number", 0),
                detail_type=item_dict.get("detail_type", "unknown"),
                short_label=item_dict.get("short_label", ""),
                exact_text=item_dict.get("exact_text", ""),
                important_words=item_dict.get("important_words", []),
                amount=item_dict.get("amount"),
                date_or_time=item_dict.get("date_or_time"),
                why_important=item_dict.get("why_important", ""),
                human_review_required=True,
                verification_status="failed",
                verification_score=0,
                critical_terms_preserved=False,
                verification_note="Candidate ID not found in source data.",
            )
            results.append(result.model_dump())
            continue

        try:
            item = LLMExtractedItem(**item_dict)
        except Exception:
            result = VerificationResult(
                candidate_id=candidate_id or "UNKNOWN",
                page_number=item_dict.get("page_number", 0),
                detail_type=item_dict.get("detail_type", "unknown"),
                short_label=item_dict.get("short_label", ""),
                exact_text=item_dict.get("exact_text", ""),
                important_words=item_dict.get("important_words", []),
                amount=item_dict.get("amount"),
                date_or_time=item_dict.get("date_or_time"),
                why_important=item_dict.get("why_important", ""),
                human_review_required=True,
                verification_status="failed",
                verification_score=0,
                verification_note="Invalid item structure from LLM.",
            )
            results.append(result.model_dump())
            continue

        source_text = candidate["exact_text"]
        verified = verify_single_item(item, source_text)
        results.append(verified.model_dump())

    # Log summary
    statuses = [r.get("verification_status") for r in results]
    exact_count = statuses.count("exact_match")
    fuzzy_count = statuses.count("fuzzy_needs_review")
    failed_count = statuses.count("failed")

    logger.info(
        "Verification complete: %d exact, %d fuzzy (needs review), %d failed. Total: %d",
        exact_count, fuzzy_count, failed_count, len(results),
    )

    return results
