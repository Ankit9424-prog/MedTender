import re
from typing import List

from src.clause_splitter import MIN_CLAUSE_LENGTH, split_into_clauses, table_to_clause_text
from src.logging_setup import get_logger
from src.models import CandidateClause, ExtractedTable, PageText
from src.pdf_parser import get_cross_page_boundaries

logger = get_logger("rule_extractor")


IMPORTANT_CONNECTORS = [
    "and/or",
    " and ",
    " or ",
    "unless",
    "except",
    "provided that",
    "notwithstanding",
    "subject to",
    "at least",
    "not less than",
    "not more than",
    "within",
    "before",
    "after",
    "prior to",
    "on condition that",
]

CATEGORY_KEYWORDS = {
    "deadline": [
        "deadline", "submission", "submit", "bid opening", "closing date",
        "date and time", "electronic bid", "e-gp", "last date",
        "opening date", "validity period", "bid validity",
        # Nepali
        "अन्तिम मिति", "बोलपत्र खोल्ने", "पेश गर्ने", "समय सीमा",
        "दर्ता गर्ने", "बुझाउने",
    ],
    "bid_security": [
        "bid security", "bank guarantee", "security amount", "earnest money",
        "bid bond", "performance security", "retention money",
        # Nepali
        "बोलपत्र जमानत", "बैंक ग्यारेन्टी", "धरौटी", "अर्नेस्ट मनी",
        "कार्य सम्पादन जमानत",
    ],
    "eligibility": [
        "eligible", "eligibility", "qualification", "experience",
        "turnover", "manufacturer authorization", "authorized distributor",
        "joint venture", "jv", "pre-qualification", "blacklisted",
        "conflict of interest",
        # Nepali
        "योग्यता", "अनुभव", "कारोबार", "इजाजतपत्र",
        "अधिकृत वितरक", "कालोसूचीमा",
    ],
    "documents_required": [
        "registration", "vat", "pan", "tax clearance", "audit report",
        "power of attorney", "declaration", "certificate", "license",
        "affidavit", "notarized", "attested copy",
        # Nepali
        "कागजात", "प्रमाणपत्र", "दर्ता", "करचुक्ता",
        "लेखापरीक्षण", "प्रतिलिपि",
    ],
    "technical_specification": [
        "specification", "technical", "compliance", "catalogue", "brochure",
        "ce", "iso", "fda", "standard", "model", "brand",
        "technical evaluation", "technical score",
        # Nepali
        "प्राविधिक विशिष्टता", "मापदण्ड", "गुणस्तर",
    ],
    "medical_specification": [
        "ce marking", "iso 13485", "biocompatibility", "sterilization",
        "shelf life", "expiry", "batch number", "lot number",
        "cold chain", "storage temperature", "dosage", "formulation",
        "generic name", "pharmacopoeia", "gmp", "who prequalified",
        "dda", "department of drug administration", "medical device",
        "class i", "class ii", "class iii", "usfda", "therapeutic",
        "surgical", "diagnostic", "reagent", "laboratory",
        # Nepali
        "औषधि", "चिकित्सा", "स्वास्थ्य सामग्री", "औषधि प्रशासन",
    ],
    "warranty_delivery": [
        "warranty", "guarantee", "delivery", "installation",
        "commissioning", "training", "after sales", "maintenance",
        "spare parts", "service center", "response time",
        "delivery schedule", "lead time",
        # Nepali
        "मर्मत", "वारेन्टी", "डेलिभरी", "जडान", "तालिम",
    ],
    "financial": [
        "price schedule", "boq", "bill of quantities", "currency",
        "tax", "customs", "vat", "payment", "advance payment",
        "price adjustment", "price escalation", "lc",
        "letter of credit",
        # Nepali
        "मूल्य", "भन्सार", "कर", "भुक्तानी",
    ],
    "disqualification_risk": [
        "rejected", "non-responsive", "disqualified",
        "shall be rejected", "failure to", "not acceptable",
        "invalid", "substantially responsive", "will not be considered",
        "ground for rejection",
        # Nepali
        "अयोग्य", "अस्वीकृत", "रद्द", "अमान्य",
    ],
}

OBLIGATION_WORDS = [
    "shall", "must", "required", "mandatory", "have to",
    "should", "may", "will", "is required", "are required",
    "obligatory", "compulsory", "necessary",
]

# Amounts: NPR, NRs, Rs, रु, Nepali numerals, Indian comma notation
AMOUNT_PATTERN = re.compile(
    r"("
    r"(NPR|NRs\.?|Rs\.?|रु\.?|रूपैयाँ?)\s?[\d,]+(\.\d+)?"
    r"|[\d,]+(\.\d+)?\s?(NPR|rupees|only|/-)"
    r"|[१२३४५६७८९०][१२३४५६७८९०,]+(\.[१२३४५६७८९०]+)?"
    r"|\d+[,.]?\d*\s*(lakh|crore|lakhs|crores)"
    r"|\d{1,3}(,\d{2})*(,\d{3})(\.\d+)?"
    r")",
    re.IGNORECASE,
)

# AD dates
AD_DATE_PATTERN = re.compile(
    r"("
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r")",
    re.IGNORECASE,
)

# Bikram Sambat dates
BS_DATE_PATTERN = re.compile(
    r"("
    r"20[78]\d[-/\.]\d{1,2}[-/\.]\d{1,2}"
    r"|२०[७८][०-९][-/\.][०-९]{1,2}[-/\.][०-९]{1,2}"
    r"|(?:बैशाख|जेठ|असार|साउन|भदौ|असोज|कार्तिक|मंसिर|पुष|माघ|फागुन|चैत)\s*[०-९\d]{1,2}"
    r"|(?:Baisakh|Jestha|Ashadh|Shrawan|Bhadra|Ashwin|Kartik|Mangsir|Poush|Magh|Falgun|Chaitra)\s+\d{1,2}"
    r")"
)

# Time patterns (important for deadlines)
TIME_PATTERN = re.compile(
    r"\d{1,2}:\d{2}\s*(AM|PM|am|pm|hrs|hours|बजे)?",
)


def find_categories(text: str) -> tuple[str | None, list[str]]:
    """
    Find all matching categories for a clause.
    Returns (primary_category, all_subcategories).
    """
    lower = text.lower()
    matched: list[str] = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower or kw in text:
                matched.append(category)
                break

    if not matched:
        return None, []

    priority = [
        "disqualification_risk",
        "deadline",
        "bid_security",
        "eligibility",
        "documents_required",
        "medical_specification",
        "technical_specification",
        "warranty_delivery",
        "financial",
    ]

    primary = None
    for p in priority:
        if p in matched:
            primary = p
            break

    if primary is None:
        primary = matched[0]

    subcategories = [c for c in matched if c != primary]
    return primary, subcategories


def find_risk_terms(text: str) -> list[str]:
    lower = f" {text.lower()} "
    found = []

    for term in IMPORTANT_CONNECTORS + OBLIGATION_WORDS:
        if term.lower() in lower:
            found.append(term)

    return found


def find_amounts(text: str) -> list[str]:
    return [m.group(0) for m in AMOUNT_PATTERN.finditer(text)]


def find_dates(text: str) -> list[str]:
    dates = []
    dates.extend(m.group(0) for m in AD_DATE_PATTERN.finditer(text))
    dates.extend(m.group(0) for m in BS_DATE_PATTERN.finditer(text))
    dates.extend(m.group(0) for m in TIME_PATTERN.finditer(text))
    return dates


def extract_candidate_clauses(
    pages: list[PageText],
    tables: list[ExtractedTable] | None = None,
) -> list[CandidateClause]:
    """
    Rule-based extractor with cross-page awareness.
    Intentionally broad: missing a tender detail is worse than showing extra candidates.
    Also extracts from table content so specs/requirements in tables aren't lost.
    """
    boundaries = get_cross_page_boundaries(pages)
    candidates: list[CandidateClause] = []
    counter = 1

    for page in pages:
        previous_trailing = boundaries.get(page.page_number - 1)

        clauses = split_into_clauses(page.text, previous_trailing)

        for clause in clauses:
            text = clause.text
            primary_category, subcategories = find_categories(text)
            risk_terms = find_risk_terms(text)
            amounts = find_amounts(text)
            dates = find_dates(text)

            has_category = primary_category is not None
            has_risk = len(risk_terms) > 0
            has_amount = len(amounts) > 0
            has_date = len(dates) > 0

            if has_category or has_risk or has_amount or has_date:
                page_numbers = [page.page_number]
                if clause.is_cross_page:
                    prev_page = page.page_number - 1
                    if prev_page >= 1:
                        page_numbers = [prev_page, page.page_number]

                candidates.append(CandidateClause(
                    candidate_id=f"C{counter:04d}",
                    page_number=page.page_number,
                    page_numbers=page_numbers,
                    category=primary_category or "general_important_clause",
                    subcategories=subcategories,
                    exact_text=text,
                    section_heading=clause.section_heading,
                    risk_terms=risk_terms,
                    has_amount=has_amount,
                    has_date=has_date,
                    amounts_found=amounts,
                    dates_found=dates,
                    is_cross_page=clause.is_cross_page,
                ))
                counter += 1

    # Extract from tables — captures specs, requirements, and items embedded in tables
    if tables:
        for table in tables:
            if len(table.rows) < 2:
                continue

            table_text = table_to_clause_text(table.rows, table.context_before)
            if not table_text or len(table_text) < MIN_CLAUSE_LENGTH:
                continue

            primary_category, subcategories = find_categories(table_text)
            risk_terms = find_risk_terms(table_text)
            amounts = find_amounts(table_text)
            dates = find_dates(table_text)

            has_category = primary_category is not None
            has_risk = len(risk_terms) > 0
            has_amount = len(amounts) > 0
            has_date = len(dates) > 0

            if has_category or has_risk or has_amount or has_date:
                candidates.append(CandidateClause(
                    candidate_id=f"C{counter:04d}",
                    page_number=table.page_number,
                    page_numbers=[table.page_number],
                    category=primary_category or "general_important_clause",
                    subcategories=subcategories,
                    exact_text=table_text,
                    section_heading=table.context_before[:100] if table.context_before else None,
                    risk_terms=risk_terms,
                    has_amount=has_amount,
                    has_date=has_date,
                    amounts_found=amounts,
                    dates_found=dates,
                    is_cross_page=False,
                ))
                counter += 1

    logger.info(
        "Extracted %d candidate clauses from %d pages",
        len(candidates), len(pages),
    )
    return candidates


def candidates_to_dicts(candidates: List[CandidateClause]) -> list[dict]:
    return [c.model_dump() for c in candidates]
