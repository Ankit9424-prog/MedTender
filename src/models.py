from typing import Literal

from pydantic import BaseModel, Field


class PageText(BaseModel):
    page_number: int
    text: str
    is_scanned: bool = False
    ocr_applied: bool = False
    char_count: int = 0

    def model_post_init(self, __context):
        if self.char_count == 0:
            self.char_count = len(self.text)


class ExtractedTable(BaseModel):
    page_number: int
    table_number: int
    rows: list[list[str | None]]
    context_before: str | None = None


class CandidateClause(BaseModel):
    candidate_id: str
    page_number: int
    page_numbers: list[int] = Field(default_factory=list)
    category: str
    subcategories: list[str] = Field(default_factory=list)
    exact_text: str
    section_heading: str | None = None
    risk_terms: list[str] = Field(default_factory=list)
    has_amount: bool = False
    has_date: bool = False
    amounts_found: list[str] = Field(default_factory=list)
    dates_found: list[str] = Field(default_factory=list)
    is_cross_page: bool = False

    def model_post_init(self, __context):
        if not self.page_numbers:
            self.page_numbers = [self.page_number]


class LLMExtractedItem(BaseModel):
    candidate_id: str
    page_number: int
    detail_type: str
    short_label: str
    exact_text: str
    important_words: list[str] = Field(default_factory=list)
    amount: str | None = None
    date_or_time: str | None = None
    why_important: str = ""
    human_review_required: bool = True


class VerificationResult(BaseModel):
    candidate_id: str
    page_number: int
    detail_type: str
    short_label: str
    exact_text: str
    important_words: list[str] = Field(default_factory=list)
    amount: str | None = None
    date_or_time: str | None = None
    why_important: str = ""
    human_review_required: bool = True
    verification_status: Literal["exact_match", "fuzzy_needs_review", "failed"] = "failed"
    verification_score: float = 0.0
    critical_terms_preserved: bool = False
    missing_critical_terms: list[str] = Field(default_factory=list)
    amounts_preserved: bool = True
    dates_preserved: bool = True
    verification_note: str = ""


class ExtractionBatchResult(BaseModel):
    batch_number: int
    total_batches: int
    items: list[LLMExtractedItem] = Field(default_factory=list)
    success: bool = True
    error_message: str | None = None


class ExtractionRunResult(BaseModel):
    all_items: list[LLMExtractedItem] = Field(default_factory=list)
    failed_batches: list[int] = Field(default_factory=list)
    total_batches: int = 0
    successful_batches: int = 0
