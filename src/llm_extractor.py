"""
LLM extraction using AWS Bedrock converse API with tool_use for structured output.

Key design decisions:
- tool_use forces structured JSON output (no free-text parsing)
- tenacity retries handle transient API failures
- Dynamic batch sizing prevents token limit issues
- Partial results are preserved on batch failure
- Enhanced prompt preserves exact wording, Nepali text, and BS dates
"""

import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_bedrock_client, get_settings
from src.logging_setup import get_logger
from src.models import ExtractionRunResult, LLMExtractedItem

logger = get_logger("llm_extractor")

SYSTEM_PROMPT = """You are a tender requirement extraction assistant for Nepal government medical procurement tenders.

CRITICAL RULES — violations will cause extraction to be rejected:
1. exact_text MUST be copied character-for-character from the provided candidate text. Do NOT paraphrase, summarize, correct typos, or rephrase.
2. Words like "and", "or", "and/or", "shall", "must", "may", "unless", "except", "within", "minimum", "maximum" are legally binding. Preserve them EXACTLY.
3. If text contains Nepali (Devanagari) script, preserve it exactly as-is.
4. Dates in format 20XX/XX/XX may be Bikram Sambat calendar — do NOT convert to AD. Preserve as-is.
5. For medical terms (CE marking, ISO 13485, GMP, WHO prequalified), preserve exact certification references.
6. Do NOT correct apparent typos in exact_text — they may be intentional amendments.
7. If a clause is ambiguous or you are uncertain about any field, set human_review_required to true.
8. Do NOT invent information not present in the candidate text.
9. If one candidate contains multiple distinct requirements, create one item per requirement, each with the same candidate_id.
10. amount and date_or_time must be copied exactly from the text, or set to null if not explicitly stated.

WHAT NOT TO DO:
- Do NOT write: "The bidder should submit..." when source says "The bidder shall submit..."
- Do NOT write: "minimum 3 years" when source says "not less than 3 years"
- Do NOT compute dates (e.g., "within 30 days" does NOT mean you should calculate a date)
- Do NOT merge text from different candidates into one exact_text"""

TOOL_SCHEMA = {
    "tools": [
        {
            "toolSpec": {
                "name": "record_tender_items",
                "description": "Record extracted tender requirement items from the provided candidate clauses. Each item preserves exact wording from the source.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "description": "List of extracted tender requirement items",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "candidate_id": {
                                            "type": "string",
                                            "description": "The candidate ID (e.g., C0001) this item was extracted from",
                                        },
                                        "page_number": {
                                            "type": "integer",
                                            "description": "Page number where this requirement appears",
                                        },
                                        "detail_type": {
                                            "type": "string",
                                            "enum": [
                                                "deadline",
                                                "bid_security",
                                                "eligibility",
                                                "required_document",
                                                "technical_requirement",
                                                "medical_specification",
                                                "warranty",
                                                "delivery",
                                                "financial",
                                                "disqualification",
                                                "other",
                                            ],
                                            "description": "Category of this tender requirement",
                                        },
                                        "short_label": {
                                            "type": "string",
                                            "description": "Brief label (3-8 words) describing this requirement",
                                        },
                                        "exact_text": {
                                            "type": "string",
                                            "description": "EXACT text copied from the candidate — do not paraphrase",
                                        },
                                        "important_words": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "Legally important words in this text (shall, must, may, and/or, etc.)",
                                        },
                                        "amount": {
                                            "type": ["string", "null"],
                                            "description": "Exact amount if present in text, null otherwise",
                                        },
                                        "date_or_time": {
                                            "type": ["string", "null"],
                                            "description": "Exact date/time if present in text, null otherwise",
                                        },
                                        "why_important": {
                                            "type": "string",
                                            "description": "Brief explanation of why this matters for bid preparation",
                                        },
                                        "human_review_required": {
                                            "type": "boolean",
                                            "description": "Set to true if uncertain about extraction accuracy",
                                        },
                                    },
                                    "required": [
                                        "candidate_id",
                                        "page_number",
                                        "detail_type",
                                        "short_label",
                                        "exact_text",
                                        "important_words",
                                        "amount",
                                        "date_or_time",
                                        "why_important",
                                        "human_review_required",
                                    ],
                                },
                            },
                        },
                        "required": ["items"],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": "record_tender_items"}},
}


def estimate_tokens(candidates: list[dict]) -> int:
    """Rough token estimate for a batch of candidates."""
    total_chars = sum(len(c.get("exact_text", "")) for c in candidates)
    total_chars += len(SYSTEM_PROMPT)
    total_chars += 200 * len(candidates)  # overhead per candidate
    return int(total_chars / 3)


def build_user_prompt(candidates: list[dict[str, Any]]) -> str:
    parts = [
        "Extract important tender details from these candidate clauses.",
        "Remember: exact_text must be copied EXACTLY from the candidate text. Do not paraphrase.",
        "",
    ]

    for c in candidates:
        parts.append(f"--- Candidate {c['candidate_id']} (Page {c['page_number']}) ---")
        parts.append(f"Category: {c['category']}")
        if c.get("section_heading"):
            parts.append(f"Section: {c['section_heading']}")
        parts.append(f"Text:\n{c['exact_text']}")
        parts.append("")

    return "\n".join(parts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.warning(
        "Bedrock API retry attempt %d after error: %s",
        retry_state.attempt_number,
        retry_state.outcome.exception() if retry_state.outcome else "unknown",
    ),
)
def _call_bedrock(
    client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    tool_config: dict,
) -> list[dict]:
    """Call Bedrock converse API with tool_use. Returns parsed items list."""
    response = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [{"text": user_prompt}],
            }
        ],
        toolConfig=tool_config,
        inferenceConfig={"temperature": 0},
    )

    # Extract tool_use response
    output_message = response["output"]["message"]
    content_blocks = output_message.get("content", [])

    for block in content_blocks:
        if "toolUse" in block:
            tool_input = block["toolUse"]["input"]
            return tool_input.get("items", [])

    # Fallback: if no tool_use block, try text content
    for block in content_blocks:
        if "text" in block:
            logger.warning("Model returned text instead of tool_use. Attempting JSON parse.")
            import json
            text = block["text"].strip()
            if text.startswith("```"):
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            return parsed.get("items", [])

    logger.error("No valid response from Bedrock API")
    return []


def extract_details_with_llm(
    candidates: list[dict[str, Any]],
    batch_size: int | None = None,
    progress_callback=None,
) -> ExtractionRunResult:
    """
    Extract structured tender details using LLM with tool_use.

    Args:
        candidates: List of candidate clause dicts
        batch_size: Override default batch size
        progress_callback: Optional callable(batch_num, total_batches) for UI progress

    Returns:
        ExtractionRunResult with all items and failure metadata
    """
    settings = get_settings()
    client = get_bedrock_client()
    model_id = settings.BEDROCK_MODEL_ID

    if batch_size is None:
        batch_size = settings.LLM_BATCH_SIZE

    # Dynamic batch sizing based on token estimates
    effective_batch_size = batch_size
    if candidates:
        avg_chars = sum(len(c.get("exact_text", "")) for c in candidates) / len(candidates)
        if avg_chars > 500:
            effective_batch_size = max(5, batch_size // 2)
            logger.info(
                "Large candidates (avg %d chars), reducing batch size to %d",
                int(avg_chars), effective_batch_size,
            )

    total_batches = (len(candidates) + effective_batch_size - 1) // effective_batch_size
    result = ExtractionRunResult(total_batches=total_batches)

    for i in range(0, len(candidates), effective_batch_size):
        batch_num = (i // effective_batch_size) + 1
        batch = candidates[i: i + effective_batch_size]

        if progress_callback:
            progress_callback(batch_num, total_batches)

        estimated_tokens = estimate_tokens(batch)
        logger.info(
            "Processing batch %d/%d: %d candidates, ~%d tokens",
            batch_num, total_batches, len(batch), estimated_tokens,
        )

        start_time = time.time()

        try:
            user_prompt = build_user_prompt(batch)
            items = _call_bedrock(
                client, model_id, SYSTEM_PROMPT, user_prompt, TOOL_SCHEMA,
            )

            elapsed = time.time() - start_time
            logger.info(
                "Batch %d/%d complete: %d items extracted in %.1fs",
                batch_num, total_batches, len(items), elapsed,
            )

            for item_dict in items:
                try:
                    item = LLMExtractedItem(**item_dict)
                    result.all_items.append(item)
                except Exception as e:
                    logger.warning("Invalid item from LLM, skipping: %s", str(e))
                    result.all_items.append(LLMExtractedItem(
                        candidate_id=item_dict.get("candidate_id", "UNKNOWN"),
                        page_number=item_dict.get("page_number", 0),
                        detail_type="other",
                        short_label="Parse error",
                        exact_text=str(item_dict.get("exact_text", "")),
                        why_important="Item had invalid structure from LLM",
                        human_review_required=True,
                    ))

            result.successful_batches += 1

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                "Batch %d/%d failed after %.1fs: %s",
                batch_num, total_batches, elapsed, str(e),
            )
            result.failed_batches.append(batch_num)

    logger.info(
        "Extraction complete: %d items from %d/%d batches. %d batches failed.",
        len(result.all_items), result.successful_batches,
        result.total_batches, len(result.failed_batches),
    )

    return result
