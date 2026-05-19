import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.audit import log_extraction_run
from src.boq_extractor import extract_all_boq_items
from src.config import get_settings, get_temp_dir
from src.llm_extractor import extract_details_with_llm
from src.pdf_parser import extract_tables, extract_text_by_page
from src.rule_extractor import candidates_to_dicts, extract_candidate_clauses
from src.verifier import verify_extractions

st.set_page_config(
    page_title="MedTender Extractor",
    layout="wide",
)

st.title("MedTender Extractor")
st.caption("Upload a tender PDF and extract exact important details with page numbers.")

st.warning(
    "**IMPORTANT**: This is an AI-assisted extraction tool. "
    "All extracted items MUST be verified against the original document "
    "by a qualified professional before use in bid preparation. "
    "AI extraction may contain errors — never rely solely on these results."
)

settings = get_settings()

uploaded_file = st.file_uploader("Upload tender PDF", type=["pdf"])

use_llm = st.checkbox(
    "Use AI structured extraction",
    value=False,
    help="If unchecked, only rule-based exact clause extraction is used.",
)

if uploaded_file:
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > settings.MAX_PDF_SIZE_MB:
        st.error(
            f"File is {file_size_mb:.1f} MB — exceeds the {settings.MAX_PDF_SIZE_MB} MB limit. "
            "Please upload a smaller file."
        )
        st.stop()

    temp_dir = get_temp_dir()
    pdf_path = temp_dir / f"upload_{uploaded_file.name}"

    try:
        pdf_path.write_bytes(uploaded_file.getvalue())

        st.success(f"Uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")

        # --- Text Extraction ---
        with st.spinner("Extracting text page by page..."):
            pages = extract_text_by_page(pdf_path)
            tables = extract_tables(pdf_path)

        total_pages = len(pages)
        scanned_pages = [p.page_number for p in pages if p.is_scanned]
        ocr_pages = [p.page_number for p in pages if p.ocr_applied]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total pages", total_pages)
        col2.metric("Scanned pages", len(scanned_pages))
        col3.metric("OCR applied", len(ocr_pages))
        col4.metric("Tables found", len(tables))

        if scanned_pages:
            if ocr_pages:
                st.info(
                    f"Pages {ocr_pages} were scanned — OCR was applied. "
                    "OCR text may be less accurate than digital text."
                )
            else:
                st.warning(
                    f"Pages {scanned_pages} appear scanned but OCR is unavailable. "
                    "Install Tesseract for automatic OCR. "
                    "These pages may have missing text."
                )

        # --- Page Preview ---
        with st.expander("Preview extracted page text"):
            selected_page = st.selectbox(
                "Select page",
                options=[p.page_number for p in pages],
            )
            page_obj = next(p for p in pages if p.page_number == selected_page)

            if page_obj.is_scanned:
                st.caption("(Scanned page" + (" — OCR applied)" if page_obj.ocr_applied else " — no OCR)"))

            st.text_area(
                f"Page {selected_page} text ({page_obj.char_count} chars)",
                page_obj.text,
                height=400,
            )

        # --- Rule-based Extraction ---
        with st.spinner("Finding important clauses..."):
            candidates = extract_candidate_clauses(pages, tables)
            candidate_dicts = candidates_to_dicts(candidates)

        st.subheader("Rule-based Important Clauses")
        st.write(
            "These are exact paragraphs/clauses extracted from the PDF. "
            "Extra clauses are okay — missing clauses are dangerous."
        )

        if candidate_dicts:
            candidate_df = pd.DataFrame(candidate_dicts)

            display_cols = [
                "candidate_id", "page_number", "category",
                "subcategories", "exact_text", "risk_terms",
                "has_amount", "has_date", "is_cross_page",
            ]
            available_cols = [c for c in display_cols if c in candidate_df.columns]

            st.dataframe(
                candidate_df[available_cols],
                use_container_width=True,
                height=450,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "Download clauses as JSON",
                    data=json.dumps(candidate_dicts, indent=2, ensure_ascii=False),
                    file_name="rule_based_clauses.json",
                    mime="application/json",
                )
            with col_b:
                st.download_button(
                    "Download clauses as CSV",
                    data=candidate_df.to_csv(index=False),
                    file_name="rule_based_clauses.csv",
                    mime="text/csv",
                )
        else:
            st.error(
                "No important clauses found. The PDF may be fully scanned "
                "without OCR, or the document format is unsupported."
            )

        # --- LLM Extraction ---
        if use_llm and candidate_dicts:
            st.subheader("AI Structured Extraction")

            if not settings.validate_aws_credentials():
                st.error(
                    "AWS credentials not configured. "
                    "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file."
                )
            else:
                max_candidates = st.slider(
                    "How many candidate clauses should AI analyze?",
                    min_value=5,
                    max_value=min(100, len(candidate_dicts)),
                    value=min(40, len(candidate_dicts)),
                    step=5,
                )

                if st.button("Run AI extraction"):
                    selected_candidates = candidate_dicts[:max_candidates]
                    progress_bar = st.progress(0, text="Starting AI extraction...")

                    def update_progress(batch_num, total_batches):
                        progress = batch_num / total_batches
                        progress_bar.progress(
                            progress,
                            text=f"Processing batch {batch_num}/{total_batches}...",
                        )

                    with st.spinner("Running AI structured extraction..."):
                        extraction_result = extract_details_with_llm(
                            selected_candidates,
                            progress_callback=update_progress,
                        )

                    progress_bar.progress(1.0, text="Extraction complete.")

                    if extraction_result.failed_batches:
                        st.warning(
                            f"Some batches failed: {extraction_result.failed_batches}. "
                            f"Results from {extraction_result.successful_batches}/{extraction_result.total_batches} batches shown."
                        )

                    extracted_items = [item.model_dump() for item in extraction_result.all_items]

                    if extracted_items:
                        with st.spinner("Verifying exact text preservation..."):
                            verified_items = verify_extractions(extracted_items, selected_candidates)

                        # Store in session state
                        st.session_state["verified_items"] = verified_items

                        # Audit log
                        verification_summary = {
                            "exact_match": sum(1 for v in verified_items if v.get("verification_status") == "exact_match"),
                            "fuzzy_needs_review": sum(1 for v in verified_items if v.get("verification_status") == "fuzzy_needs_review"),
                            "failed": sum(1 for v in verified_items if v.get("verification_status") == "failed"),
                        }
                        log_extraction_run(
                            filename=uploaded_file.name,
                            file_path=pdf_path,
                            page_count=total_pages,
                            candidate_count=len(candidate_dicts),
                            llm_items_count=len(extracted_items),
                            verification_summary=verification_summary,
                            model_id=settings.BEDROCK_MODEL_ID,
                            errors=[f"Batch {b} failed" for b in extraction_result.failed_batches],
                        )
                    else:
                        st.warning("No items extracted from AI. All batches may have failed.")

        # --- Display Verified Results ---
        if "verified_items" in st.session_state:
            verified_items = st.session_state["verified_items"]
            verified_df = pd.DataFrame(verified_items)

            st.subheader("Verified Extraction Results")

            # Summary metrics
            exact_count = sum(1 for v in verified_items if v.get("verification_status") == "exact_match")
            fuzzy_count = sum(1 for v in verified_items if v.get("verification_status") == "fuzzy_needs_review")
            failed_count = sum(1 for v in verified_items if v.get("verification_status") == "failed")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Exact match (verified)", exact_count)
            mc2.metric("Needs review", fuzzy_count)
            mc3.metric("Failed", failed_count)

            # Tabs for different views
            tab_all, tab_review, tab_failed = st.tabs(["All Items", "Needs Review", "Failed"])

            with tab_all:
                if not verified_df.empty:
                    def color_status(row):
                        status = row.get("verification_status", "")
                        if status == "exact_match":
                            return ["background-color: #d4edda"] * len(row)
                        elif status == "fuzzy_needs_review":
                            return ["background-color: #fff3cd"] * len(row)
                        else:
                            return ["background-color: #f8d7da"] * len(row)

                    display_cols = [
                        "candidate_id", "page_number", "detail_type",
                        "short_label", "exact_text", "verification_status",
                        "verification_score", "critical_terms_preserved",
                        "human_review_required", "verification_note",
                    ]
                    available = [c for c in display_cols if c in verified_df.columns]

                    styled_df = verified_df[available].style.apply(color_status, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=500)

            with tab_review:
                review_items = [
                    v for v in verified_items
                    if v.get("verification_status") == "fuzzy_needs_review"
                    or v.get("human_review_required") is True
                ]
                if review_items:
                    st.warning(
                        "These items could NOT be verified as exact quotes. "
                        "They MUST be checked against the original PDF before use."
                    )
                    st.dataframe(pd.DataFrame(review_items), use_container_width=True, height=400)
                else:
                    st.success("No items need review — all were exact matches.")

            with tab_failed:
                failed_items = [v for v in verified_items if v.get("verification_status") == "failed"]
                if failed_items:
                    st.error(
                        "These items FAILED verification. The AI may have hallucinated or "
                        "altered the text. DO NOT use these without checking the original."
                    )
                    st.dataframe(pd.DataFrame(failed_items), use_container_width=True, height=400)
                else:
                    st.success("No failed items.")

            # Download
            col_x, col_y = st.columns(2)
            with col_x:
                st.download_button(
                    "Download verified results as JSON",
                    data=json.dumps(verified_items, indent=2, ensure_ascii=False),
                    file_name="verified_tender_extraction.json",
                    mime="application/json",
                )
            with col_y:
                st.download_button(
                    "Download verified results as CSV",
                    data=verified_df.to_csv(index=False),
                    file_name="verified_tender_extraction.csv",
                    mime="text/csv",
                )

        # --- BOQ / Equipment List ---
        if tables:
            with st.spinner("Identifying BOQ / equipment tables..."):
                boq_items = extract_all_boq_items(tables)

            if boq_items:
                st.subheader("Equipment / BOQ Items")
                st.write(
                    "Medical equipment and supplies extracted from Bill of Quantities / "
                    "Schedule of Requirements tables."
                )

                boq_df = pd.DataFrame(boq_items)

                display_cols = [
                    "sn", "item_name", "quantity", "unit", "specifications",
                    "unit_price", "total_price", "lot", "delivery_location",
                    "delivery_period", "remarks", "page_number",
                ]
                available_cols = [c for c in display_cols if c in boq_df.columns]
                st.dataframe(boq_df[available_cols], use_container_width=True, height=450)

                st.metric("Total equipment items", len(boq_items))

                col_boq1, col_boq2 = st.columns(2)
                with col_boq1:
                    st.download_button(
                        "Download BOQ as JSON",
                        data=json.dumps(boq_items, indent=2, ensure_ascii=False),
                        file_name="boq_equipment_list.json",
                        mime="application/json",
                    )
                with col_boq2:
                    st.download_button(
                        "Download BOQ as CSV",
                        data=boq_df.to_csv(index=False),
                        file_name="boq_equipment_list.csv",
                        mime="text/csv",
                    )
            else:
                st.info(
                    "No BOQ / Schedule of Requirements tables detected. "
                    "Equipment lists will appear here if the PDF contains structured item tables."
                )

        # --- Raw Tables ---
        if tables:
            with st.expander("View all extracted tables (raw)"):
                for table in tables[:10]:
                    header = f"**Page {table.page_number} | Table {table.table_number}**"
                    if table.context_before:
                        header += f"\n\n_Context: ...{table.context_before[-100:]}_"
                    st.markdown(header)
                    table_df = pd.DataFrame(table.rows)
                    st.dataframe(table_df, use_container_width=True)

    finally:
        if pdf_path.exists():
            pdf_path.unlink()
