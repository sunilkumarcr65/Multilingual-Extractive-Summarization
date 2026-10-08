from __future__ import annotations

import streamlit as st

from pipelines.qa_pipeline import answer_question

# ---------------------------------------------------------
# Context Renderer
# ---------------------------------------------------------

def render_context(context: str):
    if not context:
        st.caption("No context retrieved.")
        return
    with st.expander("📌 Retrieved Context"):
        st.write(context)


# ---------------------------------------------------------
# Chunk Renderer
# ---------------------------------------------------------

def render_retrieved_chunks(chunks):
    if not chunks:
        return
    with st.expander("📚 Retrieved Chunks"):
        for index, chunk in enumerate(chunks, start=1):
            st.markdown(f"### Chunk {index}")
            if isinstance(chunk, dict):
                score = chunk.get("score")
                if score is not None:
                    st.caption(f"Score: {score:.4f}")
                st.write(chunk.get("text", ""))
            else:
                st.write(str(chunk))
            st.divider()


# ---------------------------------------------------------
# Main QA Section
# ---------------------------------------------------------

def render_qa_section(
    file_path: str,
    ocr_language: str = "eng",
):
    st.header("🔍 Ask Questions About the Document")

    # ----------------------------------------------------------
    # FIX 1: st.form captures the Enter key and routes it to
    # form_submit_button — without this, Enter triggers a full
    # Streamlit rerun before the button click handler fires,
    # so the answer never generates.
    # ----------------------------------------------------------

    with st.form(key="qa_form", clear_on_submit=False):
        question = st.text_input(
            "Your question:",
            placeholder="Ask anything about the uploaded document...",
        )
        submitted = st.form_submit_button(
            "Get Answer",
            use_container_width=True,
        )

    # ----------------------------------------------------------
    # FIX 2: All logic runs OUTSIDE the form block.
    # Inside st.form, Streamlit only collects widget values —
    # no API calls, no st.write, no spinner allowed in there.
    # ----------------------------------------------------------

    if submitted:
        question = (question or "").strip()

        if not question:
            st.warning("Please enter a question.")
            # Clear any previous answer so stale results don't show
            st.session_state.pop("qa_result", None)
        else:
            with st.spinner("Searching document..."):
                try:
                    result = answer_question(
                        file_path=file_path,
                        question=question,
                        ocr_language=ocr_language,
                    )
                    # ------------------------------------------
                    # FIX 3: Persist result in session state.
                    # Without this, Streamlit's rerun after
                    # form submission wipes the answer before
                    # it renders.
                    # ------------------------------------------
                    st.session_state["qa_result"] = result
                    st.session_state["qa_question"] = question

                except Exception as error:
                    st.error(f"Q&A failed: {error}")
                    st.session_state.pop("qa_result", None)

    # ----------------------------------------------------------
    # FIX 4: Render from session state — survives reruns.
    # This block runs on every rerun (including the one triggered
    # by form submission), so the answer is always visible as long
    # as session state holds it.
    # ----------------------------------------------------------

    result = st.session_state.get("qa_result")
    if result is None:
        return

    if not result.get("success", True):
        st.error(
            result.get("error", {}).get("message", "Unknown error.")
        )
        return

    answer           = result.get("answer", "")
    context          = result.get("context", "")
    retrieved_chunks = result.get("retrieved_chunks", [])
    processing_time  = result.get("processing_time_seconds")
    question_shown   = st.session_state.get("qa_question", "")

    if question_shown:
        st.caption(f"Q: _{question_shown}_")

    st.subheader("🧠 Answer")
    st.write(answer)

    if processing_time is not None:
        st.caption(f"Processing time: {processing_time:.2f}s")

    render_context(context)
    render_retrieved_chunks(retrieved_chunks)
