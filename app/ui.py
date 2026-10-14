import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Evaluation & Debugging Platform",
    page_icon="🔎",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #f4f7fb;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #dce5f2;
}
.pipeline {
    background: #f0f5ff;
    border: 1px solid #d8e4fa;
    border-radius: 10px;
    padding: 13px;
    text-align: center;
    min-height: 85px;
}
.small-muted {color: #64748b; font-size: 13px;}
</style>
""", unsafe_allow_html=True)

st.title("🔎 RAG Evaluation & Debugging Platform")
st.caption("Understand  •  Diagnose  •  Improve")

with st.sidebar:
    st.title("🔎 RAG Platform")
    page = st.radio(
        "Navigation",
        [
            "Home",
            "Ingestion",
            "Chunking",
            "Embeddings",
            "Vector Store",
            "Retrieval",
            "Generation",
            "Evaluation",
            "Experiments",
            "Logs & Debug",
        ],
    )
    st.divider()

    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        api_ready = health.ok
    except requests.RequestException:
        api_ready = False

    st.write("System Status")
    if api_ready:
        st.success("API connected")
    else:
        st.error("API unavailable")
        st.caption("Start FastAPI on port 8000.")

    st.caption("Powered by your RAG backend")

st.subheader("RAG Pipeline")

stages = [
    ("1. Ingestion", "Load documents"),
    ("2. Chunking", "Split into chunks"),
    ("3. Embeddings", "Create vectors"),
    ("4. Vector Store", "Qdrant"),
    ("5. Retrieval", "Find relevant chunks"),
    ("6. Context", "Build LLM context"),
    ("7. Generation", "Generate answer"),
]

columns = st.columns(len(stages))
for column, (title, subtitle) in zip(columns, stages):
    with column:
        st.markdown(
            f"""<div class="pipeline">
            <strong>{title}</strong><br>
            <span class="small-muted">{subtitle}</span>
            </div>""",
            unsafe_allow_html=True,
        )

st.divider()

if page == "Home":
    query_col, info_col = st.columns([1.8, 1])

    with query_col:
        st.subheader("Run a Query")

        with st.form("rag_query_form"):
            question = st.text_area(
                "Ask about your documents",
                placeholder="What is Retrieval-Augmented Generation?",
                height=100,
            )
            top_k = st.select_slider(
                "Top-K retrieved chunks",
                options=[1, 3, 5, 8, 10, 15, 20],
                value=5,
            )
            submitted = st.form_submit_button(
                "🔍 Run Query",
                type="primary",
                use_container_width=True,
            )

    with info_col:
        st.subheader("Quick Experiments")
        st.info(
            "**Retrieval comparison**\n\n"
            "Compare vector, BM25 and hybrid search when available."
        )
        st.info(
            "**Top-K analysis**\n\n"
            "Observe how the number of retrieved chunks changes."
        )
        st.info(
            "**Evaluation**\n\n"
            "Run your evaluation dataset to measure answer quality."
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving context and generating answer..."):
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question.strip(),
                            "top_k": top_k,
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    data = response.json()

                st.session_state["last_result"] = data

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI. Start the API first.")
            except requests.exceptions.Timeout:
                st.error("The request timed out. Try again.")
            except requests.exceptions.HTTPError as exc:
                st.error(
                    f"API error {exc.response.status_code}: "
                    f"{exc.response.text}"
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

    data = st.session_state.get("last_result")

    if data:
        st.divider()

        result_col, answer_col = st.columns([1.2, 1])

        with result_col:
            st.subheader("Retrieval Results")

            sources = data.get("sources", [])
            st.metric("Retrieved chunks", data.get("retrieved_count", 0))

            if sources:
                table = pd.DataFrame([
                    {
                        "Chunk ID": source.get("chunk_id", "Unknown"),
                        "Source": source.get("source", "Unknown"),
                        "Page": source.get("page"),
                        "Score": source.get("score"),
                    }
                    for source in sources
                ])
                st.dataframe(table, use_container_width=True, hide_index=True)
            else:
                st.info("No source chunks were returned.")

        with answer_col:
            st.subheader("Generated Answer")

            if data.get("refused"):
                st.warning(data.get("answer", "No answer available."))
            else:
                st.success(data.get("answer", "No answer available."))

            st.caption(
                "Refused: " + str(data.get("refused", False))
            )

        context_col, sources_col = st.columns([1.3, 1])

        with context_col:
            st.subheader("Retrieved Context")
            context = data.get("context_preview", "")

            if context:
                st.text_area(
                    "Context sent for generation (preview)",
                    value=context,
                    height=300,
                    disabled=True,
                )
            else:
                st.info("No context was returned.")

        with sources_col:
            st.subheader("Sources Used")

            for index, source in enumerate(data.get("sources", []), start=1):
                page_number = source.get("page")
                with st.expander(
                    f"Source {index} — Page {page_number or 'Unknown'}"
                ):
                    st.write(source.get("source", "Unknown"))
                    st.write(f"Score: {source.get('score', 'N/A')}")
                    st.write(source.get("preview", "No preview available."))

        with st.expander("Debug details"):
            st.json({
                "question": data.get("question"),
                "retrieved_count": data.get("retrieved_count"),
                "refused": data.get("refused"),
                "source_count": len(data.get("sources", [])),
            })

elif page == "Retrieval":
    st.subheader("Retrieval Debugging")
    st.write(
        "Your current API supports the existing retrieval pipeline and Top-K. "
        "BM25, hybrid search and metadata-filter controls need dedicated API "
        "endpoints before they can be run from this page."
    )
    st.code("POST /ask  →  question + top_k")

elif page == "Evaluation":
    st.subheader("Evaluation Metrics")
    st.info(
        "Run the existing evaluation modules and connect their real outputs "
        "here. No evaluation scores are fabricated by this dashboard."
    )
    st.write(
        "Existing modules: retrieval_eval.py, context_eval.py, "
        "answer_eval.py and evaluation_report.py"
    )

elif page == "Logs & Debug":
    st.subheader("Logs & Debug")
    st.write(
        "The API currently logs requests, retrieval counts, refusal status "
        "and unexpected exceptions to its server console."
    )
    st.info(
        "Persistent searchable logs and LangSmith tracing require additional "
        "integration."
    )

else:
    st.subheader(page)
    st.info(
        "This dashboard section is a navigation placeholder. "
        "Its functionality will be connected to the corresponding ingestion, "
        "chunking, embedding, vector-store, generation or experiment modules."
    )