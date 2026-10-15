"""
RAG Evaluation & Debugging Platform — Streamlit UI

Pages:
    Dashboard  — overview + recent activity
    Ask        — query interface
    Evaluation — run metrics with charts
    Compare    — two runs side by side
    Traces     — recent query log from SQLite
    About      — architecture & info
"""
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "evaluation"
RUNS_DIR = EVAL_DIR / "runs"
REPORTS_DIR = EVAL_DIR / "reports"
TRACES_DB = PROJECT_ROOT / "data" / "traces" / "traces.db"
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="RAG Evaluation Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hide default Streamlit header/footer */
    #MainMenu, footer, header {visibility: hidden;}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
        border-right: 1px solid #374151;
    }
    [data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #9CA3AF !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 2px;
        transition: all 0.15s;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(124, 58, 237, 0.15);
    }

    /* ---------- Hero header ---------- */
    .hero {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(124, 58, 237, 0.4);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .hero p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        backdrop-filter: blur(10px);
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: all 0.2s;
        height: 100%;
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    .metric-card .label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6B7280;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 0.25rem;
    }
    .metric-card .value.good {color: #059669;}
    .metric-card .value.warn {color: #D97706;}
    .metric-card .value.bad  {color: #DC2626;}

    /* ---------- Pipeline cards ---------- */
    .pipeline-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem 0.5rem;
        text-align: center;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.2s;
    }
    .pipeline-card:hover {
        border-color: #7C3AED;
        background: #F5F3FF;
    }
    .pipeline-card .icon {
        font-size: 1.4rem;
        margin-bottom: 0.3rem;
    }
    .pipeline-card .title {
        font-weight: 600;
        font-size: 0.85rem;
        color: #111827;
    }
    .pipeline-card .sub {
        font-size: 0.7rem;
        color: #6B7280;
        margin-top: 2px;
    }

    /* ---------- Status badges ---------- */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge.ok      {background: #D1FAE5; color: #065F46;}
    .badge.warn    {background: #FEF3C7; color: #92400E;}
    .badge.bad     {background: #FEE2E2; color: #991B1B;}
    .badge.neutral {background: #E5E7EB; color: #374151;}

    /* ---------- Section headers ---------- */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ---------- Empty state ---------- */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #6B7280;
        background: #F9FAFB;
        border-radius: 12px;
        border: 2px dashed #E5E7EB;
    }
    .empty-state .icon {font-size: 2.5rem; margin-bottom: 0.5rem;}
    .empty-state h3 {color: #374151; margin: 0.5rem 0;}

    /* ---------- Streamlit element tweaks ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        box-shadow: 0 2px 6px rgba(124, 58, 237, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
    }
    .stTextArea textarea, .stTextInput input {
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #7C3AED;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
    }

    /* ---------- Dataframe ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hero(title, subtitle, badge=None):
    badge_html = f'<div class="badge">{badge}</div>' if badge else ""
    # Single line — no blank lines inside, otherwise Streamlit breaks the div
    st.markdown(
        f'<div class="hero">{badge_html}<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def metric_card(label, value, sub=None, kind=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value {kind}">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def empty_state(icon, title, subtitle):
    st.markdown(f"""
        <div class="empty-state">
            <div class="icon">{icon}</div>
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)


def status_badge(text, kind="neutral"):
    return f'<span class="badge {kind}">{text}</span>'


@st.cache_data(ttl=30)
def load_runs():
    """All runs from evaluation/runs/*.json (newest first)."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for path in sorted(RUNS_DIR.glob("run_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            runs.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return runs


@st.cache_data(ttl=30)
def load_traces(limit=200):
    """Recent traces from SQLite."""
    if not TRACES_DB.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(TRACES_DB) as conn:
            df = pd.read_sql_query(
                f"SELECT trace_id, created_at, question, action, refused, "
                f"failed_stage, latency_ms, total_tokens, run_id, config_label "
                f"FROM traces ORDER BY created_at DESC LIMIT {int(limit)}",
                conn,
            )
        return df
    except Exception:
        return pd.DataFrame()

def call_api(question, top_k, preset=None):
    """POST to /ask endpoint."""
    payload = {"question": question, "top_k": top_k}
    if preset:
        payload["preset"] = preset

    try:
        r = requests.post(
            f"{API_URL}/ask",
            json=payload,
            timeout=180,
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Start FastAPI on port 8000."
    except requests.exceptions.Timeout:
        return None, "Request timed out."
    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Request failed: {e}"

def api_status():
    """Check API health."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.ok
    except Exception:
        return False

# ===========================================================================
# Mega Dashboard — everything in one page
# ===========================================================================

def _pipeline_diagram():
    stages = [
        ("📄", "Ingestion", "PDF → text"),
        ("✂️", "Chunking", "Text → chunks"),
        ("🔢", "Embeddings", "Chunks → vectors"),
        ("🗄️", "Vector Store", "Qdrant"),
        ("🔍", "Retrieval", "Find relevant"),
        ("📝", "Context", "Build prompt"),
        ("🤖", "Generation", "LLM answer"),
    ]
    cols = st.columns(len(stages))
    for col, (icon, title, sub) in zip(cols, stages):
        with col:
            st.markdown(f"""
                <div class="pipeline-card">
                    <div class="icon">{icon}</div>
                    <div class="title">{title}</div>
                    <div class="sub">{sub}</div>
                </div>
            """, unsafe_allow_html=True)


def _quick_stats():
    runs = load_runs()
    traces = load_traces(limit=1000)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Runs", len(runs), "evaluation runs")
    with col2:
        metric_card("Total Traces", len(traces), "queries processed")
    with col3:
        if not traces.empty and "refused" in traces:
            metric_card("Refusal Rate", f"{traces['refused'].mean() * 100:.1f}%", "of all queries")
        else:
            metric_card("Refusal Rate", "n/a", "no data")
    with col4:
        if not traces.empty and "latency_ms" in traces:
            metric_card("Avg Latency", f"{traces['latency_ms'].mean():.0f} ms", "per query")
        else:
            metric_card("Avg Latency", "n/a", "no data")


# ---------- ASK section ----------
def _section_ask():
    st.markdown('<div class="section-title">💬 Ask the Documents</div>', unsafe_allow_html=True)

    # Preset selector (full width)
    presets = [
        "vector",              # baseline vector search
        "bm25",                # keyword only
        "hybrid",              # vector + BM25 (RRF)
        "hybrid_weighted",     # vector + BM25 (weighted)
        "vector_rerank",       # vector + Cohere rerank
        "hybrid_rerank",       # hybrid + rerank (recommended)
        "hyde",                # hypothetical document embeddings
        "expand",              # query expansion
        "multi_query",         # multi-query fusion
        "hybrid_rerank_crag",  # hybrid + rerank + corrective RAG
    ]

    preset_labels = {
        "vector": "🎯 Vector (baseline)",
        "bm25": "🔤 BM25 (keyword)",
        "hybrid": "🔀 Hybrid (RRF)",
        "hybrid_weighted": "⚖️ Hybrid (weighted)",
        "vector_rerank": "🎯 Vector + Rerank",
        "hybrid_rerank": "⭐ Hybrid + Rerank",
        "hyde": "💭 HyDE",
        "expand": "📈 Query Expansion",
        "multi_query": "🔀 Multi-Query",
        "hybrid_rerank_crag": "🔬 Hybrid + Rerank + CRAG",
    }

    preset = st.selectbox(
        "Pipeline preset",
        options=presets,
        index=5,  # default to hybrid_rerank
        format_func=lambda x: preset_labels.get(x, x),
        key="ask_preset",
    )

    st.caption(f"**Pipeline:** `{preset}` — {preset_labels.get(preset, '')}")

    col1, col2 = st.columns([2, 1])
    with col1:
        question = st.text_area(
            "Your question",
            placeholder="What is Retrieval-Augmented Generation?",
            height=100,
            key="ask_q",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.select_slider("Top-K", options=[1, 3, 5, 8, 10, 15, 20], value=5, key="ask_k")
        run_btn = st.button("🚀 Run Query", use_container_width=True, type="primary", key="ask_btn")

    if run_btn:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating..."):
                data, error = call_api(question.strip(), top_k,preset)
            if error:
                st.error(error)
            else:
                st.session_state["last_result"] = data
                st.rerun()

    data = st.session_state.get("last_result")
    if not data:
        return

    # Answer
    if data.get("refused"):
        st.warning(f"**Refused:** {data.get('answer', '')}")
    else:
        st.success(data.get("answer", "No answer"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Retrieved", data.get("retrieved_count", 0), "chunks")
    with c2:
        metric_card("Latency", f"{data.get('latency_ms', 0):.0f} ms", "total")
    with c3:
        usage = data.get("usage") or {}
        metric_card("Tokens", usage.get("total_tokens", 0), "used")
    with c4:
        metric_card("Config", data.get("config_label", "-"), "pipeline")

    # Sources
    sources = data.get("sources", [])
    with st.expander(f"📚 Sources ({len(sources)})", expanded=True):
        if sources:
            df = pd.DataFrame([{
                "Chunk ID": s.get("chunk_id", "-"),
                "Source": s.get("source", "-"),
                "Page": s.get("page", "-"),
                "Score": f"{s.get('score', 0):.4f}" if s.get("score") is not None else "-",
            } for s in sources])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No sources retrieved.")

    # Stage breakdown
    trace_id = data.get("trace_id")
    if trace_id:
        with st.expander("🔬 Pipeline stage breakdown"):
            try:
                r = requests.get(f"{API_URL}/api/traces/{trace_id}", timeout=10)
                r.raise_for_status()
                trace = r.json()
            except Exception as e:
                st.error(f"Cannot load trace: {e}")
                trace = None

            if trace:
                for stage in trace.get("stages", []):
                    name = stage.get("name", "-")
                    status = stage.get("status", "ok")
                    latency = stage.get("latency_ms", 0)
                    details = stage.get("details", {}) or {}
                    error = stage.get("error")

                    icon = {"ok": "✅", "skipped": "⏭️", "fallback": "⚠️", "error": "❌"}.get(status, "•")

                    # Show stage + optional expansion for its details
                    if details or error:
                        with st.expander(f"{icon} **{name}** — {status} ({latency} ms)", expanded=False):
                            if error:
                                st.error(f"Error: {error}")

                            # ---- Special rendering per stage ----
                            if name == "query_transform":
                                if "hypothetical_document" in details:
                                    st.markdown("**💭 Hypothetical passage (HyDE):**")
                                    st.info(details["hypothetical_document"])
                                    st.caption(f"embed_as={details.get('embed_as')}, query_weight={details.get('query_weight')}")
                                if "queries" in details:
                                    st.markdown("**🔀 Generated query variants:**")
                                    for q in details["queries"]:
                                        st.markdown(f"- {q}")
                                if "added_terms" in details:
                                    st.markdown(f"**📈 Added terms:** {', '.join(details['added_terms'])}")

                            elif name == "rewrite":
                                st.markdown(f"**Original:** {details.get('original', '-')}")
                                st.markdown(f"**Rewritten:** {details.get('rewritten', '-')}")
                                st.caption(f"changed={details.get('changed')}, history_turns={details.get('history_turns')}")

                            elif name == "retrieve":
                                st.markdown(f"**Method:** `{details.get('method', '-')}`")
                                st.markdown(f"**Query mode:** `{details.get('query_mode', '-')}`")
                                st.markdown(f"**Returned:** {details.get('returned', 0)} chunks")
                                results = details.get("results", [])
                                if results:
                                    st.markdown("**Top chunks:**")
                                    for r in results[:5]:
                                        st.caption(
                                            f"Rank {r.get('rank', '-')} — "
                                            f"chunk {r.get('chunk_id')} "
                                            f"(page {r.get('page_number', '?')}, score {r.get('score', 0):.3f})"
                                        )

                            elif name == "rerank":
                                st.markdown(f"**Candidates:** {details.get('candidates', '-')}")
                                st.markdown(f"**Kept:** {details.get('kept', '-')}")
                                changes = details.get("changes", [])
                                if changes:
                                    st.markdown("**Rank changes (before → after):**")
                                    for c in changes[:10]:
                                        before = c.get("before", "-")
                                        after = c.get("after", "-")
                                        delta = c.get("change", 0)
                                        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
                                        st.caption(
                                            f"Chunk {c.get('chunk_id')} (page {c.get('page_number')}): "
                                            f"{before} → {after} {arrow}"
                                        )

                            elif name == "crag":
                                st.markdown(f"**Action:** `{details.get('action', '-')}`")
                                st.markdown(f"**Strategy:** `{details.get('strategy', '-')}`")
                                grades = details.get("grades", [])
                                if grades:
                                    st.markdown("**Grades:**")
                                    for i, g in enumerate(grades, 1):
                                        st.markdown(
                                            f"Attempt {i}: **{g.get('verdict', '?')}** — "
                                            f"{g.get('reason', 'no reason')}"
                                        )

                            elif name == "compress":
                                st.markdown(f"**Original:** {details.get('original_chars', 0)} chars")
                                st.markdown(f"**Compressed:** {details.get('compressed_chars', 0)} chars")
                                st.markdown(f"**Ratio:** {details.get('ratio', 1.0):.2f}")

                            elif name == "context":
                                st.markdown(f"**Chunks:** {details.get('chunks', 0)}")
                                st.markdown(f"**Chars:** {details.get('chars', 0)}")

                            elif name == "generate":
                                st.markdown(f"**Model:** `{details.get('model', '-')}`")
                                st.markdown(f"**LLM called:** {details.get('llm_called', False)}")
                                st.markdown(f"**Latency:** {details.get('llm_latency_ms', 0)} ms")
                                st.markdown(f"**Refused:** {details.get('refused', False)}")

                            else:
                                # Fallback: dump raw details
                                st.json(details)
                    else:
                        st.markdown(f"{icon} **{name}** — {status} ({latency} ms)")

                st.markdown("---")
                st.markdown("**📝 Context sent to LLM**")
                st.text_area(
                    "Context", value=trace.get("context", ""),
                    height=180, disabled=True, key=f"ctx_{trace_id}",
                    label_visibility="collapsed",
                )

            

# ---------- RETRIEVAL section ----------

def _ret_methods():
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Query", value="What is Retrieval-Augmented Generation?", key="ret_q")
    with col2:
        top_k = st.select_slider("Top-K", options=[3, 5, 10, 15, 20], value=5, key="ret_k")

    methods = st.multiselect(
        "Methods",
        options=["vector", "bm25", "hybrid"],
        default=["vector", "bm25", "hybrid"],
        key="ret_methods",
    )

    if st.button("🔍 Compare Retrieval", type="primary", use_container_width=True, key="ret_btn"):
        if not query.strip():
            st.warning("Please enter a query.")
        elif not methods:
            st.warning("Select at least one method.")
        else:
            with st.spinner("Running retrieval..."):
                try:
                    r = requests.post(f"{API_URL}/api/retrieve", json={
                        "query": query.strip(),
                        "methods": methods,
                        "top_k": top_k,
                    }, timeout=60)
                    r.raise_for_status()
                    st.session_state["last_retrieval"] = r.json()
                except Exception as e:
                    st.error(f"Request failed: {e}")

    data = st.session_state.get("last_retrieval")
    if not data:
        st.info("Enter a query and click Compare Retrieval.")
        return

    results = data.get("results", {})
    chunk_sets = {m: {c["chunk_id"] for c in v} for m, v in results.items() if isinstance(v, list)}

    if len(chunk_sets) >= 2:
        all_ids = set()
        for s in chunk_sets.values():
            all_ids.update(s)
        overlap_rows = []
        for cid in sorted(all_ids):
            row = {"Chunk ID": cid}
            for m in chunk_sets:
                row[m] = "✓" if cid in chunk_sets[m] else ""
            overlap_rows.append(row)
        st.caption(f"**{len(all_ids)}** unique chunks across methods")
        st.dataframe(pd.DataFrame(overlap_rows), use_container_width=True, hide_index=True)

    valid = [m for m in results if isinstance(results[m], list)]
    if valid:
        tabs = st.tabs([m.upper() for m in valid])
        for tab, method in zip(tabs, valid):
            with tab:
                chunks = results[method]
                df = pd.DataFrame([{
                    "#": c["rank"],
                    "Chunk ID": c["chunk_id"],
                    "Page": c["page"],
                    "Score": f"{c['score']:.4f}" if c.get("score") is not None else "-",
                    "Preview": (c.get("text", "")[:100] + "...") if c.get("text") else "",
                } for c in chunks])
                st.dataframe(df, use_container_width=True, hide_index=True)


def _ret_chunking():
    col1, col2 = st.columns([3, 1])
    with col1:
        strategies = st.multiselect(
            "Strategies",
            options=["recursive", "fixed", "structure_aware"],
            default=["recursive", "fixed", "structure_aware"],
            key="chunk_strats",
        )
    with col2:
        num_pages = st.slider("Pages", 5, 30, 10, key="chunk_pages")

    if st.button("✂️ Run Chunking Comparison", type="primary", use_container_width=True, key="chunk_btn"):
        if not strategies:
            st.warning("Select at least one strategy.")
        else:
            with st.spinner("Chunking..."):
                try:
                    r = requests.post(f"{API_URL}/api/chunking/compare", json={
                        "num_pages": num_pages,
                        "strategies": strategies,
                    }, timeout=120)
                    r.raise_for_status()
                    st.session_state["last_chunking"] = r.json()
                except Exception as e:
                    st.error(f"Request failed: {e}")

    data = st.session_state.get("last_chunking")
    if not data:
        st.info("Select strategies and click Run.")
        return

    results = data.get("results", {})
    valid = {k: v for k, v in results.items() if "error" not in v}

    if valid:
        summary = [{
            "Strategy": name,
            "Chunks": info["count"],
            "Avg chars": info["avg_chars"],
            "Min": info["min_chars"],
            "Max": info["max_chars"],
        } for name, info in valid.items()]
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        tabs = st.tabs([k.replace("_", " ").title() for k in valid])
        for tab, (name, info) in zip(tabs, valid.items()):
            with tab:
                st.caption(f"3 of {info['count']} chunks — avg {info['avg_chars']} chars")
                for sample in info.get("samples", []):
                    with st.expander(f"Chunk #{sample['index']} ({sample['length']} chars, page {sample.get('page', '?')})"):
                        st.text(sample["text"])


def _section_retrieval():
    tab1, tab2 = st.tabs(["🔀 Methods", "✂️ Chunking"])
    with tab1:
        _ret_methods()
    with tab2:
        _ret_chunking()


# ---------- EVALUATION section ----------

def _eval_runs():
    runs = load_runs()

    # ---- Run new evaluation (trigger) ----
    with st.expander("⚡ Run new evaluation", expanded=False):
        presets = [
            "vector", "bm25", "hybrid", "hybrid_weighted",
            "vector_rerank", "hybrid_rerank",
            "hyde", "expand", "multi_query", "hybrid_rerank_crag",
        ]
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            preset = st.selectbox("Preset", options=presets, index=0, key="eval_preset")
        with col2:
            limit = st.selectbox(
                "Questions",
                options=[5, 10, 20, 60],
                index=0,
                key="eval_limit",
            )
        with col3:
            use_judge = st.checkbox("LLM judge", value=False, key="eval_judge")

        if st.button("▶️ Start Evaluation", type="primary", use_container_width=True, key="eval_start"):
            with st.spinner(f"Running {limit} questions with {preset}... (this may take 5-30 min)"):
                try:
                    r = requests.post(
                        f"{API_URL}/api/evaluation/run",
                        json={"preset": preset, "limit": limit, "use_judge": use_judge},
                        timeout=3600,
                    )
                    r.raise_for_status()
                    result = r.json()

                    st.success(f"✅ Run complete: `{result['run_id']}`")

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        v = result.get("hit@5")
                        metric_card("Hit@5", f"{v * 100:.0f}%" if v is not None else "-", "top 5")
                    with c2:
                        v = result.get("mrr")
                        metric_card("MRR", f"{v:.2f}" if v is not None else "-", "rank")
                    with c3:
                        v = result.get("answerable_ok_rate")
                        metric_card("Answerable OK", f"{v * 100:.0f}%" if v is not None else "-", "correct")
                    with c4:
                        v = result.get("unanswerable_ok_rate")
                        metric_card("Refusal OK", f"{v * 100:.0f}%" if v is not None else "-", "refused")

                    st.caption("Refresh the page to see this run in the dropdown below.")
                except requests.exceptions.Timeout:
                    st.error("Evaluation timed out (>1 hour). Try fewer questions.")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    # ---- Existing run selector ----
    if not runs:
        st.info("No evaluation runs yet. Run one above or from CLI.")
        return

    run_labels = {f"{r['run_id']} — {r.get('label', '?')} ({r.get('status', '?')})": r for r in runs}
    selected = st.selectbox("Select a run", list(run_labels.keys()), key="eval_run")
    run = run_labels[selected]
    summary = run.get("summary")

    if not summary:
        st.warning("This run is not complete yet — no summary available.")
        return

   
    e2e = summary.get("end_to_end") or {}
    retrieval = summary.get("retrieval", {}).get("all", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        v = e2e.get("answerable_ok_rate")
        metric_card("Answerable OK", f"{v * 100:.0f}%" if v is not None else "-", "end-to-end", "good" if v and v > 0.7 else "warn")
    with col2:
        v = e2e.get("unanswerable_ok_rate")
        metric_card("Correct Refusal", f"{v * 100:.0f}%" if v is not None else "-", "unanswerable", "good" if v and v > 0.7 else "warn")
    with col3:
        v = retrieval.get("hit@5")
        metric_card("Hit@5", f"{v * 100:.0f}%" if v is not None else "-", "evidence in top 5", "good" if v and v > 0.8 else "warn")
    with col4:
        v = retrieval.get("mrr")
        metric_card("MRR", f"{v:.2f}" if v is not None else "-", "mean reciprocal rank")

    ks = run.get("settings", {}).get("ks", [1, 3, 5, 10])
    chart_data = [{
        "k": k,
        "Hit@k": retrieval.get(f"hit@{k}", 0) or 0,
        "Recall@k": retrieval.get(f"recall@{k}", 0) or 0,
        "nDCG@k": retrieval.get(f"ndcg@{k}", 0) or 0,
    } for k in ks]
    df_chart = pd.DataFrame(chart_data).melt(id_vars="k", var_name="Metric", value_name="Score")
    fig = px.line(df_chart, x="k", y="Score", color="Metric", markers=True,
                  color_discrete_sequence=["#7C3AED", "#4F46E5", "#EC4899"])
    fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=20, r=20, t=20, b=20),
                      yaxis=dict(range=[0, 1.05], gridcolor="#F3F4F6"),
                      xaxis=dict(gridcolor="#F3F4F6"))
    st.plotly_chart(fig, use_container_width=True)


def _eval_compare():
    runs = load_runs()
    if len(runs) < 2:
        st.info("Need at least 2 runs to compare.")
        return

    labels = {f"{r['run_id']} — {r.get('label', '?')}": r for r in runs}
    col1, col2 = st.columns(2)
    with col1:
        a_label = st.selectbox("Baseline (A)", list(labels.keys()), index=0, key="cmp_a")
    with col2:
        b_label = st.selectbox("Comparison (B)", list(labels.keys()), index=min(1, len(labels) - 1), key="cmp_b")

    run_a = labels[a_label]
    run_b = labels[b_label]

    if run_a["run_id"] == run_b["run_id"]:
        st.warning("Select two different runs.")
        return

    def get_metric(run, path):
        val = run.get("summary") or {}
        for key in path:
            val = (val or {}).get(key, {}) if isinstance(val, dict) else {}
        return val if isinstance(val, (int, float)) else None

    metrics = [
        ("Answerable OK", ["end_to_end", "answerable_ok_rate"], "%"),
        ("Correct Refusal", ["end_to_end", "unanswerable_ok_rate"], "%"),
        ("Hit@5", ["retrieval", "all", "hit@5"], "%"),
        ("MRR", ["retrieval", "all", "mrr"], "num"),
    ]

    rows = []
    for name, path, fmt in metrics:
        a = get_metric(run_a, path)
        b = get_metric(run_b, path)
        if fmt == "%":
            a_str = f"{a * 100:.1f}%" if a is not None else "-"
            b_str = f"{b * 100:.1f}%" if b is not None else "-"
            delta = f"{(b - a) * 100:+.1f}%" if a is not None and b is not None else "-"
        else:
            a_str = f"{a:.3f}" if a is not None else "-"
            b_str = f"{b:.3f}" if b is not None else "-"
            delta = f"{b - a:+.3f}" if a is not None and b is not None else "-"
        rows.append({"Metric": name, "A": a_str, "B": b_str, "Δ (B-A)": delta})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _section_evaluation():
    tab1, tab2 = st.tabs(["📊 Runs", "⚖️ Compare"])
    with tab1:
        _eval_runs()
    with tab2:
        _eval_compare()


# ---------- TRACES section ----------

def _section_traces():
    traces = load_traces(limit=500)
    if traces.empty:
        st.info("No traces yet. Ask a question first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        actions = ["All"] + sorted(traces["action"].dropna().unique().tolist())
        action_filter = st.selectbox("Action", actions, key="tr_a")
    with col2:
        refused_filter = st.selectbox("Refused", ["All", "Yes", "No"], key="tr_r")
    with col3:
        search = st.text_input("Search question", placeholder="keyword...", key="tr_s")

    filtered = traces.copy()
    if action_filter != "All":
        filtered = filtered[filtered["action"] == action_filter]
    if refused_filter == "Yes":
        filtered = filtered[filtered["refused"] == 1]
    elif refused_filter == "No":
        filtered = filtered[filtered["refused"] == 0]
    if search:
        filtered = filtered[filtered["question"].str.contains(search, case=False, na=False)]

    st.caption(f"**{len(filtered)}** traces")

    display = filtered[["created_at", "question", "action", "refused", "latency_ms", "total_tokens"]].copy()
    display["created_at"] = display["created_at"].str[:19]
    display["question"] = display["question"].str[:60] + "..."
    display["refused"] = display["refused"].map({0: "No", 1: "Yes"})
    display.columns = ["Time", "Question", "Action", "Refused", "Latency", "Tokens"]
    st.dataframe(display, use_container_width=True, hide_index=True)


# ---------- COLLECTIONS section ----------

@st.cache_data(ttl=30)
def _fetch_collections():
    try:
        r = requests.get(f"{API_URL}/api/collections", timeout=10)
        r.raise_for_status()
        return r.json().get("collections", [])
    except Exception:
        return []


def _section_collections():
    collections = _fetch_collections()
    if not collections:
        st.info("No collections. Run an ingestion first.")
        return

    df = pd.DataFrame([{
        "Name": c["name"],
        "Points": c.get("points_count", "-"),
        "Strategy": c.get("strategy", "-"),
        "Chunk size": c.get("chunk_size", "-"),
        "Overlap": c.get("chunk_overlap", "-"),
        "Model": c.get("embedding_model", "-"),
    } for c in collections])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Browse chunks from", [c["name"] for c in collections], key="col_pick")
    with col2:
        limit = st.select_slider("Show", options=[10, 20, 50], value=20, key="col_limit")

    try:
        r = requests.get(
            f"{API_URL}/api/collections/{selected}/chunks",
            params={"limit": limit, "offset": 0},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"Cannot load chunks: {e}")
        return

    st.caption(f"Total: **{data.get('total_points', 0)}** points — showing {len(data.get('chunks', []))}")

    for c in data.get("chunks", []):
        title = f"Chunk {c.get('chunk_id')} — page {c.get('page', '?')} ({c.get('strategy', '-')})"
        with st.expander(title):
            st.caption(c.get("text", "No text"))

def _section_ingest():
    st.markdown("**Ingest the bundled PDF or upload your own**")
    st.caption("This will chunk, embed, and store in Qdrant.")

    # Source
    source = st.radio(
        "Source",
        ["Bundled PDF (rag_survey.pdf)", "Upload new"],
        horizontal=True,
        key="ingest_src",
    )

    uploaded_file = None
    if source == "Upload new":
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], key="ingest_file")

    col1, col2, col3 = st.columns(3)
    with col1:
        strategy = st.selectbox(
            "Strategy",
            options=["recursive", "fixed", "structure_aware"],
            key="ingest_strategy",
        )
    with col2:
        chunk_size = st.number_input("Chunk size", value=1000, min_value=100, max_value=5000, step=100, key="ingest_cs")
    with col3:
        chunk_overlap = st.number_input("Overlap", value=150, min_value=0, max_value=1000, step=50, key="ingest_co")

    recreate = st.checkbox("Recreate collection (delete existing data)", value=False, key="ingest_recreate")

    if st.button("⚡ Run Ingestion", type="primary", use_container_width=True, key="ingest_btn"):
        if source == "Upload new" and not uploaded_file:
            st.warning("Please upload a PDF first.")
            return

        # Only bundled for now
        if source == "Upload new":
            st.info("Upload support coming soon. Using bundled PDF for now.")

        with st.spinner(f"Ingesting with {strategy} strategy..."):
            try:
                r = requests.post(
                    f"{API_URL}/api/ingest",
                    json={
                        "strategy": strategy,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "recreate": recreate,
                    },
                    timeout=600,
                )
                r.raise_for_status()
                result = r.json()

                st.success(f"✅ Ingestion complete!")
                summary = result.get("summary", {})

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card("Chunks", summary.get("chunks", 0), "stored")
                with c2:
                    metric_card("Pages", summary.get("pages", 0), "processed")
                with c3:
                    metric_card("Strategy", summary.get("strategy", "-"), "chunking")
                with c4:
                    metric_card("Collection", summary.get("collection", "-")[:20], "in Qdrant")

                st.caption(
                    f"Avg chunk: {summary.get('avg_chunk_chars', 0)} chars · "
                    f"Min: {summary.get('min_chunk_chars', 0)} · "
                    f"Max: {summary.get('max_chunk_chars', 0)}"
                )
                st.json(summary.get("timings", {}))
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
# ---------- ABOUT section ----------

def _section_about():
    st.markdown("""
### What This Platform Does

A production-grade RAG evaluation and debugging system that:

- **Ingests** PDFs with multiple chunking strategies (recursive, fixed, semantic, structure-aware)
- **Embeds** chunks using Cohere's `embed-english-v3.0` (1024-dim)
- **Stores** vectors in Qdrant with UUIDv5 deduplication
- **Retrieves** using vector search, BM25, or hybrid (RRF / weighted fusion)
- **Reranks** with Cohere's rerank model
- **Generates** grounded answers with Groq LLM (refusal-aware)
- **Evaluates** with hit@k, MRR, nDCG, precision, LLM-judged faithfulness & correctness
- **Traces** every request to a SQLite store for debugging
- **Diagnoses** failures: retrieval_miss, lost_before_llm, generation_error, hallucination
""")

    st.markdown("### Tech Stack")
    st.dataframe(pd.DataFrame([
        {"Layer": "API", "Technology": "FastAPI + Uvicorn"},
        {"Layer": "UI", "Technology": "Streamlit + Plotly"},
        {"Layer": "Vector DB", "Technology": "Qdrant (local + cloud)"},
        {"Layer": "Embeddings", "Technology": "Cohere embed-english-v3.0"},
        {"Layer": "Reranker", "Technology": "Cohere rerank-v3.5"},
        {"Layer": "LLM", "Technology": "Groq openai/gpt-oss-20b"},
        {"Layer": "BM25", "Technology": "rank-bm25"},
        {"Layer": "Tracing", "Technology": "SQLite + WAL"},
        {"Layer": "Evaluation", "Technology": "Custom (bootstrap CI, paired tests)"},
    ]), use_container_width=True, hide_index=True)


# ===========================================================================
# The one page
# ===========================================================================

def page_dashboard():
    hero(
        "RAG Evaluation Platform",
        "Understand • Diagnose • Improve your retrieval-augmented generation pipeline",
        badge="🔬 v1.0",
    )
        # Demo mode banner
    st.info(
        "⚠️ **Demo instance** — shared across users. "
        "Feel free to explore. Do not upload sensitive documents."
    )

    # Quick stats at top
    _quick_stats()

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown('<div class="section-title">🏗️ Pipeline Architecture</div>', unsafe_allow_html=True)
    _pipeline_diagram()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Ask section — primary interaction
    _section_ask()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Bottom tabs — all deep-dive features
    st.markdown('<div class="section-title">🔬 Deep Dive</div>', unsafe_allow_html=True)
    tabs = st.tabs([
        "🔍 Retrieval",
        "📊 Evaluation",
        "📋 Traces",
        "📦 Collections",
        "📥 Ingest",
        "ℹ️ About",
    ])
    with tabs[0]:
        _section_retrieval()
    with tabs[1]:
        _section_evaluation()
    with tabs[2]:
        _section_traces()
    with tabs[3]:
        _section_collections()
    with tabs[4]:
        _section_ingest()
    with tabs[5]:
        _section_about()


# ===========================================================================
# Sidebar + main
# ===========================================================================

with st.sidebar:
    st.markdown("""
        <div style="padding: 1rem 0.5rem 1.5rem;">
            <h2 style="color: white; margin: 0; font-size: 1.25rem;">🔬 RAG Platform</h2>
            <p style="color: #9CA3AF; margin: 0.25rem 0 0; font-size: 0.8rem;">
                Evaluation & Debugging
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if api_status():
        st.markdown('<div style="color: #10B981; font-weight: 600;">● API connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #EF4444; font-weight: 600;">● API offline</div>', unsafe_allow_html=True)
        st.caption("Start with `uvicorn app.api:app --port 8000`")

    st.markdown("---")
    st.caption("v1.0 · Made with ❤️")

page_dashboard()