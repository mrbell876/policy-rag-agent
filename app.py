"""Streamlit web UI for the policy Q&A agent.

Run with: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from agent import REFUSAL, build_agent
from llm import get_llm
from retrieval import HybridRetriever, load_chunks

ROOT = Path(__file__).parent

st.set_page_config(page_title="Policy RAG Agent", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Building the search index...")
def get_retriever() -> HybridRetriever:
    return HybridRetriever(load_chunks(ROOT / "data"))


@st.cache_resource
def get_agent(offline: bool):
    llm = get_llm(offline=offline)
    return build_agent(get_retriever(), llm), llm.name


@st.cache_data
def example_questions() -> list[str]:
    items = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    in_scope = [i["q"] for i in items if i["expected"]]
    out_of_scope = [i["q"] for i in items if not i["expected"]]
    return [in_scope[1], in_scope[7], in_scope[12], out_of_scope[0]]


def render_result(result: dict, llm_name: str, elapsed_ms: float) -> None:
    if result["answer"] == REFUSAL:
        st.warning(f"**Refused:** {result['answer']}")
    else:
        st.markdown(result["answer"])

    by_id = {c.id: c for c in result.get("chunks", [])}
    for cid in result.get("citations", []):
        chunk = by_id.get(cid)
        if chunk:
            with st.expander(f"📎 {cid} · {chunk.doc} / {chunk.section}"):
                st.markdown(chunk.text)

    with st.expander("🔍 Agent trace"):
        for i, step in enumerate(result.get("trace", []), start=1):
            st.markdown(f"{i}. `{step}`")

    st.caption(f"LLM: {llm_name} · {elapsed_ms:.0f} ms")


# Sidebar
has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "LLM",
        ["Claude", "Offline (no API key)"],
        index=0 if has_key else 1,
        disabled=not has_key,
        help=None if has_key else "Set ANTHROPIC_API_KEY to enable Claude mode.",
    )
    offline = mode != "Claude"
    st.caption(f"Retrieval: {get_retriever().describe()}")

    st.header("How it works")
    st.markdown(
        "1. **Retrieve**: BM25 + dense embeddings, fused with RRF, then reranked\n"
        "2. **Grade** relevance and rewrite the query if retrieval missed\n"
        "3. **Generate** an answer that cites passage IDs\n"
        "4. **Check** every claim against the passages\n"
        "5. **Answer or refuse**: ungrounded answers are never shown"
    )

    st.header("Sample policies")
    st.markdown("- Motor\n- Homeowners\n- Term life / mortgage protection")
    st.caption("All documents are fictional.")

    if st.button("Clear chat"):
        st.session_state.history = []

# Main area
st.title("📄 Policy RAG Agent")
st.markdown(
    "Ask a question about the sample insurance policies. The agent answers with citations, "
    "checks itself for hallucinations, and **refuses** rather than guessing."
)

st.session_state.setdefault("history", [])

cols = st.columns(4)
clicked = None
for col, q in zip(cols, example_questions()):
    if col.button(q, use_container_width=True):
        clicked = q

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_result(turn["result"], turn["llm"], turn["ms"])

question = st.chat_input("e.g. Is flood damage covered?") or clicked
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving, grading, generating, checking..."):
            agent, llm_name = get_agent(offline)
            start = time.perf_counter()
            result = agent.invoke({"question": question})
            elapsed = (time.perf_counter() - start) * 1000
        render_result(result, llm_name, elapsed)
    st.session_state.history.append(
        {"question": question, "result": result, "llm": llm_name, "ms": elapsed}
    )
