r"""Policy Q&A agent built as a LangGraph state machine.

Flow:
  retrieve -> grade --(not relevant, retries left)--> rewrite -> retrieve
                  \--(relevant)--> generate -> check --(not grounded, 1 retry)--> generate (strict)
                                                    \--> finalize
Finalize refuses instead of returning an answer that failed the groundedness check.
"""
from __future__ import annotations

import argparse
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from llm import CITE_RE, get_llm
from retrieval import Chunk, HybridRetriever, load_chunks

MAX_REWRITES = 1
MAX_REGENERATIONS = 1
REFUSAL = "I can't answer that reliably from the policy documents. Please refer it to a claims officer."


class State(TypedDict, total=False):
    question: str
    query: str
    chunks: list[Chunk]
    relevant: bool
    rewrites: int
    regenerations: int
    answer: str
    grounded: bool
    unsupported: list[str]
    citations: list[str]
    trace: list[str]


def build_agent(retriever: HybridRetriever, llm, k: int = 4):
    def log(state: State, msg: str) -> list[str]:
        return state.get("trace", []) + [msg]

    def retrieve(state: State) -> State:
        query = state.get("query") or state["question"]
        hits = retriever.search(query, k=k)
        return {"query": query, "chunks": [c for c, _ in hits],
                "trace": log(state, f"retrieve '{query}' -> {[c.id for c, _ in hits]}")}

    def grade(state: State) -> State:
        ok = llm.grade(state["question"], state["chunks"])
        return {"relevant": ok, "trace": log(state, f"grade -> relevant={ok}")}

    def rewrite(state: State) -> State:
        new_query = llm.rewrite(state["question"])
        return {"query": new_query, "rewrites": state.get("rewrites", 0) + 1,
                "trace": log(state, f"rewrite -> '{new_query}'")}

    def generate(state: State) -> State:
        strict = state.get("regenerations", 0) > 0
        answer = llm.generate(state["question"], state["chunks"], strict=strict)
        return {"answer": answer, "trace": log(state, f"generate (strict={strict})")}

    def check(state: State) -> State:
        grounded, unsupported = llm.check_grounded(state["answer"], state["chunks"])
        retrieved_ids = {c.id for c in state["chunks"]}
        cited = CITE_RE.findall(state["answer"])
        if not cited or any(cid not in retrieved_ids for cid in cited):
            grounded = False
            unsupported = unsupported + ["missing or invalid citation"]
        out: State = {"grounded": grounded, "unsupported": unsupported,
                      "trace": log(state, f"check -> grounded={grounded}")}
        if not grounded:
            out["regenerations"] = state.get("regenerations", 0) + 1
        return out

    def finalize(state: State) -> State:
        if not state.get("relevant") or not state.get("grounded"):
            return {"answer": REFUSAL, "citations": [], "trace": log(state, "finalize -> refused")}
        cites = sorted(set(CITE_RE.findall(state["answer"])))
        return {"citations": cites, "trace": log(state, "finalize -> answered")}

    def after_grade(state: State) -> str:
        if state["relevant"]:
            return "generate"
        return "rewrite" if state.get("rewrites", 0) < MAX_REWRITES else "finalize"

    def after_check(state: State) -> str:
        if state["grounded"] or state.get("regenerations", 0) > MAX_REGENERATIONS:
            return "finalize"
        return "generate"

    g = StateGraph(State)
    for name, fn in [("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("generate", generate), ("check", check), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite", "finalize"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", after_check, ["generate", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the insurance policy agent a question.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--offline", action="store_true", help="use the extractive LLM, no API key needed")
    parser.add_argument("--trace", action="store_true", help="print each step the agent took")
    args = parser.parse_args()

    retriever = HybridRetriever(load_chunks())
    llm = get_llm(offline=args.offline)
    agent = build_agent(retriever, llm)

    start = time.perf_counter()
    result = agent.invoke({"question": " ".join(args.question)})
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Citations: {', '.join(result['citations']) or 'none'}")
    print(f"LLM: {llm.name} | Retrieval: {retriever.describe()} | {elapsed:.0f} ms")
    if args.trace:
        print("\nTrace:")
        for step in result["trace"]:
            print(f"  - {step}")


if __name__ == "__main__":
    main()
