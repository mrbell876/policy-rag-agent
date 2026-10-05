"""Evaluate retrieval quality (with an ablation) and end-to-end agent behaviour.

Retrieval: hit@1, hit@4 and MRR for BM25 only, dense only, and the hybrid pipeline.
Agent: answer rate on answerable questions, correct refusals on out-of-scope
questions, citation accuracy, and latency (p50 / p95).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from agent import REFUSAL, build_agent
from llm import get_llm
from retrieval import HybridRetriever, load_chunks, tokenize


def ranking_metrics(ranked_ids: list[list[str]], expected: list[str]) -> dict:
    hit1 = hit4 = rr = 0.0
    for ids, exp in zip(ranked_ids, expected):
        if exp in ids:
            pos = ids.index(exp) + 1
            rr += 1 / pos
            hit1 += pos == 1
            hit4 += pos <= 4
    n = len(expected)
    return {"hit@1": hit1 / n, "hit@4": hit4 / n, "MRR": rr / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    cases = json.loads(Path("eval/questions.json").read_text())
    answerable = [c for c in cases if c["expected"]]
    retriever = HybridRetriever(load_chunks())
    ids = [c.id for c in retriever.chunks]

    bm25_rank, dense_rank, hybrid_rank = [], [], []
    for c in answerable:
        bm25 = retriever.bm25.get_scores(tokenize(c["q"]))
        dense = retriever.doc_vecs @ retriever.encoder.encode([c["q"]])[0]
        bm25_rank.append([ids[i] for i in np.argsort(-bm25)])
        dense_rank.append([ids[i] for i in np.argsort(-dense)])
        hybrid_rank.append([ch.id for ch, _ in retriever.search(c["q"], k=len(ids))])

    expected = [c["expected"] for c in answerable]
    print(f"Retrieval ({retriever.describe()}), {len(answerable)} answerable questions")
    print(f"{'method':<12}{'hit@1':>8}{'hit@4':>8}{'MRR':>8}")
    for name, ranks in [("BM25", bm25_rank), ("Dense", dense_rank), ("Hybrid", hybrid_rank)]:
        m = ranking_metrics(ranks, expected)
        print(f"{name:<12}{m['hit@1']:>8.2f}{m['hit@4']:>8.2f}{m['MRR']:>8.2f}")

    llm = get_llm(offline=args.offline)
    agent = build_agent(retriever, llm)
    latencies, answered, correct_cite, refused_ok, failures = [], 0, 0, 0, []
    for c in cases:
        t0 = time.perf_counter()
        out = agent.invoke({"question": c["q"]})
        latencies.append((time.perf_counter() - t0) * 1000)
        refused = out["answer"] == REFUSAL
        if c["expected"]:
            if not refused:
                answered += 1
                correct_cite += c["expected"] in out["citations"]
            else:
                failures.append(f"refused answerable: {c['q']}")
        else:
            refused_ok += refused
            if not refused:
                failures.append(f"answered out-of-scope: {c['q']}")

    oos = len(cases) - len(answerable)
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    print(f"\nAgent ({llm.name}), {len(cases)} questions")
    print(f"  answered (of answerable):        {answered}/{len(answerable)}")
    print(f"  correct source cited (answered): {correct_cite}/{answered}")
    print(f"  correct refusals (out of scope): {refused_ok}/{oos}")
    print(f"  latency p50 / p95:               {statistics.median(latencies):.0f} / {p95:.0f} ms")
    if failures:
        print("\nFailures to review:")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
