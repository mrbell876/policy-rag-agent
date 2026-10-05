# Policy RAG Agent

A small, production-minded RAG agent that answers questions about insurance policy documents for a claims team. It cites its sources, checks its own answers for hallucinations, and refuses when it can't ground an answer instead of guessing.

Built by Javid Bell as a working demo of agentic RAG with LangGraph and Claude.

## What it shows

- **Hybrid retrieval:** BM25 keyword search plus dense embeddings, merged with Reciprocal Rank Fusion, then reranked with a cross-encoder.
- **Agentic workflow (LangGraph):** retrieve, grade relevance, rewrite the query and retry if retrieval missed, generate, check groundedness, regenerate in strict mode if needed, then answer or refuse.
- **Hallucination control:** every answer must cite retrieved passage IDs, and a separate checker verifies each claim against those passages. Answers that fail are never returned.
- **Evaluation:** retrieval ablation (BM25 vs dense vs hybrid: hit@1, hit@4, MRR), answer rate, citation accuracy, correct refusals on out-of-scope questions, and p50/p95 latency.
- **Graceful degradation:** runs fully offline with no API key or model downloads, so the pipeline is testable in CI. Plug in a key and better models without code changes.

## Architecture

```
question
   |
retrieve (BM25 + dense -> RRF -> rerank)
   |
grade relevance --no--> rewrite query --> retrieve   (1 retry)
   | yes
generate answer with [citations]
   |
check groundedness --fail--> regenerate (strict)     (1 retry)
   | pass
finalize: answer with citations, or refuse
```

| File | Purpose |
|---|---|
| `retrieval.py` | Section-level chunking, BM25, dense encoder, RRF fusion, reranker |
| `llm.py` | `ClaudeLLM` (Anthropic API) and `OfflineLLM` (deterministic extractive fallback) |
| `agent.py` | LangGraph state machine and CLI |
| `app.py` | Streamlit web UI: chat, citations with source passages, agent trace |
| `evaluate.py` | Retrieval ablation and end-to-end agent metrics |
| `data/` | Three fictional sample policies: motor, homeowners, term life / mortgage protection |
| `eval/questions.json` | 17 answerable questions with the expected source, plus 3 out-of-scope questions |

## Run it

```bash
pip install -r requirements.txt

# Offline mode: no API key needed
python agent.py --offline --trace "Is there an extra excess for drivers under 25?"
python evaluate.py --offline

# Claude mode
export ANTHROPIC_API_KEY=...
python agent.py --trace "What deductible applies to hurricane damage?"
python evaluate.py
```

### Web UI

```bash
streamlit run app.py
```

Ask questions in a chat interface, expand each citation to see the source passage, and open the agent trace to see every step the graph took. It uses Claude when `ANTHROPIC_API_KEY` is set and offline mode otherwise.

<!-- ![Screenshot](docs/screenshot.png) -->

Set `CLAUDE_MODEL` to choose the model. Install `sentence-transformers` to switch from the TF-IDF/LSA fallback to `all-MiniLM-L6-v2` embeddings and the `ms-marco-MiniLM-L-6-v2` cross-encoder reranker.

## Offline baseline results

```
method         hit@1   hit@4     MRR
BM25            0.71    1.00    0.83
Dense           0.65    1.00    0.80
Hybrid          0.71    1.00    0.83

answered (of answerable):        16/17
correct source cited (answered): 14/16
correct refusals (out of scope): 2/3
latency p50 / p95:               3 / 5 ms
```

The offline baseline uses lexical matching only, so it misses paraphrases (for example "underinsured" vs "average clause") and can be fooled by keyword overlap on out-of-scope questions. Those failures are exactly what the LLM grader, semantic embeddings and reranker are there to fix, and the eval script lists every failure so changes can be measured rather than guessed.

## Taking it to production

- Swap the in-memory index for a vector database (pgvector or Qdrant) and add incremental ingestion.
- Serve the graph behind a FastAPI endpoint with request tracing (LangSmith or OpenTelemetry).
- Log every grade and groundedness verdict, and alert when refusal or failure rates drift.
- Grow the eval set from real user questions and run it in CI on every prompt or model change.

All policy documents in `data/` are fictional and for demonstration only.
