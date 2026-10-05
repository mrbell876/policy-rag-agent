"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion, then reranked.

Dense embeddings use sentence-transformers when installed, otherwise a TF-IDF + SVD
(LSA) fallback so the demo runs anywhere. Reranking uses a cross-encoder when installed,
otherwise the fused order is kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    "a an the of to and or in on for is are be by with what how do does if my i it at as "
    "this that from any can when which who will under after before".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    doc: str
    section: str
    text: str

    def render(self) -> str:
        return f"[{self.id}] {self.doc} / {self.section}\n{self.text}"


def load_chunks(data_dir: str | Path = "data") -> list[Chunk]:
    """Split each markdown policy into one chunk per '## Section' heading."""
    chunks: list[Chunk] = []
    for path in sorted(Path(data_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip()
        for i, part in enumerate(re.split(r"\n## ", raw)[1:], start=1):
            heading, _, body = part.partition("\n")
            chunks.append(Chunk(f"{path.stem}#{i}", title, heading.strip(), body.strip()))
    return chunks


class DenseEncoder:
    def __init__(self, corpus: list[str]):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.name = "all-MiniLM-L6-v2"
        except Exception:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.model = None
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            matrix = self.tfidf.fit_transform(corpus)
            self.svd = TruncatedSVD(n_components=min(64, matrix.shape[0] - 1), random_state=0)
            self.svd.fit(matrix)
            self.name = "tfidf-lsa (fallback)"

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            vecs = self.model.encode(texts, normalize_embeddings=True)
        else:
            vecs = self.svd.transform(self.tfidf.transform(texts))
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        return np.asarray(vecs)


class Reranker:
    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.name = "ms-marco-MiniLM-L-6-v2"
        except Exception:
            self.model = None
            self.name = "none (fused order kept)"

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if self.model is None or not chunks:
            return chunks
        scores = self.model.predict([(query, c.text) for c in chunks])
        return [c for _, c in sorted(zip(scores, chunks), key=lambda p: -p[0])]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], rrf_k: int = 60):
        self.chunks = chunks
        self.rrf_k = rrf_k
        texts = [f"{c.section}. {c.text}" for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.encoder = DenseEncoder(texts)
        self.doc_vecs = self.encoder.encode(texts)
        self.reranker = Reranker()

    def search(self, query: str, k: int = 4, candidates: int = 10) -> list[tuple[Chunk, float]]:
        """Return top-k chunks with their fused RRF score (higher is better)."""
        bm25_scores = self.bm25.get_scores(tokenize(query))
        dense_scores = self.doc_vecs @ self.encoder.encode([query])[0]

        fused: dict[int, float] = {}
        for scores in (bm25_scores, dense_scores):
            for rank, idx in enumerate(np.argsort(-scores)[:candidates]):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ordered = sorted(fused, key=lambda i: -fused[i])
        score_of = {self.chunks[i].id: fused[i] for i in ordered}
        reranked = self.reranker.rerank(query, [self.chunks[i] for i in ordered])
        return [(c, score_of[c.id]) for c in reranked[:k]]

    def describe(self) -> str:
        return f"BM25 + {self.encoder.name}, RRF fusion, reranker: {self.reranker.name}"
