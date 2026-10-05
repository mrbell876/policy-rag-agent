"""LLM backends used by the agent.

ClaudeLLM calls the Anthropic API when ANTHROPIC_API_KEY is set.
OfflineLLM is a deterministic, extractive stand-in so the full graph can run
(and be tested) with no API key.
"""
from __future__ import annotations

import json
import os
import re

from retrieval import Chunk, tokenize

CITE_RE = re.compile(r"\[([a-z_]+#\d+)\]")


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else {}


class ClaudeLLM:
    name = "claude"

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

    def _ask(self, system: str, user: str, max_tokens: int = 600) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def grade(self, question: str, chunks: list[Chunk]) -> bool:
        context = "\n\n".join(c.render() for c in chunks)
        out = self._ask(
            "You judge whether retrieved insurance policy passages contain the information "
            'needed to answer a question. Reply only with JSON: {"relevant": true|false}.',
            f"Question: {question}\n\nPassages:\n{context}",
            max_tokens=50,
        )
        return bool(_parse_json(out).get("relevant", False))

    def rewrite(self, question: str) -> str:
        return self._ask(
            "Rewrite the user's question as a short search query using insurance policy "
            "terminology (excess, deductible, exclusion, grace period and so on). Return only the query.",
            question,
            max_tokens=60,
        ).strip()

    def generate(self, question: str, chunks: list[Chunk], strict: bool = False) -> str:
        context = "\n\n".join(c.render() for c in chunks)
        rule = (
            "Every sentence must be directly supported by a passage. Quote figures exactly."
            if strict
            else "Use only the passages."
        )
        return self._ask(
            "You answer questions about insurance policies for a claims team. "
            f"{rule} Cite the passage id in square brackets after each claim, e.g. [motor_policy#2]. "
            "If the passages do not answer the question, say so.",
            f"Question: {question}\n\nPassages:\n{context}",
        ).strip()

    def check_grounded(self, answer: str, chunks: list[Chunk]) -> tuple[bool, list[str]]:
        context = "\n\n".join(c.render() for c in chunks)
        out = self._ask(
            "You are a strict fact checker. List every claim in the answer that is not fully "
            "supported by the passages (numbers, conditions and scope must match). Reply only "
            'with JSON: {"grounded": true|false, "unsupported": ["..."]}.',
            f"Answer:\n{answer}\n\nPassages:\n{context}",
            max_tokens=300,
        )
        data = _parse_json(out)
        return bool(data.get("grounded", False)), list(data.get("unsupported", []))


class OfflineLLM:
    """Extractive fallback: picks the best-matching sentences and checks lexical support."""

    name = "offline-extractive"

    def grade(self, question: str, chunks: list[Chunk]) -> bool:
        generic = {"policy", "cover", "covered", "insurance", "pay", "paid", "get", "happens"}
        q = set(tokenize(question)) - generic
        best = max((len(q & set(tokenize(c.text + " " + c.section))) for c in chunks), default=0)
        return best >= max(1, len(q) // 3)

    def rewrite(self, question: str) -> str:
        synonyms = {"deductible": "excess", "excess": "deductible", "lapse": "grace period",
                    "cancel": "free look", "hurricane": "catastrophe deductible"}
        extra = [v for k, v in synonyms.items() if k in question.lower()]
        return question + (" " + " ".join(extra) if extra else "")

    def generate(self, question: str, chunks: list[Chunk], strict: bool = False) -> str:
        q = set(tokenize(question))
        scored = []
        for c in chunks[:2]:
            for sent in re.split(r"(?<=[.!?])\s+", c.text):
                overlap = len(q & set(tokenize(sent)))
                if overlap:
                    scored.append((overlap, sent, c.id))
        scored.sort(key=lambda s: -s[0])
        picked = scored[:2] if scored else []
        if not picked:
            return "The policy documents provided do not answer this question."
        return " ".join(f"{s} [{cid}]" for _, s, cid in picked)

    def check_grounded(self, answer: str, chunks: list[Chunk]) -> tuple[bool, list[str]]:
        by_id = {c.id: c.text for c in chunks}
        unsupported = []
        for sent in re.split(r"(?<=\])\s+", answer):
            ids = CITE_RE.findall(sent)
            claim = CITE_RE.sub("", sent).strip()
            if not claim:
                continue
            if not ids or not any(claim in by_id.get(i, "") for i in ids):
                unsupported.append(claim)
        return not unsupported, unsupported


def get_llm(offline: bool = False):
    if not offline and os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeLLM()
    return OfflineLLM()
