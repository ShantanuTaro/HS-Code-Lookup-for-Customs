"""Classify merchandise to an HS subheading against retrieved CROSS rulings, behind a confidence gate.

A wrong HS code on an entry is a false statement to CBP under 19 U.S.C. 1592, so an unsupported
answer is worse than no answer. The gate is therefore the product: every result is either an answer
traced to named rulings, or an explicit escalation that says why it was withheld.
"""
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal, Protocol

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from retrieve import Hit, search

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# The one auto-answer threshold. Calibrated by backtest.py against measured accuracy, not chosen by
# taste: raising it trades coverage for precision, and both numbers are printed by the backtest.
ANSWER_CONFIDENCE_THRESHOLD = 0.90
RETRIEVAL_TOP_K = 8
DEFAULT_MODEL = "openai/gpt-oss-120b"
CACHE_FILE = Path(__file__).resolve().parent / "data" / "llm_cache.jsonl"
# The free tier's per-minute token budget is the binding constraint on a 500-case backtest, so a
# 429 is an expected part of a normal run rather than a failure. Waiting is correct; falling back
# to the baseline here would quietly turn a rate limit into a worse accuracy number.
RATE_LIMIT_RETRIES = 6
# Marks a result the retrieval baseline produced. The fallback is deliberate in production but
# must never be invisible: a backtest that silently scores the baseline while reporting the model
# is a measurement of nothing, which is exactly how the first run of this backtest lied.
BASELINE_MARKER = "Retrieval baseline"

SYSTEM_PROMPT = """You are a US customs classification specialist. Classify the merchandise into a \
6-digit HS subheading using ONLY the CBP rulings provided as precedent.

Rules:
- Cite the ruling numbers that actually support your answer. Never cite a ruling that is not listed.
- If the provided rulings do not cover this merchandise, say so and give a low confidence.
- confidence is your probability that the 6-digit subheading is exactly right, from 0.0 to 1.0.

Reply with JSON only: {"hs6": "XXXXXX", "reasoning": "...", "citations": ["N123456"], "confidence": 0.0}"""


class ChatClient(Protocol):
    """The single call classification makes, so a stub or cache can stand in for the provider."""

    def complete_json(self, *, system: str, user: str) -> dict: ...


class Classification(BaseModel):
    """A classification result and the auditable decision about whether it may be returned."""

    hs6: str | None
    reasoning: str
    citations: list[str]
    confidence: float = Field(ge=0, le=1)
    disposition: Literal["answered", "escalated"]
    reason: str
    candidates: list[Hit]

    @property
    def answer(self) -> str | None:
        """The subheading a caller may act on, which is `None` whenever the gate escalated."""
        return self.hs6 if self.disposition == "answered" else None


class GroqChatClient:
    """Groq chat client constrained to JSON replies, with an on-disk cache keyed by the prompt.

    The cache exists because the backtest re-runs the same 500 prompts every time the prompt or the
    threshold changes; without it, each iteration pays for the corpus again.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, cache_file: Path = CACHE_FILE) -> None:
        self.api_key, self.model, self.cache_file = api_key, model, cache_file
        self.cache = {}
        if cache_file.exists():
            for line in cache_file.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry["value"]
                except (ValueError, LookupError):
                    continue

    def complete_json(self, *, system: str, user: str) -> dict:
        """Return one parsed JSON object, from cache when the exact prompt was seen before."""
        key = hashlib.sha256(f"{self.model}\n{system}\n{user}".encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        for attempt in range(RATE_LIMIT_RETRIES):
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
                timeout=90.0,
            )
            if response.status_code != 429:
                break
            time.sleep(float(response.headers.get("retry-after") or 0) or min(2 ** attempt, 30))
        response.raise_for_status()
        payload = json.loads(response.json()["choices"][0]["message"]["content"])
        if not isinstance(payload, dict):
            raise ValueError("The model returned JSON that is not an object.")
        self.cache[key] = payload
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": payload}) + "\n")
        return payload


def default_chat_client() -> ChatClient | None:
    """Resolve the configured model client, or `None` when the retrieval baseline is authoritative."""
    key = os.getenv("GROQ_API_KEY")
    return GroqChatClient(key, os.getenv("GROQ_MODEL") or DEFAULT_MODEL) if key else None


def build_prompt(description: str, hits: list[Hit]) -> str:
    """Render the merchandise and its candidate precedents into the user turn."""
    precedents = "\n\n".join(
        f"[{hit.ruling_number}] HS6 {hit.hs6} ({hit.date})\nSubject: {hit.subject}\n{hit.description[:1200]}" for hit in hits
    )
    return f"MERCHANDISE TO CLASSIFY:\n{description[:4000]}\n\nCANDIDATE CBP RULINGS:\n{precedents}"


def baseline(hits: list[Hit]) -> Classification:
    """Classify by majority vote of the retrieved rulings, used when no model is configured.

    This is the floor the model has to beat. It is also the fallback path, so it is labelled rather
    than silently substituted: `reasoning` names it, and its confidence is the vote share.
    """
    if not hits:
        return Classification(hs6=None, reasoning=f"{BASELINE_MARKER}: no CROSS ruling matched this merchandise.", citations=[], confidence=0.0, disposition="escalated", reason="Retrieval returned no candidate rulings.", candidates=[])
    votes = Counter(hit.hs6 for hit in hits)
    hs6, count = votes.most_common(1)[0]
    confidence = count / len(hits)
    return gate(
        Classification(
            hs6=hs6,
            reasoning=f"{BASELINE_MARKER}: {count} of {len(hits)} nearest CROSS rulings classify comparable merchandise under {hs6}.",
            citations=[hit.ruling_number for hit in hits if hit.hs6 == hs6],
            confidence=confidence,
            disposition="escalated",
            reason="",
            candidates=hits,
        )
    )


def gate(result: Classification) -> Classification:
    """Answer only when confidence clears the threshold and every citation is a retrieved ruling.

    Deterministic and inspectable on purpose: the model proposes, this decides. Citation validity is
    a correctness check rather than a tunable, because a ruling the model invented cannot support a
    filing no matter how confident the model is about it.
    """
    retrieved = {hit.ruling_number for hit in result.candidates}
    invented = [citation for citation in result.citations if citation not in retrieved]
    if result.hs6 is None or not re.fullmatch(r"\d{6}", result.hs6):
        return result.model_copy(update={"disposition": "escalated", "reason": "No valid 6-digit subheading was produced."})
    if invented:
        return result.model_copy(update={"disposition": "escalated", "reason": f"Cited ruling(s) not in the retrieved set: {', '.join(invented)}."})
    if not result.citations:
        return result.model_copy(update={"disposition": "escalated", "reason": "No CBP ruling was cited in support of the classification."})
    if result.confidence < ANSWER_CONFIDENCE_THRESHOLD:
        return result.model_copy(update={"disposition": "escalated", "reason": f"Confidence {result.confidence:.2f} is below the {ANSWER_CONFIDENCE_THRESHOLD:.2f} auto-answer threshold."})
    return result.model_copy(update={"disposition": "answered", "reason": f"Confidence at or above {ANSWER_CONFIDENCE_THRESHOLD:.2f} with support from {len(result.citations)} cited CBP ruling(s)."})


def classify(description: str, *, client: QdrantClient, chat: ChatClient | None = None, top_k: int = RETRIEVAL_TOP_K) -> Classification:
    """Retrieve precedent rulings, classify against them, and apply the gate to the result.

    A model failure degrades to the labelled retrieval baseline rather than to an error, so the
    caller always gets either an answer or a stated reason it was withheld.
    """
    hits = search(description, top_k=top_k, client=client)
    if chat is None or not hits:
        return baseline(hits)
    try:
        payload = chat.complete_json(system=SYSTEM_PROMPT, user=build_prompt(description, hits))
        proposal = Classification(
            hs6=str(payload.get("hs6") or "").replace(".", "")[:6] or None,
            reasoning=str(payload.get("reasoning") or ""),
            citations=[str(item) for item in payload.get("citations") or []],
            confidence=min(max(float(payload.get("confidence") or 0.0), 0.0), 1.0),
            disposition="escalated",
            reason="",
            candidates=hits,
        )
    except (httpx.HTTPError, LookupError, TypeError, ValueError):
        return baseline(hits)
    return gate(proposal)
