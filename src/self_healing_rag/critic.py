"""Critic agent — evaluates generation quality for grounding and hallucination detection.
Uses OpenRouter (OpenAI-compatible API) and respects rate limits."""

import json
import logging
import os
import re

from openai import OpenAI

from .state import Document
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """You are a strict **factuality critic** for a RAG system. Your job is to
determine whether the generated answer is properly grounded in the provided
context documents.

## Context (retrieved documents)
{context}

## Generated Answer
{generation}

## Evaluation criteria
1. **Grounding** — Is every factual claim in the answer supported by the context?
2. **Faithfulness** — Does the answer avoid introducing information not present in the context?
3. **Relevance** — Does the answer actually address the user's question?

Return a JSON object **only** (no markdown fences, no explanation):

{{
  "passed": true or false,
  "score": 0.0 to 1.0,
  "feedback": "Brief explanation of what passed or what specific issues were found."
}}

Be strict. If the answer contains even one claim not supported by the context,
set passed=false and explain which claim is ungrounded.
"""

QUERY_REFORMULATION_PROMPT = """You are a query reformulation assistant for a RAG system. The last
generation attempt was **rejected** by the factuality critic.

## User's original question
{query}

## Critic feedback (what went wrong)
{critic_feedback}

## Retrieved documents that were used
{context}

## Instructions
Rewrite the user's original question to make it more specific, so the
retriever finds better documents to answer it. Focus on the information
gap identified by the critic. Output ONLY the reformulated query — no
explanations, no prefixes.
"""


class Critic:
    """Evaluates a generation's grounding against retrieved documents using OpenRouter."""

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-nano-30b-a3b:free",
        rpm: int = 60,
    ):
        self.model_name = model
        self._client = None
        self._limiter = RateLimiter(rpm=rpm)

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/self-healing-rag",
                    "X-Title": "Self-Healing RAG Pipeline",
                },
            )
        return self._client

    def evaluate(self, query: str, documents: list[Document], generation: str):
        """Run the critique and return (passed: bool, score: float, feedback: str)."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Source: {doc.source}]\n{doc.content}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = CRITIC_PROMPT.format(
            context=context,
            generation=generation,
        )

        with self._limiter:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )

        raw = response.choices[0].message.content.strip()

        # Try to extract JSON from the response (handle markdown fences)
        try:
            cleaned = re.sub(r'^```(?:json)?\s*', '', raw)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Critic returned non-JSON output:\n%s", raw)
            return True, 0.5, "Could not parse critic output — assuming passed."

        passed = bool(result.get("passed", True))
        score = float(result.get("score", 0.5))
        feedback = str(result.get("feedback", "No feedback provided."))

        return passed, score, feedback


class QueryReformulator:
    """Reformulates the query based on critic feedback using OpenRouter."""

    def __init__(
        self,
        model: str = "nvidia/nemotron-3-nano-30b-a3b:free",
        rpm: int = 60,
    ):
        self.model_name = model
        self._client = None
        self._limiter = RateLimiter(rpm=rpm)

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/self-healing-rag",
                    "X-Title": "Self-Healing RAG Pipeline",
                },
            )
        return self._client

    def reformulate(self, query: str, feedback: str, documents: list[Document]) -> str:
        """Produce a better query for the next retrieval round."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[Source: {doc.source}]\n{doc.content}")
        context = "\n\n---\n\n".join(context_parts)

        prompt = QUERY_REFORMULATION_PROMPT.format(
            query=query,
            critic_feedback=feedback,
            context=context,
        )

        with self._limiter:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=256,
            )

        new_query = response.choices[0].message.content.strip()
        logger.info("Reformulated query: %r -> %r", query, new_query)
        return new_query
