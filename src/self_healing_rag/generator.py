"""LLM-based answer generator using OpenRouter."""
import logging
import os
from openai import OpenAI
from .state import Document
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

GENERATION_PROMPT = """You are a helpful assistant that answers questions based **only** on the provided context documents.

## Context
{context}

## Question
{query}

## Instructions
- Answer concisely using **only** the information from the context above.
- If the context does not contain enough information to answer, say "The provided documents do not contain enough information to answer this question."
- Cite the source of each piece of information you use (source filename).
- Do NOT add external knowledge or make up facts.

## Answer
"""

class Generator:
    def __init__(self, model: str = "nvidia/nemotron-3-nano-30b-a3b:free", rpm: int = 60):
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

    def generate(self, query: str, documents: list[Document]) -> str:
        context_parts = [f"[Source: {doc.source}]\n{doc.content}" for doc in documents]
        context = "\n\n---\n\n".join(context_parts)
        prompt = GENERATION_PROMPT.format(query=query, context=context)

        with self._limiter:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )
        return response.choices[0].message.content.strip()
