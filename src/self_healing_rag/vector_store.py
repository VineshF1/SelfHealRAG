"""Vector store — ChromaDB with NVIDIA NIM embeddings (fallback to local ONNX)."""

import logging
import os
import tempfile
from typing import List, Optional

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from .state import Document

logger = logging.getLogger(__name__)

class NvidiaNIMEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding using NVIDIA NIM API directly."""

    @staticmethod
    def name() -> str:
        return "nvidia_nim"

    def __init__(self):
        api_key = os.getenv("NVIDIA_API_KEY", "")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = os.getenv("EMBED_MODEL", "nvidia/nv-embed-v1")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("NVIDIA embedding — model=%s, base_url=%s", self.model, base_url)
    def __call__(self, input: Documents) -> Embeddings:
        results = []
        for text in input:
            resp = self._client.embeddings.create(model=self.model, input=text)
            results.append(resp.data[0].embedding)
        return results

    # embed_query inherited from EmbeddingFunction(default: calls self.__call__)


def create_embedding_function():
    """Return NvidiaNIMEmbeddingFunction if OPENROUTER_API_KEY is set, else None (ChromaDB default)."""
    key = os.getenv("NVIDIA_API_KEY", "")
    if key and key not in ("***", "your_key_here", "your_o..."):
        logger.info("Using NVIDIA NIM embeddings")
        return NvidiaNIMEmbeddingFunction()
    logger.info("NVIDIA_API_KEY not set or placeholder — using ChromaDB default ONNX embedding")
    return None


SEED_DOCUMENTS = [
    {"content": "LangGraph is a library from LangChain for building stateful, multi-actor applications with LLMs. It extends the LangChain ecosystem by adding the ability to create cyclical graphs, persistent state, and human-in-the-loop workflows.", "source": "langgraph_overview.md"},
    {"content": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. A retriever fetches relevant document chunks, then a generator produces an answer grounded in those chunks.", "source": "rag_concepts.md"},
    {"content": "Hallucination in LLMs refers to the generation of factually incorrect content. RAG and critique-and-verify loops are key mitigation strategies.", "source": "hallucination_guide.md"},
    {"content": "A critic agent (or verifier) evaluates generated responses for grounding, relevance, and faithfulness to the retrieved context.", "source": "critic_agent.md"},
    {"content": "Self-healing RAG systems automatically detect and fix failures: empty retrieval triggers re-retrieval, low-confidence scores trigger fallback, and critic agents validate outputs cyclically.", "source": "self_healing.md"},
    {"content": "Query reformulation rewrites user queries to improve retrieval quality, using techniques like expansion, synonym addition, and decomposition.", "source": "query_reformulation.md"},
    {"content": "ChromaDB is an open-source embedding database that supports NVIDIA NIM embedding functions via the OpenAI-compatible API.", "source": "chromadb_guide.md"},
]


class VectorStore:
    """ChromaDB wrapper with auto-selected embedding function."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "self_healing_store",
    ) -> None:
        if persist_directory is None:
            persist_directory = tempfile.mkdtemp(prefix="chroma_self_heal_")

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        ef = create_embedding_function()
        kwargs: dict = {"name": collection_name}
        if ef is not None:
            kwargs["embedding_function"] = ef
        self._collection = self._client.get_or_create_collection(**kwargs)
        if self._collection.count() == 0:
            self._seed()
        logger.info("VectorStore ready — %d documents", self._collection.count())

    def retrieve(self, query: str, n_results: int = 4) -> list:
        results = self._collection.query(query_texts=[query], n_results=n_results)
        docs = []
        for i, content in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            docs.append(Document(content=content, source=meta.get("source", "unknown"), score=1.0))
        return docs

    def add_documents(self, docs: list) -> None:
        texts = [d["content"] for d in docs]
        ids = [f"doc_{self._collection.count() + i}" for i in range(len(docs))]
        sources = [d.get("source", "unknown") for d in docs]
        self._collection.add(documents=texts, ids=ids, metadatas=[{"source": s} for s in sources])

    def count(self) -> int:
        return self._collection.count()

    def _seed(self) -> None:
        self.add_documents(SEED_DOCUMENTS)
