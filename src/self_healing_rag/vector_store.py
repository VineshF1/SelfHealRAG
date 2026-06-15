"""Vector store — ChromaDB-backed retrieval with seed documents."""

import logging
import os
import tempfile
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .state import Document

logger = logging.getLogger(__name__)

# ── Seed document corpus (knowledge base for the RAG pipeline) ──────────
SEED_DOCUMENTS = [
    {
        "content": (
            "LangGraph is a library from LangChain for building stateful, "
            "multi-actor applications with LLMs. It extends the LangChain "
            "ecosystem by adding the ability to create cyclical graphs, "
            "persistent state, and human-in-the-loop workflows. LangGraph "
            "is inspired by Pregel and Apache Beam. The core primitive is "
            "the StateGraph, which defines nodes (computation steps) and "
            "edges (control flow between nodes). Conditional edges let you "
            "route between nodes based on the current state."
        ),
        "source": "langgraph_overview.md",
    },
    {
        "content": (
            "Retrieval-Augmented Generation (RAG) is a technique that "
            "combines information retrieval with text generation. In a RAG "
            "pipeline, a retriever first fetches relevant document chunks "
            "from a knowledge base, then a generator (LLM) produces an "
            "answer grounded in those retrieved chunks. RAG reduces "
            "hallucinations and allows the model to access up-to-date or "
            "domain-specific information without fine-tuning."
        ),
        "source": "rag_concepts.md",
    },
    {
        "content": (
            "Hallucination in LLMs refers to the generation of content "
            "that is factually incorrect, ungrounded, or not present in the "
            "provided context. Common causes include: model overconfidence, "
            "ambiguous prompts, and lack of grounding in retrieved documents. "
            "Mitigation strategies include: retrieval-augmented generation, "
            "critique-and-verify loops, prompt engineering, and fine-tuning "
            "on faithful datasets."
        ),
        "source": "hallucination_guide.md",
    },
    {
        "content": (
            "A critic agent (or verifier) is an LLM-based component that "
            "evaluates the quality of a generated response. It checks for: "
            "(1) Grounding — is every claim supported by the retrieved "
            "documents? (2) Relevance — does the answer address the user's "
            "question? (3) Faithfulness — does the answer avoid introducing "
            "external information not present in the context? If the check "
            "fails, the system can trigger a self-correction loop."
        ),
        "source": "critic_agent.md",
    },
    {
        "content": (
            "Query reformulation is a self-correction strategy for RAG "
            "pipelines. When the generated answer fails a critique check, "
            "the system rewrites the user's query to be more specific or "
            "to target information missing from the first retrieval. For "
            "example: 'Tell me about X' may be reformulated to 'What does "
            "the documentation say about X's performance and limitations?' "
            "This helps the retriever find more relevant chunks on the "
            "next attempt."
        ),
        "source": "query_reformulation.md",
    },
    {
        "content": (
            "Self-healing RAG pipelines use a cyclical workflow to "
            "detect and recover from generation errors autonomously. "
            "The typical loop is: Retrieve → Generate → Critique → "
            "Self-Correct. The critique step evaluates factual grounding "
            "and relevance. If the generation fails the critique, the "
            "system either reformulates the query and re-retrieves, or "
            "outputs a safe fallback response like 'I don't have enough "
            "information to answer that question.' This improves "
            "robustness compared to single-pass RAG systems."
        ),
        "source": "self_healing_rag.md",
    },
    {
        "content": (
            "LangGraph conditional edges allow a graph to branch based "
            "on state values. The edge function receives the full state "
            "and returns the name of the next node to route to. This "
            "enables cyclical workflows like the self-healing RAG loop: "
            "if the critique passes → END; if it fails and retries remain "
            "→ reformulate; if it fails and retries exhausted → fallback."
        ),
        "source": "conditional_edges.md",
    },
    {
        "content": (
            "Pydantic and TypedDict are both supported for defining "
            "LangGraph state schemas. TypedDict is lighter and preferred "
            "for simpler graphs. For graphs with complex state transitions, "
            "Pydantic BaseModel or dataclasses provide validation and "
            "serialization. LangGraph's StateGraph supports reducer "
            "functions that control how state updates are merged when "
            "multiple nodes write to the same key."
        ),
        "source": "state_management.md",
    },
    {
        "content": (
            "The critic agent should return a structured evaluation "
            "consisting of: (1) a binary pass/fail decision, (2) a "
            "confidence score between 0 and 1, and (3) specific feedback "
            "explaining why the generation passed or failed. The feedback "
            "is used by the query reformulation step to craft a better "
            "retrieval query. Example feedback: 'The answer claims X but "
            "none of the retrieved documents mention X.'"
        ),
        "source": "critic_output_format.md",
    },
    {
        "content": (
            "Chromadb is an open-source embedding database that stores "
            "vector embeddings alongside metadata. It supports in-memory "
            "(ephemeral) and persistent modes. The default embedding "
            "function uses all-MiniLM-L6-v2 via ONNX Runtime, producing "
            "384-dimensional vectors. Chroma integrates with LangChain "
            "via the langchain_chroma package and supports metadata "
            "filtering, collection management, and distance-based search."
        ),
        "source": "chromadb_guide.md",
    },
]


class VectorStore:
    """Thin ChromaDB wrapper for document retrieval."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "self_healing_rag",
        n_results: int = 4,
    ):
        self.n_results = n_results

        # Use a temp directory if no persist path is given (ephemeral mode)
        if persist_directory is None:
            persist_directory = tempfile.mkdtemp(prefix="chroma_rag_")

        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create the collection (default embedding is all-MiniLM-L6-v2)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
        )

        # Seed if empty
        if self._collection.count() == 0:
            self._seed(SEED_DOCUMENTS)

        logger.info(
            "VectorStore ready at %s — %d documents",
            persist_directory,
            self._collection.count(),
        )

    # ── Public API ──────────────────────────────────────────

    def retrieve(self, query: str, n_results: Optional[int] = None) -> List[Document]:
        """Query the vector store and return *n_results* closest chunks."""
        k = n_results or self.n_results
        results = self._collection.query(query_texts=[query], n_results=k)

        documents: List[Document] = []
        # Chroma returns lists-of-lists; we queried one string so take index 0
        for i, content in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results.get("distances") else 0.0
            documents.append(
                Document(
                    content=content,
                    source=meta.get("source", "unknown"),
                    score=1.0 - dist,  # convert distance to similarity
                )
            )
        return documents

    def add_documents(self, docs: List[dict]) -> None:
        """Ingest new documents into the store. Each dict needs 'content' and
        optionally 'source' (defaults to 'user_upload')."""
        texts = []
        metadatas = []
        ids = []

        for i, doc in enumerate(docs):
            texts.append(doc["content"])
            metadatas.append({"source": doc.get("source", "user_upload")})
            ids.append(f"doc_{self._collection.count() + i}")

        self._collection.add(documents=texts, metadatas=metadatas, ids=ids)
        logger.info("Added %d documents (%d total)", len(docs), self._collection.count())

    def count(self) -> int:
        return self._collection.count()

    # ── Internal helpers ────────────────────────────────────

    def _seed(self, documents: List[dict]) -> None:
        self.add_documents(documents)
