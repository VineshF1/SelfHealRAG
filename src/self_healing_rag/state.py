"""State definition for the Self-Healing RAG pipeline."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """A single retrieved document chunk."""
    content: str
    source: str
    score: float = 0.0


@dataclass
class GraphState:
    """
    State passed between nodes in the cyclical RAG graph.

    Each field carries information across the retrieve → generate → critique
    → (reformulate | fallback | end) loop.
    """
    # ── Query ───────────────────────────────────────────────
    query: str                             # Original user question (never changes)
    current_query: str = ""                # May be reformulated after a rejection
    reformulation_history: List[str] = field(default_factory=list)

    # ── Retrieval ───────────────────────────────────────────
    documents: List[Document] = field(default_factory=list)

    # ── Generation ──────────────────────────────────────────
    generation: Optional[str] = None       # Draft answer from the LLM

    # ── Critique ────────────────────────────────────────────
    critique_passed: Optional[bool] = None  # True = grounded, False = hallucinated
    critique_feedback: Optional[str] = None # Explanation from the critic
    critique_score: Optional[float] = None  # 0.0 – 1.0 confidence

    # ── Loop control ────────────────────────────────────────
    retry_count: int = 0
    max_retries: int = 3

    # ── Output ──────────────────────────────────────────────
    final_response: Optional[str] = None
    error: Optional[str] = None

    # ── Debug / metadata ────────────────────────────────────
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure current_query starts as a copy of query if not already set."""
        if not self.current_query:
            self.current_query = self.query
