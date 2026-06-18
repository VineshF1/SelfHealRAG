"""Self-Healing RAG — State definitions."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    content: str
    source: str
    score: float = 0.0


@dataclass
class GraphState:
    query: str = ""
    current_query: str = ""
    reformulation_history: List[str] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    generation: Optional[str] = None
    critique_passed: Optional[bool] = None
    critique_feedback: Optional[str] = None
    critique_score: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    final_response: Optional[str] = None
    error: Optional[str] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.current_query:
            self.current_query = self.query
