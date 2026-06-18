"""LangGraph workflow — cyclical self-healing RAG pipeline.

    START → retrieve → generate → critique ──passed──→ END
                                        │
                                    failed (retries left) → reformulate → retrieve → …
                                        │
                                    failed (exhausted) → fallback → END
"""
import logging
from typing import Literal
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from .state import Document, GraphState
from .vector_store import VectorStore
from .generator import Generator
from .critic import Critic, QueryReformulator

logger = logging.getLogger(__name__)
FALLBACK_RESPONSE = "I don't have enough information to answer that question."


def node_retrieve(state: GraphState, vector_store: VectorStore) -> dict:
    query = state.current_query or state.query
    docs = vector_store.retrieve(query)
    return {"documents": docs, "trace": [{"node": "retrieve", "query": query, "doc_count": len(docs)}]}


def node_generate(state: GraphState, generator: Generator) -> dict:
    query = state.current_query or state.query
    try:
        generation = generator.generate(query, state.documents)
        return {"generation": generation, "trace": [{"node": "generate", "generation_preview": generation[:120]}]}
    except Exception as exc:
        logger.error("Generate failed: %s", exc)
        return {"generation": "", "error": str(exc), "trace": [{"node": "generate", "error": str(exc)[:100]}]}


def node_critique(state: GraphState, critic: Critic) -> dict:
    query = state.current_query or state.query
    if not state.generation:
        return {"critique_passed": False, "critique_score": 0.0, "critique_feedback": "Generation was empty.",
                "trace": [{"node": "critique", "passed": False, "score": 0.0, "error": "No generation to critique"}]}
    try:
        passed, score, feedback = critic.evaluate(query, state.documents, state.generation)
        return {
            "critique_passed": passed, "critique_score": score, "critique_feedback": feedback,
            "trace": [{"node": "critique", "passed": passed, "score": score, "feedback_preview": feedback[:150]}],
        }
    except Exception as exc:
        logger.error("Critique failed: %s", exc)
        return {"critique_passed": False, "critique_score": 0.0, "critique_feedback": f"Critique error: {exc}",
                "trace": [{"node": "critique", "passed": False, "error": str(exc)[:100]}]}


def node_reformulate(state: GraphState, reformulator: QueryReformulator) -> dict:
    query = state.current_query or state.query
    feedback = state.critique_feedback or "Answer was not grounded."
    try:
        new_query = reformulator.reformulate(query, feedback, state.documents)
    except Exception as exc:
        logger.error("Reformulate failed: %s", exc)
        new_query = query
    history = list(state.reformulation_history) + [new_query]
    return {
        "current_query": new_query, "reformulation_history": history, "retry_count": state.retry_count + 1,
        "trace": [{"node": "reformulate", "old_query": query, "new_query": new_query, "retry_count": state.retry_count + 1}],
    }


def node_fallback(state: GraphState) -> dict:
    return {
        "final_response": FALLBACK_RESPONSE,
        "trace": [{"node": "fallback", "reason": f"Critique failed after {state.retry_count} retries."}],
    }


def router_after_critique(state: GraphState) -> Literal["reformulate", "fallback", "__end__"]:
    if state.critique_passed:
        return END
    if state.retry_count < state.max_retries:
        return "reformulate"
    return "fallback"


def build_pipeline(vector_store: VectorStore, generator: Generator, critic: Critic, reformulator: QueryReformulator) -> StateGraph:
    builder = StateGraph(GraphState)
    builder.add_node("retrieve", lambda s: node_retrieve(s, vector_store))
    builder.add_node("generate", lambda s: node_generate(s, generator))
    builder.add_node("critique", lambda s: node_critique(s, critic))
    builder.add_node("reformulate", lambda s: node_reformulate(s, reformulator))
    builder.add_node("fallback", lambda s: node_fallback(s))
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "critique")
    builder.add_conditional_edges("critique", router_after_critique, {"reformulate": "reformulate", "fallback": "fallback", END: END})
    builder.add_edge("reformulate", "retrieve")
    builder.add_edge("fallback", END)
    builder.set_entry_point("retrieve")
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
