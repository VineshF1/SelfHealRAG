"""LangGraph workflow — the stateful, cyclical RAG pipeline.

Graph structure::

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


# ── Graph node implementations ─────────────────────────────

def node_retrieve(state: GraphState, vector_store: VectorStore) -> dict:
    """Fetch relevant document chunks from the vector store."""
    query = state.current_query or state.query
    docs = vector_store.retrieve(query)
    logger.info("Retrieved %d documents for query: %s", len(docs), query)

    trace_entry = {
        "node": "retrieve",
        "query": query,
        "doc_count": len(docs),
        "sources": [d.source for d in docs],
    }

    return {
        "documents": docs,
        "trace": [trace_entry],
    }


def node_generate(state: GraphState, generator: Generator) -> dict:
    """Draft an answer using the LLM."""
    query = state.current_query or state.query
    generation = generator.generate(query, state.documents)
    logger.info("Generated answer (%d chars)", len(generation))

    trace_entry = {
        "node": "generate",
        "generation_preview": generation[:120],
    }

    return {
        "generation": generation,
        "trace": [trace_entry],
    }


def node_critique(state: GraphState, critic: Critic) -> dict:
    """Run the critic agent."""
    query = state.current_query or state.query
    passed, score, feedback = critic.evaluate(
        query, state.documents, state.generation
    )
    logger.info("Critique result: passed=%s score=%.2f", passed, score)

    trace_entry = {
        "node": "critique",
        "passed": passed,
        "score": score,
        "feedback_preview": feedback[:150],
    }

    return {
        "critique_passed": passed,
        "critique_score": score,
        "critique_feedback": feedback,
        "trace": [trace_entry],
    }


def node_reformulate(state: GraphState, reformulator: QueryReformulator) -> dict:
    """Reformulate the query and increment the retry counter."""
    query = state.current_query or state.query
    feedback = state.critique_feedback or "Answer was not grounded."

    new_query = reformulator.reformulate(query, feedback, state.documents)

    history = list(state.reformulation_history)
    history.append(new_query)

    trace_entry = {
        "node": "reformulate",
        "old_query": query,
        "new_query": new_query,
        "retry_count": state.retry_count + 1,
    }

    return {
        "current_query": new_query,
        "reformulation_history": history,
        "retry_count": state.retry_count + 1,
        "trace": [trace_entry],
    }


def node_fallback(state: GraphState) -> dict:
    """Return the fallback response."""
    logger.warning(
        "Fallback triggered after %d retries for query: %s",
        state.retry_count,
        state.query,
    )

    trace_entry = {
        "node": "fallback",
        "reason": f"Critique failed after {state.retry_count} retries.",
    }

    return {
        "final_response": FALLBACK_RESPONSE,
        "trace": [trace_entry],
    }


# ── Conditional edge routers ───────────────────────────────

def router_after_critique(state: GraphState) -> Literal["reformulate", "fallback", "__end__"]:
    """Decide the next node based on critique outcome.

    - Passed → END (output the generation as is).
    - Failed + retries remaining → reformulate (try again).
    - Failed + retries exhausted → fallback (safe refusal).
    """
    if state.critique_passed:
        return END

    if state.retry_count < state.max_retries:
        return "reformulate"

    return "fallback"


# ── Graph builder ─────────────────────────────────────────

def build_pipeline(
    vector_store: VectorStore,
    generator: Generator,
    critic: Critic,
    reformulator: QueryReformulator,
) -> StateGraph:
    """Assemble and return the compiled LangGraph."""
    builder = StateGraph(GraphState)

    # ── Register nodes ──────────────────────────────────────
    builder.add_node("retrieve", lambda s: node_retrieve(s, vector_store))
    builder.add_node("generate", lambda s: node_generate(s, generator))
    builder.add_node("critique", lambda s: node_critique(s, critic))
    builder.add_node("reformulate", lambda s: node_reformulate(s, reformulator))
    builder.add_node("fallback", lambda s: node_fallback(s))

    # ── Edges ───────────────────────────────────────────────
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "critique")
    builder.add_conditional_edges(
        "critique",
        router_after_critique,
        {
            "reformulate": "reformulate",
            "fallback": "fallback",
            END: END,
        },
    )
    builder.add_edge("reformulate", "retrieve")
    builder.add_edge("fallback", END)

    builder.set_entry_point("retrieve")

    # Compile with in-memory checkpointing (persists state across graph runs)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
