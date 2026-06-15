# Test — verify the graph compiles, routes correctly, and handles the full cycle
# without requiring an OpenAI API key (uses deterministic mock components).

import json
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Suppress the LangGraph msgpack warning for the Document dataclass
warnings.filterwarnings("ignore", message=".*Deserializing unregistered type.*")
warnings.filterwarnings("ignore", message=".*msgpack.*checkpoint.*")
import logging
logging.getLogger("langgraph.checkpoint").setLevel(logging.ERROR)

from self_healing_rag.state import GraphState, Document
from self_healing_rag.vector_store import VectorStore
from self_healing_rag.generator import Generator
from self_healing_rag.critic import Critic, QueryReformulator
from self_healing_rag.graph import build_pipeline, router_after_critique, FALLBACK_RESPONSE


# ── Mock components (no API key needed) ────────────────────

class MockGenerator:
    def generate(self, query, documents):
        return f"A mock answer about: {query}"


class MockCritic:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on or set()

    def evaluate(self, query, documents, generation):
        if query in self.fail_on:
            return False, 0.3, "The answer is not grounded in the provided context."
        return True, 0.95, "Answer is well-grounded."


class MockReformulator:
    def reformulate(self, query, feedback, documents):
        return f"REFORMULATED: {query} (addressing: {feedback[:60]})"


def test_graph_compiles():
    """The StateGraph should build and compile without errors."""
    vs = VectorStore()
    g = build_pipeline(vs, MockGenerator(), MockCritic(), MockReformulator())
    assert g is not None
    print("  ✅ Graph compiled successfully")


def test_critique_pass_goes_to_end():
    """When critique passes, router should return END."""
    state = GraphState(query="What is RAG?")
    state.critique_passed = True
    state.retry_count = 0
    state.max_retries = 3

    result = router_after_critique(state)
    assert result == "__end__", f"Expected __end__, got {result}"
    print("  ✅ Critique PASS routes to END")


def test_critique_fail_with_retries_goes_to_reformulate():
    """When critique fails and retries remain, router returns 'reformulate'."""
    state = GraphState(query="What is RAG?")
    state.critique_passed = False
    state.retry_count = 0
    state.max_retries = 3

    result = router_after_critique(state)
    assert result == "reformulate", f"Expected 'reformulate', got {result}"
    print("  ✅ Critique FAIL + retries left → reformulate")


def test_critique_fail_exhausted_goes_to_fallback():
    """When critique fails and no retries left, router returns 'fallback'."""
    state = GraphState(query="What is RAG?")
    state.critique_passed = False
    state.retry_count = 3
    state.max_retries = 3

    result = router_after_critique(state)
    assert result == "fallback", f"Expected 'fallback', got {result}"
    print("  ✅ Critique FAIL + retries exhausted → fallback")


def test_full_success_cycle():
    """Run the full graph with mock components — should produce an answer."""
    vs = VectorStore()

    generator = MockGenerator()
    critic = MockCritic(fail_on=set())  # always passes
    reformulator = MockReformulator()

    app = build_pipeline(vs, generator, critic, reformulator)

    initial = GraphState(query="What is RAG?", current_query="What is RAG?")
    config = {"configurable": {"thread_id": "test-success"}}

    for _ in app.stream(initial.__dict__, config):
        pass

    final = app.get_state(config).values
    answer = final.get("final_response") or final.get("generation", "")

    assert answer, "Expected a non-empty answer"
    assert "RAG" in answer or "mock" in answer.lower(), f"Unexpected answer: {answer}"
    assert final.get("critique_passed") is True
    print(f"  ✅ Full success cycle produced answer: {answer[:80]}...")


def test_full_fail_then_fallback_cycle():
    """When the critic always fails, the pipeline should fall back after N retries."""
    vs = VectorStore()

    generator = MockGenerator()
    # Always fails regardless of query
    critic = MockCritic(fail_on={"*"}) # '*' wildcard — any query fails
    critic.evaluate = lambda q, docs, gen: (False, 0.3, "Not grounded in context.")
    reformulator = MockReformulator()

    app = build_pipeline(vs, generator, critic, reformulator)

    initial = GraphState(query="What is RAG?", current_query="What is RAG?", max_retries=2)
    config = {"configurable": {"thread_id": "test-fallback"}}

    for _ in app.stream(initial.__dict__, config):
        pass

    final = app.get_state(config).values
    response = final.get("final_response") or final.get("generation", "")
    retries = final.get("retry_count", 0)
    critique_passed = final.get("critique_passed")

    assert response == FALLBACK_RESPONSE, f"Expected fallback, got: {response}"
    assert critique_passed is False
    assert retries == 2, f"Expected 2 retries, got {retries}"
    print(f"  ✅ Fallback cycle triggered after {retries} retries: \"{response}\"")


def test_query_reformulation_history():
    """The reformulation history should accumulate across cycles."""
    vs = VectorStore()

    # A critic that fails twice then passes
    class ConditionalCritic:
        def __init__(self):
            self.call_count = 0
        def evaluate(self, query, documents, generation):
            self.call_count += 1
            if self.call_count <= 2:
                return False, 0.3, "Not grounded enough."
            return True, 0.95, "Now grounded."

    generator = MockGenerator()
    critic = ConditionalCritic()
    reformulator = MockReformulator()

    app = build_pipeline(vs, generator, critic, reformulator)

    initial = GraphState(query="What is LangGraph?", current_query="What is LangGraph?", max_retries=5)
    config = {"configurable": {"thread_id": "test-history"}}

    for _ in app.stream(initial.__dict__, config):
        pass

    final = app.get_state(config).values
    history = final.get("reformulation_history", [])
    retries = final.get("retry_count", 0)

    assert len(history) > 0, "Expected reformulation history"
    assert retries == 2, f"Expected 2 retries before passing, got {retries}"
    print(f"  ✅ Reformulation history: {len(history)} reformulation(s), {retries} retries")
    for i, q in enumerate(history, 1):
        print(f"       #{i}: {q}")


def test_vector_store_retrieval():
    """The vector store should return relevant documents."""
    vs = VectorStore()
    docs = vs.retrieve("What is RAG?")
    assert len(docs) > 0, "Expected at least one document"
    assert any("RAG" in d.content or "Retrieval" in d.content for d in docs)
    print(f"  ✅ Vector store returned {len(docs)} documents for 'What is RAG?'")
    for d in docs:
        print(f"       [{d.source}] score={d.score:.3f}")


if __name__ == "__main__":
    print()
    print("  ╭─ Self-Healing RAG Pipeline — Test Suite ──────────────────╮")
    print("  │  (mock components, no API key required)                    │")
    print("  ╰────────────────────────────────────────────────────────────╯")
    print()

    tests = [
        test_graph_compiles,
        test_critique_pass_goes_to_end,
        test_critique_fail_with_retries_goes_to_reformulate,
        test_critique_fail_exhausted_goes_to_fallback,
        test_full_success_cycle,
        test_full_fail_then_fallback_cycle,
        test_query_reformulation_history,
        test_vector_store_retrieval,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print(f"  {'─' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"  {'─' * 60}")
    sys.exit(0 if failed == 0 else 1)
