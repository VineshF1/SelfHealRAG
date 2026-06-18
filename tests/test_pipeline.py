"""Self-Healing RAG tests — mock components, no API key required."""

import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

warnings.filterwarnings("ignore", message=".*Deserializing unregistered type.*")
warnings.filterwarnings("ignore", message=".*msgpack.*checkpoint.*")
import logging
logging.getLogger("langgraph.checkpoint").setLevel(logging.ERROR)

from self_healing_rag.state import GraphState, Document
from self_healing_rag.vector_store import VectorStore
from self_healing_rag.generator import Generator
from self_healing_rag.critic import Critic, QueryReformulator
from self_healing_rag.graph import build_pipeline, router_after_critique, FALLBACK_RESPONSE


class MockGenerator:
    def generate(self, query, documents):
        return f"A mock answer about: {query}"


class MockCritic:
    def __init__(self, always_fail=False):
        self.always_fail = always_fail

    def evaluate(self, query, documents, generation):
        if self.always_fail:
            return False, 0.3, "The answer is not grounded in the provided context."
        return True, 0.95, "Answer is well-grounded."


class MockReformulator:
    def reformulate(self, query, feedback, documents):
        return f"REFORMULATED: {query}"


def test_graph_compiles():
    vs = VectorStore()
    g = build_pipeline(vs, MockGenerator(), MockCritic(), MockReformulator())
    assert g is not None
    print("  ✅ Graph compiled successfully")


def test_critique_pass_goes_to_end():
    state = GraphState(query="What is RAG?")
    state.critique_passed = True
    result = router_after_critique(state)
    assert result == "__end__"
    print("  ✅ Critique PASS routes to END")


def test_critique_fail_with_retries_goes_to_reformulate():
    state = GraphState(query="What is RAG?")
    state.critique_passed = False
    state.retry_count = 0
    state.max_retries = 3
    result = router_after_critique(state)
    assert result == "reformulate"
    print("  ✅ Critique FAIL + retries left → reformulate")


def test_critique_fail_exhausted_goes_to_fallback():
    state = GraphState(query="What is RAG?")
    state.critique_passed = False
    state.retry_count = 3
    state.max_retries = 3
    result = router_after_critique(state)
    assert result == "fallback"
    print("  ✅ Critique FAIL + retries exhausted → fallback")


def test_full_success_cycle():
    vs = VectorStore()
    generator = MockGenerator()
    critic = MockCritic(always_fail=False)
    reformulator = MockReformulator()
    app = build_pipeline(vs, generator, critic, reformulator)

    initial = GraphState(query="What is RAG?", current_query="What is RAG?")
    config = {"configurable": {"thread_id": "test-success"}}

    for _ in app.stream(initial.__dict__, config):
        pass

    final = app.get_state(config).values
    answer = final.get("final_response") or final.get("generation", "")
    assert answer, "Expected a non-empty answer"
    assert "RAG" in answer or "mock" in answer.lower()
    assert final.get("critique_passed") is True
    print(f"  ✅ Full success cycle produced answer: {answer[:80]}...")


def test_full_fail_then_fallback_cycle():
    vs = VectorStore()
    generator = MockGenerator()
    critic = MockCritic(always_fail=True)
    reformulator = MockReformulator()
    app = build_pipeline(vs, generator, critic, reformulator)

    initial = GraphState(query="What is RAG?", current_query="What is RAG?", max_retries=2)
    config = {"configurable": {"thread_id": "test-fallback"}}

    for _ in app.stream(initial.__dict__, config):
        pass

    final = app.get_state(config).values
    response = final.get("final_response") or final.get("generation", "")
    assert response == FALLBACK_RESPONSE, f"Expected fallback, got: {response}"
    assert final.get("critique_passed") is False
    assert final.get("retry_count", 0) == 2
    print(f"  ✅ Fallback cycle triggered after 2 retries")


def test_vector_store_retrieval():
    vs = VectorStore()
    docs = vs.retrieve("What is RAG?")
    assert len(docs) > 0
    assert any("RAG" in d.content or "Retrieval" in d.content for d in docs)
    print(f"  ✅ Vector store returned {len(docs)} documents for 'What is RAG?'")


def test_reformulation_history():
    vs = VectorStore()
    class ConditionalCritic:
        def __init__(self):
            self.call_count = 0
        def evaluate(self, query, documents, generation):
            self.call_count += 1
            return (False, 0.3, "Not grounded.") if self.call_count <= 2 else (True, 0.95, "Now grounded.")

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
    assert len(history) > 0
    assert retries == 2
    print(f"  ✅ Reformulation history: {len(history)} reformulation(s), {retries} retries")


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
        test_vector_store_retrieval,
        test_reformulation_history,
    ]

    passed = failed = 0
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
