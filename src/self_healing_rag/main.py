"""Self-Healing RAG — CLI entry point (cyclical LangGraph pipeline)."""
import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path
from dotenv import load_dotenv

# Load from root .env (parent of the system dir)
dotenv_path = Path(__file__).resolve().parents[3] / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    load_dotenv()

from .state import GraphState
from .vector_store import VectorStore
from .generator import Generator
from .critic import Critic, QueryReformulator
from .graph import build_pipeline, FALLBACK_RESPONSE

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(name)-28s | %(message)s")
for noisy in ("chromadb", "httpx", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("self-healing-rag")


def pretty_print_trace(trace: list) -> None:
    for entry in trace:
        node = entry.get("node", "?")
        if node == "retrieve":
            print(f"  │  ├── 📄 Retrieve ({entry['doc_count']} docs, query: {entry['query']})")
        elif node == "generate":
            print(f"  │  ├── ✏️  Generate ({len(entry.get('generation_preview', ''))} chars)")
        elif node == "critique":
            status = "✅ PASS" if entry.get("passed") else "❌ FAIL"
            print(f"  │  ├── 🔍 Critique: {status} (score={entry.get('score', 0):.2f})")
            fb = entry.get("feedback_preview", "")
            if fb:
                for line in textwrap.wrap(fb, width=70):
                    print(f"  │  │     {line}")
        elif node == "reformulate":
            print(f"  │  ├── 🔄 Reformulate (retry #{entry.get('retry_count', '?')})")
            print(f"  │  │     Old: {entry.get('old_query', '')}")
            print(f"  │  │     New: {entry.get('new_query', '')}")
        elif node == "fallback":
            print(f"  │  └── 🛑 Fallback (reason: {entry.get('reason', '')})")


def run_pipeline(query: str, max_retries: int = 3, model: str = "nvidia/nemotron-3-nano-30b-a3b:free", rpm: int = 60) -> str:
    logger.info("Initialising pipeline (model=%s, max_retries=%s)", model, max_retries)
    vector_store = VectorStore()
    generator = Generator(model=model, rpm=rpm)
    critic = Critic(model=model, rpm=rpm)
    reformulator = QueryReformulator(model=model, rpm=rpm)
    app = build_pipeline(vector_store, generator, critic, reformulator)

    initial_state = GraphState(query=query, current_query=query, max_retries=max_retries)
    config = {"configurable": {"thread_id": "self-healing-rag-session-1"}}

    print()
    print(f"  ╭─{'─' * 60}─╮")
    print(f"  │  Self-Healing RAG Pipeline{' ' * 33}│")
    print(f"  │  Query: {query[:68]}{' ' * (57 - min(len(query), 68))}│")
    print(f"  ╰─{'─' * 60}─╯")
    print()

    output_state = None
    try:
        for _ in app.stream(initial_state.__dict__, config):
            pass
        output_state = app.get_state(config).values
        final = output_state.get("final_response") or output_state.get("generation", "")
    except Exception:
        logger.exception("Pipeline crashed")
        final = "An error occurred while processing your query. Please try again."
        output_state = {"trace": [{"node": "error", "message": str(sys.exc_info()[1])}]}

    trace = output_state.get("trace", [])
    print("  ┌─ Execution Trace ──────────────────────────────────────────┐")
    pretty_print_trace(trace)
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    verdict = output_state.get("critique_passed")
    score = output_state.get("critique_score", 0)
    retries = output_state.get("retry_count", 0)
    if verdict is True:
        status = f"✅ Critic PASSED (score={score:.2f}, took {retries} reformulation(s))"
    elif verdict is False:
        status = f"🛑 Critic FAILED after {retries} retries — used fallback"
    else:
        status = "⚠️  Pipeline completed (unknown critic state)"
    print(f"  Status: {status}")
    print()
    print(f"  {'─' * 64}")
    print("  ANSWER")
    print(f"  {'─' * 64}")
    print()
    for line in final.split("\n"):
        print(f"  {line}")
    print()
    print(f"  {'─' * 64}")
    print()
    return final


def main():
    parser = argparse.ArgumentParser(description="Self-Healing RAG — LangGraph cyclical RAG with critique-and-correct loop.")
    parser.add_argument("query", nargs="?", help="Question to answer.")
    parser.add_argument("-f", "--file", help="Read question from a text file.")
    parser.add_argument("-m", "--model", default=os.getenv("RAG_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free"))
    parser.add_argument("-r", "--max-retries", type=int, default=3, help="Max retries before fallback.")
    parser.add_argument("--rpm", type=int, default=60)
    parser.add_argument("--list-docs", action="store_true", help="List documents in vector store.")
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set.")
        sys.exit(1)

    if args.list_docs:
        vs = VectorStore()
        print(f"Vector store has {vs.count()} documents.")
        sys.exit(0)

    query = args.query
    if args.file:
        with open(args.file) as f:
            query = f.read().strip()
    if not query:
        parser.print_help()
        print("\nError: Provide a query or use -f.")
        sys.exit(1)

    run_pipeline(query, max_retries=args.max_retries, model=args.model, rpm=args.rpm)


if __name__ == "__main__":
    main()
