# RUN — Self-Healing RAG

## 1. Environment

```bash
cp .env.example .env
```

Edit `.env` and set your `OPENROUTER_API_KEY` and `NVIDIA_API_KEY`.

## 2. Install

```bash
python -m venv .venv
# .venv\Scripts\activate           # Windows
source .venv/bin/activate          # Linux/Mac

pip install -e .
```

## 3. Run

```bash
# Single query
python run.py "What is a self-healing RAG pipeline?"

# Custom model
python run.py "How does the critic work?" -m "nvidia/nemotron-3-nano-30b-a3b:free"

# Max retries
python run.py "What is LangGraph?" --max-retries 2
```

## 4. Run Tests

```bash
python tests/test_pipeline.py
```
