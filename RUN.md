# Run Guide

Create `.env` in the project folder with your OpenRouter key:

cp .env.example .env

```ini
OPENROUTER_API_KEY=***
```

Then create a venv and activate it:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install langgraph chromadb openai python-dotenv
```

Run:

```bash
python run.py "What is a self-healing RAG pipeline?"
```
