# RAG Chatbot

An interactive Q&A chatbot for a marketing agency, built with LangChain and a
local LLM. It answers questions about services, pricing, and process by
retrieving relevant passages from the agency's own documents and grounding
its answers in them.

No API keys required. No GPU required. Everything runs locally.

---

## What it does

Ask a question, get an answer sourced from the agency's docs:

```
> How much does the Growth package cost?

📄 Sources:
  1. GROWTH PACKAGE — $5,500/month. Best for scaling businesses...

💬 Answer: The Growth package costs $5,500 per month.

> Can I cancel early?

📄 Sources:
  1. Early termination before the minimum commitment requires...

💬 Answer: Yes, with 50% of the remaining contract value.
```

---

## How it works

1. **Load & chunk** — documents in `data/` are loaded and split into
   overlapping chunks (`src/knowledge_base.py`).
2. **Embed & index** — chunks are embedded locally and stored in a FAISS
   vector store.
3. **Retrieve** — a user's question is embedded and matched against the
   vector store to pull the top 3 most relevant chunks.
4. **Generate** — the retrieved chunks are inserted into a prompt template
   along with the question, and passed to a local LLM to produce a
   grounded answer (`src/pipeline.py`).

---

## Stack

| Component    | Library / Tool                          |
| ------------ | ---------------------------------------- |
| Framework    | LangChain (v0.3.x)                      |
| Embeddings   | HuggingFace (`all-MiniLM-L6-v2`, local) |
| Vector Store | FAISS (local)                           |
| LLM          | `google/flan-t5-base` (local, CPU)      |
| Testing      | pytest                                  |

---

## Getting started

1. Clone the repo and `cd` into it
2. Set up the environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run it:

```bash
python -m src.pipeline
```

Or ask a single question and exit:

```bash
python -m src.pipeline --query "What services do you offer?"
```

> First run downloads two models (~1.2GB total: the embedding model and the
> LLM). They're cached locally after that.

4. Run the tests:

```bash
pytest tests/ -v
```

---

## Project structure

```
RAG-chatbot/
├── README.md
├── requirements.txt
├── data/
│   ├── services.txt          ← agency service descriptions
│   ├── pricing.txt           ← packages and pricing
│   ├── faq.txt                ← client FAQ and process
│   ├── product_faq.txt
│   └── company_handbook.txt
├── src/
│   ├── __init__.py
│   ├── knowledge_base.py     ← document loading, chunking, vector store
│   └── pipeline.py           ← retrieval + generation, interactive CLI
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

To point the chatbot at different source material, drop `.txt` files into
`data/` — they're picked up automatically on the next run.

---

## Troubleshooting

**`command not found: python`** — Use `python3`.

**`ModuleNotFoundError`** — Activate the venv and run
`pip install -r requirements.txt`.

**Slow first run** — Models download once (~1.2GB), then are cached.

---

## FAQ

**Do I need an API key?** No — embeddings and generation both run locally.

**What Python version?** 3.10+
