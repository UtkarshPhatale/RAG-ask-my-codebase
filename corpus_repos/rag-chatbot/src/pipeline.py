"""
Document Q&A Pipeline

Retrieves relevant chunks from the knowledge base (see knowledge_base.py)
and generates an answer, and wires that up into an interactive CLI.
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.knowledge_base import build_knowledge_base
from typing import Callable, List, Dict



# ──────────────────────────────────────────────
# Local LLM (no API key needed)
# ──────────────────────────────────────────────
def get_llm() -> Callable[[str], List[Dict[str, str]]]:
    """Return a callable local LLM using flan-t5-base.

    Downloads ~1GB on first run, then cached.
    Usage:
        llm = get_llm()
        result = llm("What color is the sky?")
        print(result[0]["generated_text"])  # "blue"
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=150)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return generate


# ──────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant for a marketing agency. Use the following context to answer the client's question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Client question: {question}

Answer:"""


# ──────────────────────────────────────────────
# Retrieve relevant chunks and generate an answer
# ──────────────────────────────────────────────
def ask_question(vector_store, llm, question: str) -> dict:
    """Retrieve relevant chunks and generate an answer.

    Looks up the top 3 most relevant chunks from the vector store,
    combines them into a context string, formats PROMPT_TEMPLATE with
    that context and the question, and passes the result to the LLM.

    Args:
        vector_store: FAISS vector store from knowledge_base.py
        llm: Callable from get_llm()
        question: The user's question string

    Returns:
        dict with two keys:
            "answer"  -> str: the generated answer
            "sources" -> list[str]: the chunk texts that were retrieved
    """
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "sources": []}
    docs = vector_store.similarity_search(question, k=3)
    sources = [doc.page_content for doc in docs]
    context = "\n\n".join(sources)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    result = llm(prompt)
    answer = result[0]["generated_text"]

    return {"answer": answer, "sources": sources}

# ──────────────────────────────────────────────
# Interactive CLI loop
# ──────────────────────────────────────────────
def main() -> None:
    """Interactive Q&A loop.

    Builds the knowledge base and loads the LLM, then either answers a
    single question passed via --query, or starts an interactive prompt
    that calls ask_question() for each input until the user types "quit".
    """
    parser = argparse.ArgumentParser(description="Marketing agency Q&A chatbot")
    parser.add_argument("--query", type=str, help="Ask a single question and exit")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found at {data_dir}")
        return

    vector_store = build_knowledge_base(data_dir)
    llm = get_llm()

    if args.query:
        result = ask_question(vector_store, llm, args.query)
        print("\n📄 Sources:")
        for i, source in enumerate(result["sources"], 1):
            preview = source[:100].replace("\n", " ")
            print(f"  {i}. {preview}...")
        print(f"\n💬 Answer: {result['answer']}\n")
        return

    print("Ask me anything about our services! (type 'quit' to exit)\n")

    while True:
        question = input("> ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        result = ask_question(vector_store, llm, question)

        print("\n📄 Sources:")
        for i, source in enumerate(result["sources"], 1):
            preview = source[:100].replace("\n", " ")
            print(f"  {i}. {preview}...")

        print(f"\n💬 Answer: {result['answer']}\n")

if __name__ == "__main__":
    main()