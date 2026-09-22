"""
Day 5: end-to-end ingestion for ONE repo (interface-ai-computer-use-project).

Pipeline: walk repo files -> chunk (ingestion/chunking.py) -> embed each
chunk locally with all-MiniLM-L6-v2 -> insert into `documents` + `chunks`
in Supabase.

Deliberately uses the SECRET key, not the publishable key: ingestion is a
backend/admin operation that must write chunks of every scope (contractor
AND senior_engineer), so it needs to bypass RLS by design. This is the
one place in the whole project where using the secret key is correct --
everywhere a user actually reads data (scripts/verify_rls.py, and later
the FastAPI /ask endpoint), the publishable key + that user's own session
is used instead, so RLS applies. Mixing these up would be a real security
bug, so it's called out explicitly here.

Usage:
  pip install sentence-transformers langchain-text-splitters supabase python-dotenv
  python scripts/ingest_repo.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.chunking import chunk_repo, load_access_map

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]  # ingestion only -- see docstring

REPO_NAME = "interface-ai-computer-use-project"
REPO_DIR = Path(__file__).parent.parent / "corpus_repos" / REPO_NAME
ACCESS_MAP_PATH = Path(__file__).parent.parent / "ingestion" / "access_map.json"


def main():
    print(f"Loading access map from {ACCESS_MAP_PATH}")
    access_map = load_access_map(ACCESS_MAP_PATH)

    print(f"Chunking {REPO_DIR} ...")
    chunks = chunk_repo(REPO_DIR, REPO_NAME, access_map)
    print(f"  -> {len(chunks)} chunks produced")

    if not chunks:
        print("No chunks produced -- check REPO_DIR path and try again.")
        return

    print("Loading embedding model (all-MiniLM-L6-v2, first run downloads ~80MB) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to Supabase with the secret key (ingestion bypasses RLS by design) ...")
    client: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

    # Group chunks by (repo, path) so we create one `documents` row per
    # file, then attach all of that file's chunks to it.
    by_path: dict[str, list] = {}
    for c in chunks:
        by_path.setdefault(c.path, []).append(c)

    inserted_documents = 0
    inserted_chunks = 0

    for path, file_chunks in by_path.items():
        scope = file_chunks[0].required_scope

        doc_result = (
            client.table("documents")
            .insert({"repo": REPO_NAME, "path": path, "required_scope": scope})
            .execute()
        )
        document_id = doc_result.data[0]["id"]
        inserted_documents += 1

        texts = [c.content for c in file_chunks]
        embeddings = model.encode(texts).tolist()

        rows = [
            {
                "document_id": document_id,
                "content": c.content,
                "embedding": emb,
                "required_scope": c.required_scope,
            }
            for c, emb in zip(file_chunks, embeddings)
        ]
        client.table("chunks").insert(rows).execute()
        inserted_chunks += len(rows)

        print(f"  {path}  ({scope}, {len(rows)} chunk(s))")

    print()
    print(f"Done. Inserted {inserted_documents} documents, {inserted_chunks} chunks.")


if __name__ == "__main__":
    main()