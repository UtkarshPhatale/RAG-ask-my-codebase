"""
Ingestion pipeline: chunk a repo -> embed each chunk locally with
all-MiniLM-L6-v2 -> insert into `documents` + `chunks` in Supabase.

Day 5 built and proved this end-to-end for one repo
(interface-ai-computer-use-project). Day 6 generalizes it to run against
any of the 4 corpus repos via a command-line argument, so the same code
proves the pipeline isn't hardcoded to one repo's structure.

Deliberately uses the SECRET key, not the publishable key: ingestion is a
backend/admin operation that must write chunks of every scope (contractor
AND senior_engineer), so it needs to bypass RLS by design. This is the
one place in the whole project where using the secret key is correct --
everywhere a user actually reads data (scripts/verify_rls.py, and later
the FastAPI /ask endpoint), the publishable key + that user's own session
is used instead, so RLS applies. Mixing these up would be a real security
bug, so it's called out explicitly here.

Usage:
  python scripts/ingest_repo.py interface-ai-computer-use-project
  python scripts/ingest_repo.py nextplay_kanban
  python scripts/ingest_repo.py brain-tumor-segmentation
  python scripts/ingest_repo.py rag-chatbot
"""

import argparse
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

CORPUS_ROOT = Path(__file__).parent.parent / "corpus_repos"
ACCESS_MAP_PATH = Path(__file__).parent.parent / "ingestion" / "access_map.json"


def ingest(repo_name: str, client: Client, model: SentenceTransformer, access_map: dict):
    repo_dir = CORPUS_ROOT / repo_name
    if not repo_dir.is_dir():
        print(f"  SKIP: {repo_dir} does not exist")
        return

    print(f"Chunking {repo_dir} ...")
    chunks = chunk_repo(repo_dir, repo_name, access_map)
    print(f"  -> {len(chunks)} chunks produced")
    if not chunks:
        print("  No chunks produced -- check the repo directory and try again.")
        return

    by_path: dict[str, list] = {}
    for c in chunks:
        by_path.setdefault(c.path, []).append(c)

    inserted_documents = 0
    inserted_chunks = 0

    for path, file_chunks in by_path.items():
        scope = file_chunks[0].required_scope

        doc_result = (
            client.table("documents")
            .insert({"repo": repo_name, "path": path, "required_scope": scope})
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

    print(f"  Done: {inserted_documents} documents, {inserted_chunks} chunks for {repo_name}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest one corpus repo into Supabase.")
    parser.add_argument(
        "repo_name",
        choices=[
            "interface-ai-computer-use-project",
            "nextplay_kanban",
            "brain-tumor-segmentation",
            "rag-chatbot",
        ],
        help="Which corpus_repos/<name> directory to ingest.",
    )
    args = parser.parse_args()

    print(f"Loading access map from {ACCESS_MAP_PATH}")
    access_map = load_access_map(ACCESS_MAP_PATH)

    print("Loading embedding model (all-MiniLM-L6-v2, cached after first run) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to Supabase with the secret key (ingestion bypasses RLS by design) ...\n")
    client: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

    ingest(args.repo_name, client, model, access_map)


if __name__ == "__main__":
    main()