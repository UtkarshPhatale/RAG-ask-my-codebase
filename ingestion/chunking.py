"""
Chunking strategy (Day 5).

Code doesn't chunk like prose. A character-count splitter run over a .py
file will happily cut a function in half, leaving one chunk with a
function's signature and docstring and no body, and another chunk with a
dangling `return` statement and no context. Neither chunk is useful to
retrieve on its own.

LangChain's language-aware splitters solve this by splitting along
syntax-meaningful boundaries (function/class definitions) for supported
languages, and falling back to a plain recursive character splitter for
prose (.md, .txt) and anything without a dedicated language splitter
(.sh, .json, etc).

This module also tags every chunk with its `required_scope`, looked up
from access_map.json (the machine-readable mirror of
docs/access_design.md) -- so a chunk's access boundary is decided at
ingestion time, not guessed later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# Extensions we actually want to ingest as text/code. Everything else
# (images, binaries, lockfiles, etc.) is skipped.
LANGUAGE_BY_EXTENSION = {
    ".py": Language.PYTHON,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".js": Language.JS,
}

PROSE_EXTENSIONS = {".md", ".txt"}

# Everything else we're willing to ingest as plain text (shell scripts,
# config-ish files with real prose value). Kept short and explicit rather
# than "ingest everything" to avoid accidentally pulling in lockfiles,
# .DS_Store, compiled artifacts, etc.
PLAIN_TEXT_EXTENSIONS = {".sh"}

SKIP_DIR_NAMES = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".pytest_cache",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    repo: str
    path: str
    required_scope: str
    content: str


def load_access_map(access_map_path: Path) -> dict:
    with open(access_map_path) as f:
        return json.load(f)


def resolve_scope(repo: str, relative_path: str, access_map: dict) -> str:
    """Look up a file's required_scope: the repo's default, unless the
    path is explicitly listed as a senior_engineer override."""
    repo_config = access_map.get(repo)
    if repo_config is None:
        raise ValueError(
            f"Repo '{repo}' has no entry in access_map.json -- add one "
            f"before ingesting it (see docs/access_design.md)."
        )
    normalized = relative_path.replace("\\", "/")  # Windows safety
    if normalized in repo_config.get("senior_engineer_overrides", []):
        return "senior_engineer"
    return repo_config["default_scope"]


def _splitter_for(extension: str) -> RecursiveCharacterTextSplitter:
    if extension in LANGUAGE_BY_EXTENSION:
        return RecursiveCharacterTextSplitter.from_language(
            language=LANGUAGE_BY_EXTENSION[extension],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    # Prose and plain text both use the generic recursive splitter --
    # it already prefers paragraph/line breaks over mid-sentence cuts.
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def chunk_repo(repo_dir: Path, repo_name: str, access_map: dict) -> list[Chunk]:
    """Walk repo_dir, chunk every ingestible file, and tag each chunk
    with its required_scope. Returns a flat list of Chunk objects ready
    to embed and insert."""
    chunks: list[Chunk] = []

    for path in sorted(repo_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue

        extension = path.suffix.lower()
        ingestible = (
            extension in LANGUAGE_BY_EXTENSION
            or extension in PROSE_EXTENSIONS
            or extension in PLAIN_TEXT_EXTENSIONS
        )
        if not ingestible:
            continue

        relative_path = str(path.relative_to(repo_dir))
        scope = resolve_scope(repo_name, relative_path, access_map)

        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue

        splitter = _splitter_for(extension)
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append(
                    Chunk(
                        repo=repo_name,
                        path=relative_path,
                        required_scope=scope,
                        content=piece,
                    )
                )

    return chunks