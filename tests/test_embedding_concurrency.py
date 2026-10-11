"""
Embedding must be CPU-pinned and safe under concurrent requests.

FastAPI runs sync endpoints on a threadpool, so concurrent /ask requests encode
from several threads at once. On Apple Silicon the default MPS backend segfaulted
in that situation (ADR 0006). If the CPU pin is ever removed, the concurrency test
below can take down the whole pytest process, which is a loud failure by design.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.retrieval import _embedder, embed_query


def test_embedder_is_pinned_to_cpu():
    assert _embedder.device.type == "cpu"


def test_concurrent_embedding_is_safe_and_matches_sequential():
    questions = [f"how does component {i} handle errors and retries?" for i in range(24)]
    sequential = [embed_query(q) for q in questions]
    with ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(embed_query, questions))
    for a, b in zip(sequential, parallel):
        assert len(b) == 384
        assert np.allclose(a, b, atol=1e-5)
