# ADR 0006: Pin the query embedder to CPU

Status: Accepted
Date: 2026-10-10

## Context

While building the frontend (which sends one question as both roles at once), the
backend repeatedly died on two simultaneous requests to a freshly started server.
Observed: curl "empty reply" (exit 52) within 0.7 s, then no response to /health; once,
a 60 s hang without dying. The macOS crash report showed SIGSEGV (EXC_BAD_ACCESS, null
address) on a worker thread inside PyTorch:

    at::native::mps::MetalShaderLibrary::exec_unary_kernel
      <- copy_cast_kernel_mps <- Tensor.to(device) <- the embedding call

The innermost frame is an insert into a string hash table, which looks like a
kernel-name cache being modified by two threads at once. On Apple Silicon,
sentence-transformers selects the MPS (Metal) backend by default, and FastAPI runs sync
endpoints on a threadpool, so concurrent requests are concurrent threads.

## Decision

Load the embedder with `device="cpu"` (`app/retrieval.py`). A query is one short
sentence, for which MiniLM on CPU is fast, and production hosts (Linux) have no MPS, so
development now matches deployment.

## Evidence

- Before: a 5-run loop (fresh server, two cold parallel requests per run) crashed on
  run 1 and hung on run 2. An earlier crash was observed under `--reload`.
- After: the same loop, 5 of 5 runs, 10 of 10 requests returned 200, no tracebacks.
- Re-running the eval (56 rows): identical hit/phrase results and top-5 files, same
  metrics, 0 leaks.
- New tests: an assertion that the embedder is on CPU, and one that embeds from 8
  threads and compares against sequential results.

## Not established

- MPS thread-unsafety was not isolated with a minimal reproduction. The cause is
  inferred from the stack trace plus the before/after loop results.
- The single 60 s hang was not diagnosed (plausibly the same race).
- `--reload` was suspected first and turned out to be a red herring: crashes also
  occurred without it.
- Chunk embeddings were generated at ingestion on the default device (probably MPS),
  while queries now use CPU. Vectors can differ in low-order digits; the eval shows no
  change in results.

## Consequences

- The Phase 1 and eval tests send requests one at a time, which is why this was missed.
  The concurrency test now covers it.
- The first request after startup takes about 4-5 s (cold start). Warming the model at
  startup would hide that; not done.
