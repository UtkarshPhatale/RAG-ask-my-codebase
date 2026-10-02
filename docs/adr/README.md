# Architecture Decision Records

Short records of the decisions in this project that weren't obvious defaults
and are worth being able to explain precisely — not a log of every choice
made.

| ADR | Decision |
|---|---|
| [0001](0001-security-invoker-on-match-chunks.md) | `match_chunks()` is `SECURITY INVOKER`, not `SECURITY DEFINER` — the single decision the entire RLS guarantee depends on. |
| [0002](0002-jwks-es256-over-static-secret.md) | JWT verification uses JWKS/ES256 (Supabase's current signing scheme), not a static HS256 shared secret. |
| [0003](0003-test-retrieval-function-directly.md) | The adversarial suite tests `retrieve_chunks()` directly for the security guarantee; a separate, smaller suite tests the `/ask` HTTP endpoint for correct wiring. Neither replaces the other. |
| [0004](0004-rls-enforcement-vs-retrieval-quality.md) | RLS enforcement (wording-independent, database-enforced) and retrieval quality (wording-dependent, embedding-model-limited) are different claims, tested and documented separately. |