# ADR 0002: Verify JWTs via JWKS/ES256, not a static HS256 shared secret

## Status
Accepted.

## Context

Most Supabase JWT-verification tutorials (and the legacy pattern) assume a
single shared secret (`SUPABASE_JWT_SECRET`) and HS256 (HMAC) signing: the
backend holds the same secret Supabase used to sign the token and verifies
by re-computing the HMAC. This project initially assumed the same thing —
`app/config.py` originally had a `SUPABASE_JWT_SECRET` entry — and the
first verification attempt failed with:

```
The specified alg value is not allowed
```

Investigating this revealed that the Supabase project here uses the newer
"JWT Signing Keys" system: tokens are signed with ES256 (ECDSA, asymmetric),
not HS256. There is no shared secret to hold at all — verification instead
requires fetching Supabase's public signing keys from its JWKS endpoint
(`/auth/v1/.well-known/jwks.json`) and verifying the token's signature
against whichever public key matches the token's `kid` (key ID) header.

## Decision

`app/auth.py` fetches and caches the JWKS document, finds the matching key
by `kid`, and verifies with `algorithms=["ES256"]` and `audience="authenticated"`.
No shared secret is stored anywhere in this project's configuration.

## Why this matters

This isn't a style preference — it's what the actual tokens require, and
it changes the security properties in a way worth being able to explain
precisely:

- **Asymmetric verification doesn't require holding a secret that, if
  leaked, lets someone forge tokens.** The keys this project fetches are
  public by design; nothing in `app/config.py` is sensitive to an
  HS256-style leak.
- **Key rotation is handled for free.** If Supabase rotates its signing
  key, the JWKS endpoint reflects the new key and the `kid`-based lookup
  picks it up automatically — no redeploying a new shared secret.
- **A hardcoded or misremembered `alg` is a known JWT vulnerability
  class** (e.g. accepting `alg: none`, or accepting HS256 when the
  intended algorithm is RS256/ES256, which under some library
  misconfigurations lets an attacker sign their own token using the
  public key as an HMAC secret). This project pins `algorithms=["ES256"]`
  explicitly rather than accepting whatever algorithm the token claims,
  closing that class of bug outright.

## Consequence worth stating explicitly

The JWKS document is cached in a module-level global (`_jwks_cache`) after
first fetch, not re-fetched per request. This is a deliberate tradeoff: it
avoids a network round-trip on every single authenticated request, at the
cost of not picking up a Supabase key rotation until the process restarts.
For a portfolio project at this scale that's an acceptable tradeoff; a
production system would want a TTL or explicit cache-busting on a `kid`
miss rather than an unconditional one-time cache.

## Alternative considered and rejected

Reverting to a static `SUPABASE_JWT_SECRET` + HS256 was briefly considered
as "the simpler path" after the first failure, since most tutorials assume
it. Rejected because it doesn't match what the actual Supabase project
issues — forcing it would have meant working around Supabase's current
auth system rather than with it, for no real benefit.