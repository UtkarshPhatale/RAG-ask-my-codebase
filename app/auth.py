from fastapi import Header, HTTPException, status
from jose import jwt, JWTError
import httpx
from supabase import create_client, Client

from app.config import SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY

JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"

_jwks_cache: dict | None = None


def _get_jwks() -> dict:
    global _jwks_cache
    if _jwks_cache is None:
        resp = httpx.get(JWKS_URL, timeout=5.0)
        resp.raise_for_status()
        _jwks_cache = resp.json()
    return _jwks_cache


class AuthedUser:
    def __init__(self, user_id: str, email: str | None, role: str, client: Client):
        self.user_id = user_id
        self.email = email
        self.role = role
        # The Supabase client scoped to THIS user's session (publishable key +
        # their JWT via postgrest.auth()). Every retrieval call in app/retrieval.py
        # must use this client, never a fresh unscoped one and never the
        # secret-key client -- this is what makes RLS apply.
        self.client = client


def _verify_jwt(token: str) -> dict:
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        jwks = _get_jwks()
        matching_key = next((k for k in jwks["keys"] if k["kid"] == kid), None)
        if matching_key is None:
            raise HTTPException(status_code=401, detail="No matching JWKS key found for token")

        payload = jwt.decode(
            token,
            matching_key,
            algorithms=["ES256"],
            audience="authenticated",
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
        )


def get_current_user(authorization: str = Header(...)) -> AuthedUser:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    payload = _verify_jwt(token)

    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing subject claim")

    user_client: Client = create_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)
    user_client.postgrest.auth(token)

    role_row = (
        user_client.table("user_roles")
        .select("role")
        .eq("user_id", user_id)
        .single()
        .execute()
    )

    if not role_row.data:
        raise HTTPException(
            status_code=403,
            detail="Authenticated user has no assigned role in user_roles",
        )

    return AuthedUser(
        user_id=user_id,
        email=email,
        role=role_row.data["role"],
        client=user_client,
    )