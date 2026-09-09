"""Verify Supabase Auth access tokens.

The API no longer issues its own JWTs. The browser authenticates with Supabase
(email + password / Google OAuth) and sends Supabase's access token as a Bearer
token; this module validates it.

Supabase projects on asymmetric signing keys (the current default) publish their
public keys at ``<SUPABASE_URL>/auth/v1/.well-known/jwks.json`` — we verify
ES256/RS256 against those. Legacy projects still on the shared HS256 secret set
``SUPABASE_JWT_SECRET`` instead.
"""
import functools
import os

import jwt
from jwt import PyJWKClient

_AUDIENCE = "authenticated"


class TokenError(Exception):
    """Raised when a Supabase access token cannot be verified."""


@functools.lru_cache(maxsize=1)
def _jwk_client() -> PyJWKClient:
    url = os.environ.get("SUPABASE_URL")
    if not url:
        raise RuntimeError("SUPABASE_URL must be set to verify Supabase tokens")
    return PyJWKClient(f"{url.rstrip('/')}/auth/v1/.well-known/jwks.json")


def verify_supabase_jwt(token: str) -> dict:
    """Return the verified claims of a Supabase access token, or raise TokenError."""
    try:
        alg = jwt.get_unverified_header(token).get("alg", "")
        if alg == "HS256":
            secret = os.environ.get("SUPABASE_JWT_SECRET")
            if not secret:
                raise TokenError("SUPABASE_JWT_SECRET is required for HS256 tokens")
            key, algorithms = secret, ["HS256"]
        else:
            key = _jwk_client().get_signing_key_from_jwt(token).key
            algorithms = ["ES256", "RS256"]

        return jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience=_AUDIENCE,
            options={"require": ["exp", "sub"]},
        )
    except TokenError:
        raise
    except Exception as exc:  # jwt.* errors, JWKS fetch failures, malformed tokens
        raise TokenError(str(exc)) from exc
