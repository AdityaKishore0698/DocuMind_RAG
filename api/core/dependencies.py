from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core.database import get_db
from core.supabase_auth import TokenError, verify_supabase_jwt
from models.user import User

bearer_scheme = HTTPBearer(auto_error=False)

_credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the local profile row for the caller's Supabase session.

    Verifies the Supabase access token, then upserts a `users` row keyed by the
    Supabase user id (`sub`). All tenant scoping downstream uses this row's
    integer `id`, so the rest of the schema is unchanged.
    """
    if credentials is None or not credentials.credentials:
        raise _credentials_exception

    try:
        claims = verify_supabase_jwt(credentials.credentials)
    except TokenError:
        raise _credentials_exception

    supabase_uid = claims.get("sub")
    if not supabase_uid:
        raise _credentials_exception

    email = claims.get("email")
    user = db.query(User).filter(User.supabase_uid == supabase_uid).first()
    if user is None:
        user = User(
            supabase_uid=supabase_uid,
            email=email,
            username=(email.split("@")[0] if email else None),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    elif email and user.email != email:
        user.email = email
        db.commit()

    return user
