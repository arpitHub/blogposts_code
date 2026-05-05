"""JWT issuance/decoding, password hashing, and FastAPI auth dependencies.

Two distinct token scopes share the same Authorization: Bearer header:

* ``SCOPE_MODERATOR`` — issued by ``POST /api/auth/login``, grants access to
  ``/api/mod/*`` and (with role=admin) ``/api/admin/*``.
* ``SCOPE_SIGHTING``  — issued by ``POST /api/otp/verify``, grants the bearer
  permission to submit a single sighting via ``POST /api/sightings``.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from models import Moderator, ModeratorRole

SCOPE_MODERATOR = "moderator"
SCOPE_SIGHTING = "sighting"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login", auto_error=False
)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(
    *, sub: str, scope: str, ttl_seconds: int, extra: Optional[dict] = None
) -> str:
    payload: dict = {
        "sub": sub,
        "scope": scope,
        "exp": datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def _decode(token: str) -> dict:
    try:
        return jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
        )


def get_current_moderator(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Moderator:
    if not token:
        raise HTTPException(status_code=401, detail="authentication required")
    payload = _decode(token)
    if payload.get("scope") != SCOPE_MODERATOR:
        raise HTTPException(status_code=403, detail="moderator token required")
    moderator = (
        db.query(Moderator).filter(Moderator.email == payload.get("sub")).first()
    )
    if not moderator:
        raise HTTPException(status_code=401, detail="moderator not found")
    return moderator


def require_admin(
    moderator: Moderator = Depends(get_current_moderator),
) -> Moderator:
    if moderator.role != ModeratorRole.admin:
        raise HTTPException(status_code=403, detail="admin only")
    return moderator


def get_otp_session_phone(
    token: Optional[str] = Depends(oauth2_scheme),
) -> str:
    """Return the phone number embedded in a verified OTP session token."""
    if not token:
        raise HTTPException(status_code=401, detail="OTP session required")
    payload = _decode(token)
    if payload.get("scope") != SCOPE_SIGHTING:
        raise HTTPException(status_code=403, detail="OTP session token required")
    return payload["sub"]
