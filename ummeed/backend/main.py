"""FastAPI app entry. Mounts the auth/public routers; cases, moderator,
admin, and matching wiring follow in steps 4-7."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from auth import SCOPE_MODERATOR, create_token, verify_password
from config import settings
from database import get_db, init_db
from models import Moderator
from routers import public

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Ummeed API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Moderator login. Lives here rather than in routers/ to keep the spec's
# auth.py focused on token utilities.
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str


@auth_router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    moderator = (
        db.query(Moderator).filter(Moderator.email == req.email).first()
    )
    if not moderator or not verify_password(
        req.password, moderator.hashed_password
    ):
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = create_token(
        sub=moderator.email,
        scope=SCOPE_MODERATOR,
        ttl_seconds=settings.MODERATOR_TOKEN_TTL_HOURS * 3600,
        extra={"role": moderator.role.value},
    )
    return LoginResponse(
        access_token=token,
        role=moderator.role.value,
        name=moderator.name,
    )


app.include_router(auth_router)
app.include_router(public.router)


# Serve uploaded case + sighting photos. Photos are stored under
# PHOTO_STORAGE_PATH and exposed at /storage/<relative_path>.
_photo_root = Path(settings.PHOTO_STORAGE_PATH)
_photo_root.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(_photo_root)), name="storage")
