import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import Base, engine
from .routers import ai, posts

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Blog Admin Portal API", version="0.1.0")

_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173")
allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(posts.router)
app.include_router(ai.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}
