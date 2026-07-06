# Blog Admin Portal

A small, modern blog admin built as a portfolio showcase. Write posts in a
rich text editor, save them to a local SQLite database via a FastAPI backend,
and stream AI writing suggestions from a locally-running [Ollama](https://ollama.com/)
model (`qwen3:8b` by default).

## Stack

- **Frontend** — React 18, TypeScript, Vite, Tailwind CSS, shadcn-style UI
  primitives, TanStack Query v5, React Router v6, TipTap.
- **Backend** — FastAPI, SQLAlchemy, SQLite, httpx.
- **AI** — Ollama (`qwen3:8b`) for local dev, or Anthropic Claude for
  hosted deployments. Both stream responses.

## Project layout

```
blog-admin-portal/
├── backend/         FastAPI app (port 8000)
│   ├── app/
│   │   ├── main.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   └── routers/
│   │       ├── posts.py
│   │       └── ai.py
│   └── requirements.txt
└── frontend/        Vite + React app (port 5173)
    ├── src/
    │   ├── components/
    │   ├── pages/
    │   ├── lib/
    │   └── ...
    └── package.json
```

## Prerequisites

- Python 3.10+
- Node.js 18+ (or 20+)
- Ollama running locally with the `qwen3:8b` model pulled:
  ```sh
  ollama pull qwen3:8b
  ollama serve   # listens on http://localhost:11434
  ```

## Backend setup

```sh
cd backend
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

The first run creates `blog.db` in the backend directory. Visit
<http://localhost:8000/docs> for an interactive API reference.

### Environment variables

| Variable              | Default                       | Description                                                                     |
| --------------------- | ----------------------------- | ------------------------------------------------------------------------------- |
| `LLM_PROVIDER`        | `ollama`                      | `ollama` (local) or `anthropic` (hosted)                                        |
| `OLLAMA_URL`          | `http://localhost:11434`      | Base URL for the Ollama API                                                     |
| `OLLAMA_MODEL`        | `qwen3:8b`                    | Model used for suggestions                                                      |
| `ANTHROPIC_API_KEY`   | —                             | Required when `LLM_PROVIDER=anthropic`                                          |
| `ANTHROPIC_MODEL`     | `claude-haiku-4-5-20251001`   | Anthropic model id                                                              |
| `ANTHROPIC_MAX_TOKENS`| `1024`                        | Max tokens per suggestion                                                       |
| `DATABASE_URL`        | `sqlite:///./blog.db`         | SQLAlchemy URL (any SQLAlchemy-supported DB)                                    |
| `CORS_ORIGINS`        | `http://localhost:5173`       | Comma-separated list of allowed origins                                         |

## Frontend setup

```sh
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api/*` to the backend on port
8000, so no extra configuration is required during development.

## API reference

| Method | Path             | Description                                       |
| ------ | ---------------- | ------------------------------------------------- |
| GET    | `/posts`         | List all posts (most recently updated first)      |
| GET    | `/posts/{id}`    | Fetch a single post                               |
| POST   | `/posts`         | Create a post                                     |
| PUT    | `/posts/{id}`    | Update a post                                     |
| DELETE | `/posts/{id}`    | Delete a post                                     |
| POST   | `/ai/suggest`    | Stream a writing continuation from Ollama         |
| GET    | `/health`        | Liveness probe                                    |

`POST /ai/suggest` accepts `{ "body": "<plain text>" }` and streams
`text/plain` chunks. The frontend appends each chunk to the editor as it
arrives.

## Features

- **Dashboard** — total / published / draft counts plus the five most
  recently updated posts.
- **Posts list** — table with title, status badge, last-updated date, and
  edit / delete actions.
- **Editor** — title, TipTap rich text body (bold, italic, headings,
  lists, quotes, code), comma-separated tags input, draft / published
  toggle, and a "✨ Suggest continuation" button that streams text from
  the local Ollama model into the editor in real time.

## Deployment

The app is split cleanly so the frontend can go on Vercel and the backend
somewhere with a persistent disk. A `vercel.json` (frontend) and
`Dockerfile` + `fly.toml` (backend) are included.

### Frontend on Vercel

1. In Vercel, import the repo and set the **root directory** to
   `blog-admin-portal/frontend`. The framework auto-detects as Vite.
2. Set one environment variable:
   - `VITE_API_BASE_URL` = `https://<your-backend-host>` (e.g. your Fly
     app URL). No trailing slash.
3. Deploy. The included `vercel.json` handles SPA routing.

Locally, leave `VITE_API_BASE_URL` unset — Vite proxies `/api/*` to
`localhost:8000`.

### Backend on Fly.io

Ollama on `localhost` is fine for development, but a hosted backend
can't reach it. Set `LLM_PROVIDER=anthropic` so `/ai/suggest` streams
from Claude instead. Streaming behaviour and the API contract stay
identical.

```sh
cd backend
fly launch --copy-config --no-deploy      # edit generated app name if prompted
fly volumes create blog_data --size 1
fly secrets set \
  ANTHROPIC_API_KEY=sk-ant-... \
  CORS_ORIGINS=https://<your-frontend>.vercel.app
fly deploy
```

The Dockerfile writes `blog.db` under `/data`, which the `[[mounts]]`
block in `fly.toml` binds to the `blog_data` volume — so SQLite data
survives redeploys.

### Other hosts

Any platform that runs a container works: Render, Railway, DigitalOcean
App Platform, a plain VPS with Docker. The two moving parts are the
persistent volume for SQLite and the env vars listed above. On any host
that can reach a self-hosted Ollama (e.g. a Tailscale tunnel), leave
`LLM_PROVIDER=ollama` and set `OLLAMA_URL` to the reachable address.

### Why not Vercel for the backend?

Vercel's Python functions have a per-invocation timeout (10s Hobby, 60s
Pro) and an ephemeral filesystem — neither plays well with SQLite writes
or long LLM streams. Deploying the backend as a container avoids both.

## Notes

- CORS defaults to `http://localhost:5173`. In production set
  `CORS_ORIGINS` to your Vercel URL (comma-separated for multiple).
- If Ollama is unreachable (local dev), the streamed response contains a
  single `[Failed to reach Ollama at …]` chunk that is appended to the
  editor — easy to spot, easy to delete.
