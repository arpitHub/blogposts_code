# Blog Admin Portal

A small, modern blog admin built as a portfolio showcase. Write posts in a
rich text editor, save them to a local SQLite database via a FastAPI backend,
and stream AI writing suggestions from a locally-running [Ollama](https://ollama.com/)
model (`qwen3:8b` by default).

## Stack

- **Frontend** — React 18, TypeScript, Vite, Tailwind CSS, shadcn-style UI
  primitives, TanStack Query v5, React Router v6, TipTap.
- **Backend** — FastAPI, SQLAlchemy, SQLite, httpx.
- **AI** — Ollama, streaming `qwen3:8b` over HTTP.

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

| Variable       | Default                  | Description                   |
| -------------- | ------------------------ | ----------------------------- |
| `OLLAMA_URL`   | `http://localhost:11434` | Base URL for the Ollama API   |
| `OLLAMA_MODEL` | `qwen3:8b`               | Model used for suggestions    |

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

## Notes

- CORS is restricted to `http://localhost:5173`.
- If Ollama is unreachable, the streamed response contains a single
  `[Failed to reach Ollama at …]` chunk that is appended to the editor —
  easy to spot, easy to delete.
