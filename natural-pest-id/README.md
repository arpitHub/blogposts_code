# Natural Pest ID

A small Vite + React app that identifies garden insects and creatures and tells you whether each is a friend (beneficial), foe (pest), or neutral.

Runs entirely in the browser against a **local Ollama** instance — no cloud API, no key.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running locally.
- A multimodal Gemma model pulled, e.g.:

  ```bash
  ollama pull gemma3:4b
  ```

  (If your tag differs, set `VITE_OLLAMA_MODEL` accordingly.)

## Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. (Optional) Copy `.env.example` to `.env` to override defaults:

   ```bash
   cp .env.example .env
   ```

   - `VITE_OLLAMA_URL` — defaults to `http://localhost:11434`
   - `VITE_OLLAMA_MODEL` — defaults to `gemma3:4b`

3. Run the dev server:

   ```bash
   npm run dev
   ```

   Then open http://localhost:5173.

## How it works

- Describe what you saw (and/or upload a photo) and hit **Identify**.
- The app POSTs to `http://localhost:11434/api/chat` with `format: "json"` and a strict JSON system prompt; images are sent as base64 in the message's `images` array.
- Results are shown as a card with name, friend/foe/neutral verdict, summary, what it does, what to do, and confidence.

## Notes

- Ollama only accepts browser requests from `localhost`/`127.0.0.1` by default. If you serve this app from another origin, start Ollama with `OLLAMA_ORIGINS="*" ollama serve` (or a more specific origin).
- Image identification requires a multimodal model. Text-only Gemma variants will fail when an image is attached.
- No external UI libraries; plain CSS only.
