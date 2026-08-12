# Walkie-Talkie · Base Camp

A single-page React app that simulates a walkie-talkie experience. Hold the
push-to-talk button to speak, release to send. Speech is transcribed in the
browser, sent to a locally running [Ollama](https://ollama.com/) instance, and
the reply is read aloud — fully client-side, no API keys.

## Stack

- Vite + React 18
- Web Speech API (`webkitSpeechRecognition` / `SpeechRecognition`)
- `speechSynthesis` for playback
- Ollama `qwen3:latest` over the local network

## Setup

1. Install deps:

   ```bash
   npm install
   ```

2. Copy `.env.example` to `.env` and point it at your Ollama host:

   ```bash
   cp .env.example .env
   # then edit .env
   VITE_OLLAMA_URL=http://192.168.1.x:11434
   ```

   You can also change the URL at runtime via the ⚙ icon in the app — it's
   stored in `localStorage`.

3. Run the dev server:

   ```bash
   npm run dev
   ```

   Open the printed URL in **Chrome** or **Safari** (the Web Speech API isn't
   available in Firefox).

## Ollama setup (on the Mac mini)

1. Make sure CORS is open so the browser can reach the API:

   ```bash
   export OLLAMA_ORIGINS='*'
   ollama serve
   ```

2. Pull the model:

   ```bash
   ollama pull qwen3:latest
   ```

3. Find the Mac mini's local IP:

   ```bash
   ipconfig getifaddr en0
   ```

   Use that IP in `VITE_OLLAMA_URL` (e.g. `http://192.168.1.42:11434`).

## How it works

- Hold the round Push-to-Talk button — speech recognition starts.
- Release — the final transcript is sent to `${OLLAMA_URL}/api/chat` with the
  `qwen3:latest` model, full message history, and `think: false` (the system
  prompt also ends with `/no_think` as a belt-and-braces measure).
- Any residual `<think>…</think>` block is stripped from the reply before it's
  shown or spoken.
- The response is read aloud via `speechSynthesis` using a calm English voice
  (Daniel / Google UK English Male / Alex when available).

The assistant plays "Base Camp" — a calm field operator that keeps replies to
1–3 sentences and occasionally drops radio lingo.
