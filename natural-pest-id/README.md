# Natural Pest ID

A small Vite + React app that identifies garden insects and creatures and tells you whether each is a friend (beneficial), foe (pest), or neutral.

## Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Copy `.env.example` to `.env` and add your Anthropic API key:

   ```bash
   cp .env.example .env
   # then edit .env and set VITE_ANTHROPIC_API_KEY
   ```

3. Run the dev server:

   ```bash
   npm run dev
   ```

## How it works

- Describe what you saw (and/or upload a photo) and hit **Identify**.
- The app calls the Anthropic API (`claude-sonnet-4-20250514`) directly from the browser with a strict JSON system prompt.
- Results are shown as a card with name, friend/foe verdict, summary, what it does, what to do, and confidence.

## Notes

- Purely client-side — your API key is exposed to the browser. Do not deploy this with a real key without a backend proxy.
- No external UI libraries; plain CSS only.
