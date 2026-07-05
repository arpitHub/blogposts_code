# Song Recognition

A small Vite + React app that listens through your microphone, identifies the
song playing, and links straight to it on Spotify.

## Prerequisites

- An [AudD](https://audd.io/) API token (free tier available).

## Setup

1. Install dependencies:

   ```bash
   npm install
   ```

2. Copy `.env.example` to `.env` and add your AudD token:

   ```bash
   cp .env.example .env
   ```

   - `VITE_AUDD_API_TOKEN` — your API token from https://dashboard.audd.io/

3. Run the dev server:

   ```bash
   npm run dev
   ```

   Then open http://localhost:5173.

## How it works

- Tap the mic button to record ~8 seconds of audio from your microphone.
- The recording is sent to the AudD recognition API (`return=spotify`) as a
  multipart upload.
- If a match is found, the app shows the title, artist, album art, and an
  **Open in Spotify** button linking to the track (falling back to AudD's
  universal song link if no direct Spotify match is returned).
- Microphone access requires a secure context (`https://` or `localhost`).

## Notes

- No external UI libraries; plain CSS only.
- The AudD token is used directly from the browser, so keep in mind it will
  be visible in client-side network requests — use a token from a plan you're
  comfortable exposing client-side, or add a small server-side proxy if not.
