# Web-Slinger Watch

A retro-arcade "guy in the chair" mood piece: a stylized city map that logs
simulated sightings of a neighborhood hero as they pop up around town.
Markers spawn, cluster, and fade into a scrolling log while a radar sweep
spins in the corner and alert banners slide through with flavor text.

This is an **original, unaffiliated fan tribute**. It borrows the general
vibe of a retro-pixel hero-sighting tracker concept — original code,
original flavor text, original geometric pixel-art glyphs. It is not
affiliated with, endorsed by, or representing any studio, publisher, or
copyrighted character.

No backend, no accounts — every sighting is generated and simulated
entirely client-side for the current session.

## Setup

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## Stack

- React 18 + Vite
- Tailwind CSS
- Framer Motion for micro-animations
- Self-hosted fonts via `@fontsource` (Press Start 2P, Work Sans, IBM Plex Mono)
- Web Audio API for synthesized ambient blips (no audio files)
