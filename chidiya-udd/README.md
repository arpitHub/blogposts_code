# Chidiya Udd (Bird, Fly!)

A nostalgic, local pass-and-play reflex game for 2-8 players on one shared
screen. The app calls out birds, animals, and objects — players raise a
hand only for the flying ones. Misses are adjudicated by the group by
tapping the offending player's card to dock a life. Last player standing
wins.

Fully offline-capable, installable as a PWA. No backend, no accounts, no
data leaves the device.

## Setup

```bash
npm install
npm run dev
```

Open the printed local URL on a phone or tablet, add 2-8 player names, and
pass the device around.

## Build

```bash
npm run build
npm run preview
```

## Stack

- React 19 + Vite
- Tailwind CSS
- vite-plugin-pwa (installable, offline-ready)
- Web Audio API for call beeps (no audio files)
