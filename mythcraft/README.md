# Mythcraft

Ten familiar claims, taken apart one at a time. The hub lists every myth
across science, history, space, psychology, health, tech, and money. Each
one opens a scrollytelling deep-dive: the hook, your guess (myth or fact),
where the idea came from, three pieces of evidence, and a verdict that
reacts to what you guessed.

Your guesses are saved locally, so the verdict still knows what you said
after a reload.

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

- Vite + React 18 + Tailwind CSS
- React Router — `/` (hub) and `/myth/:slug` (deep-dive)
- Zustand, persisted to `localStorage` under `mythcraft-progress`
- `vite-plugin-pwa` for installability
- lucide-react for icons
- Self-hosted fonts via `@fontsource`: Fraunces (display), Work Sans (body),
  IBM Plex Mono (numbers only)

## Adding a myth

Everything lives in `src/data/myths.js`. One object per myth:

```js
{
  slug, category, claim, hook,
  origin: [para1, para2],
  evidence: [{ label, title, body, stat, statLabel }, ...3 items],
  verdict: 'busted' | 'confirmed' | 'mixed',
  correctReaction, incorrectReaction, closing,
}
```

The route, the hub card, the category filter, and the related-myths footer
all read from that list — no other file needs touching. A `category` that
isn't already in `CATEGORIES` needs adding there to get its own filter chip.

`verdict` drives both the stamp's word and its color: `busted` → "Busted"
in red, `confirmed` → "Confirmed" in green, `mixed` → "It's Complicated"
in orange. Guessing "fact" is the correct call only on `confirmed`.

## Notes

- PWA icons in `public/icons/` are placeholders.
- Deploys to Vercel with this folder as the root directory; `vercel.json`
  carries the SPA rewrite so `/myth/:slug` survives a refresh.
