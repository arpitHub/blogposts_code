# Court Craft 🎾

**Learn tennis by seeing it work.** An interactive tennis-education web app in the
"explorable explanations" style: every concept ships with a widget you can drag,
scrub, or play with, instead of walls of text.

## Modules

- **Fundamentals** — a living scoreboard that teaches the scoring system by playing
  points; a grip wheel showing hand position per bevel; a clickable court diagram
  with rules and etiquette.
- **Strokes** — phase-by-phase stick-figure explorers (scrub through the swing) for
  serve, forehand, backhand (1H vs 2H side by side), volley, and overhead, each with
  good-form-vs-mistake comparisons. The flagship **Spin & Ball Flight lab** simulates
  real ball physics (drag + Magnus) with sliders for swing path, face angle, and speed.
- **Movement & tactics** — a split-step timing game, recovery-position geometry,
  a drag-yourself-around positioning simulator, and animated patterns of play.
- **Equipment & fitness** — a racket tuner (weight/head/tension trade-offs) and a
  clickable injury-prevention body map.
- **Progression** — a four-stage skill roadmap with persistent milestone checklists,
  and a filterable drills library with animated court patterns.

## Stack

React 19 + Vite, Tailwind CSS v4, React Router. All diagrams are hand-rolled SVG;
animation via `requestAnimationFrame` and SMIL. No other runtime dependencies.

## Develop

```bash
npm install
npm run dev      # local dev server
npm run build    # production build to dist/
npm run preview  # serve the production build
```
