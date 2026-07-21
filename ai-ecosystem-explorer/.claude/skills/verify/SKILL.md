---
name: verify
description: Build, run, and drive the AI Ecosystem Explorer to verify changes end-to-end.
---

# Verifying ai-ecosystem-explorer

Static Next.js app (App Router, Tailwind v4, Fuse.js). No env vars, no DB.

## Build & launch

```bash
cd ai-ecosystem-explorer
npm run build          # must stay fully static (○ routes only)
npm run start &        # serves on http://localhost:3000
```

## Drive (Playwright)

Chromium is preinstalled; launch with `executablePath: "/opt/pw-browsers/chromium"`
(install `playwright-core` in the scratchpad, not in this project).

Flows worth driving:
- Directory renders 11 category sections; tool-card count should match
  `data/tools-data.json` totals. Note: category headers are also `<button class="group ...">`,
  so a loose button selector counts headers + cards.
- Search input filters live and drops empty categories; a typo (e.g. "pincone")
  should still match via fuzzy search; garbage shows the "No tools match" empty state.
- Clicking a card opens the modal (role="dialog"); Esc and backdrop close it.
- A tool with empty `website` (e.g. "Agno (verify name)") must NOT show "Visit site".
- Category header click collapses/expands its grid.
- 390px viewport → 1 grid column; 1440px → 4 columns.

Explore view (`/explore`):
- Expect 99 node groups (11 categories + 88 tools — Redis and Chroma are
  merged across categories), 90 links, 2 dashed shared-tool rings.
- Hover a category → its tools stay lit, everything else dims (opacity < 1).
- Click a tool node → same ToolModal as the directory.
- Search box dims non-matching nodes; search takes precedence over hover
  (regression: pointer resting on a node used to mask search results).
- Zoom buttons, wheel zoom, background pan all change `svg > g`'s transform;
  dragging a node changes that node group's transform while the mouse is down
  (it settles to a new position after release — compare mid-drag, not just
  after, or the check can false-negative).
- Mobile tap needs a Playwright context with `hasTouch: true`.
