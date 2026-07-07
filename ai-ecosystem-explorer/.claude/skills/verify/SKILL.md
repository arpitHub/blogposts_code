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
