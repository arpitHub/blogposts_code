# public/ — PWA icon assets needed

These binary icon files are referenced by `vite.config.js`'s PWA manifest but
are NOT included in this pass (no image-generation tooling available):

- `icon-192.png` — 192x192 PNG
- `icon-512.png` — 512x512 PNG
- `icon-maskable-512.png` — 512x512 PNG, with ~20% safe-zone padding
  (maskable icon per https://web.dev/maskable-icon/)

The dev server and `npm run build` work without them. They only affect the
icon shown after a user installs the PWA to their home screen. Generate them
from `public/favicon.svg` (e.g. via `npx pwa-asset-generator favicon.svg .`
or any image editor) and drop the PNGs directly in this folder.
