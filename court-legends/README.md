# Court Legends

One full-page, sourced profile per tennis great, each told like a match, set by set, in the look
of the player's era. It starts with **Rod Laver** (1960s grass-court theme) and **Arthur Ashe**
(1970s broadcast theme).

Next.js (App Router) · TypeScript · Tailwind CSS v4 · Framer Motion · `next/font`. Every page is
statically generated.

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # also runs `npm run images` in soft mode first (see Images)
npm run lint       # type-check
```

## How it's organised

```
content/
  research/<slug>.md     sourced fact sheet: the only place page facts may come from
  players/<slug>.ts      the player's page as typed data (journey, records, era, sources…)
  players/index.ts       registry: order of players on the home page
  types.ts               the Player data model
  images.json            portrait credits, written by scripts/fetch-images.mjs
app/
  themes.css             one [data-theme] block per player: palette, fonts, texture
  players/[slug]/        profile page and its OG image
components/              court-line dividers, scoreboard tiles, journey rail, sections
scripts/fetch-images.mjs Wikimedia Commons downloader and licence checker
```

Page sections, in order: Hero (with "On this page" links) → Journey (8–12 "sets", with a sticky
year rail on desktop and, on phones and tablets, a sticky strip that shows the current set and
opens into a jump menu) → Records & stats → Era context → Beyond the court → Legacy, Where sources
differ, Sources and Image credit → Next legend. Each profile also carries schema.org
`ProfilePage`/`Person` JSON-LD built from the same data.

## Adding a player

1. **Research first.** Write `content/research/<slug>.md`: dates, opponents, scores and links.
   Tag each fact ✅ (2+ sources), ⚠️ (one source) or ❗ (sources disagree). Get it reviewed.
2. **Data file.** Copy `content/players/laver.ts` to `content/players/<slug>.ts` and fill it in
   from the approved sheet only. Every chapter, match, record and note has a `sources` array.
   Leave `score` out of a match when no source gives it; the page says so rather than guessing.
   Label records honestly with `status`: `still-stands`, `since-broken`, `first`, `sources-differ`
   or `stat`.
3. **Theme.** Add a `[data-theme="<slug>"]` block to `app/themes.css`. Copy an existing one and
   keep `--ink`, `--muted` and `--accent-ink` at 4.5:1 or better on `--paper`. For new typefaces,
   add them to `lib/fonts.ts` with `preload: false`. Add the name to `ThemeName` in
   `content/types.ts` and the OG colours to `lib/og.tsx`.
4. **Register.** Import the player in `content/players/index.ts`.
5. **Portrait.** Set `portrait.commons` (pinned `files`, a `category`, the theme's `era`), then run
   `npm run images -- <slug>`. Commit the downloaded image and `content/images.json`. If the crop
   cuts off the face, set `portrait.focus` (a CSS `object-position`, e.g. `"40% 15%"`).

## Images

Photos come **only from Wikimedia Commons**, downloaded into `public/images/players/` (never
hotlinked). `scripts/fetch-images.mjs` asks the Commons API for each candidate's licence and keeps
only public domain, CC0, CC BY or CC BY-SA files. Fair-use, NC, ND and unknown licences are
rejected. Photos dated within the theme's era are preferred. The author, licence, licence URL and
source page are written to `content/images.json`, and the page shows them in a credit line. Photos
are displayed cropped and toned, and the credit line says so, as CC BY-SA requires.

If no file qualifies, or Commons can't be reached, the page falls back to a typographic portrait.
The build runs the script in `--soft` mode, so it never fails a deploy.

```bash
npm run images                 # all players (errors if Commons is unreachable)
npm run images -- ashe         # one player
npm run images -- --force      # re-pick even if a portrait is recorded
```

> The first version was built in a sandbox that couldn't reach Wikimedia Commons, so no portraits
> are committed yet. The first Vercel build (or `npm run images` on any machine with internet
> access) fetches them. Check the choices in `content/images.json`, and if you want to keep them,
> commit the files so later builds don't depend on Commons.

## Deploying to Vercel

This folder is one project in the `blogposts_code` monorepo, so Vercel needs its root directory set.

1. In Vercel: **Add New… → Project**, then import `arpitHub/blogposts_code`.
2. **Root Directory:** `court-legends`. Framework preset: **Next.js**, detected automatically
   (`vercel.json` also sets it). Leave the build, install and output settings at their defaults.
3. **Node.js version:** 22.x or later (Project → Settings → General). The image script imports
   the TypeScript data files directly, which needs Node 22.18 or later.
4. No environment variables are required. Canonical URLs and OG images use
   `VERCEL_PROJECT_PRODUCTION_URL`, which Vercel sets automatically. Without it they fall back to
   `https://court-legends.vercel.app`.
5. Deploy. The build log shows `fetch-images` picking a portrait for each player.
6. Optional: add a custom domain under Project → Settings → Domains.

`vercel.json` has the monorepo's `ignoreCommand`, so pushes that don't touch `court-legends/` don't
trigger a rebuild.

Check after deploying:

- `/`, `/players/laver` and `/players/ashe` render. Each portrait is either a credited photo or the
  typographic fallback.
- `/players/laver/opengraph-image` returns a 1200×630 PNG. Paste a profile URL into a social-card
  preview tool to confirm.
- `/sitemap.xml` and `/robots.txt` list the production domain.

## Quality notes

- **Accessibility:** skip link that moves focus to the main content; landmark sections, each labelled by its heading; visible focus
  rings; W/L badges and count-up numbers have screen-reader text; decorative court lines are
  `aria-hidden`; theme text colours meet WCAG AA.
- **Motion:** scroll reveals and count-ups are skipped entirely under `prefers-reduced-motion`,
  and content is never hidden without JavaScript.
- **Readable scores:** set scores in running text never wrap at the dash (`components/Text.tsx`).
- **No layout shift:** images have fixed dimensions in a 4:5 frame, and fonts use `next/font` with
  size-adjusted fallbacks.
- Lighthouse (mobile, local production build): Performance 91–92, Accessibility 100,
  Best Practices 100, SEO 100. axe-core reports no violations at 320px or 1440px, including with the
  phone set menu open.
- **Print:** navigation is hidden and external links print their URLs.
