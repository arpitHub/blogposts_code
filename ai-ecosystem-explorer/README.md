# AI Ecosystem Explorer

A browsable directory of AI tools and frameworks across the modern AI stack —
LLM providers, agentic AI, RAG, embeddings, MCP, AI security, observability,
memory, agent SDKs, automation, and vector databases.

Two views share one data layer:

- **Directory** (`/`) — searchable, filterable card grid grouped by category
- **Explore** (`/explore`) — interactive force-directed graph of tools
  connected to their categories, with pan/zoom, node dragging, hover
  highlighting of neighbors, and search highlighting. Tools that live in
  several categories (e.g. Redis, Chroma) are merged into a single node with
  an edge to each category, marked with a dashed ring.

## Stack

- [Next.js](https://nextjs.org) (App Router) + TypeScript
- Tailwind CSS v4
- [Fuse.js](https://fusejs.io) for client-side fuzzy search
- [d3-force](https://github.com/d3/d3-force) for the graph layout (rendered
  as plain SVG — no charting library)
- Fully static — no server-side dependencies, deploys as-is to Vercel

## Getting started

```bash
npm install
npm run dev      # http://localhost:3000
npm run build    # production build (static)
npm run start    # serve the production build
```

## Data

The seed dataset lives at [`data/tools-data.json`](data/tools-data.json) and is
loaded at build time. Shape:

```jsonc
{
  "categories": [
    {
      "id": "llm",
      "name": "LLM Providers",
      "tools": [
        { "id": "...", "name": "...", "description": "...", "website": "...", "logo": "?", "tags": ["?"] }
      ]
    }
  ]
}
```

- `logo` and `tags` are optional; cards fall back to a letter avatar, and tag
  chips render in the modal when present.
- Tools with an empty `website` simply omit the "Visit site" button.
- A few tool names were cut off in the source infographic and are marked
  `(verify name)` in the JSON — the UI surfaces a small note about this so
  they can be corrected rather than silently guessed.

`lib/graph.ts` derives the Explore view's nodes and links from the same JSON
at build time, merging same-named tools across categories into single nodes.

## Deploy

Push to Vercel with defaults — the build is fully static, so no `vercel.json`
or environment configuration is needed:

```bash
npx vercel
```
