# AI Ecosystem Explorer

A browsable directory of AI tools and frameworks across the modern AI stack —
LLM providers, agentic AI, RAG, embeddings, MCP, AI security, observability,
memory, agent SDKs, automation, and vector databases.

This is **Phase 1** (directory view). Phase 2 will add a graph/node-based
"Explore" view reusing the same data layer.

## Stack

- [Next.js](https://nextjs.org) (App Router) + TypeScript
- Tailwind CSS v4
- [Fuse.js](https://fusejs.io) for client-side fuzzy search
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

`lib/types.ts` also defines a flat `ToolEntry` (tool + category) shape, which
is what the Phase 2 graph view will consume as nodes.

## Deploy

Push to Vercel with defaults — the build is fully static, so no `vercel.json`
or environment configuration is needed:

```bash
npx vercel
```
