# How LLMs Actually Work

A single-page interactive web app that visually explains core generative-AI
concepts — tokens, embeddings, attention, the transformer stack, and
next-token prediction — to a mixed audience.

Every section has a persistent **"Explain like I'm… Beginner / Technical"**
toggle in the header: both modes share the same visuals, only the language and
labels change. Everything runs client-side on precomputed/mocked data; there
is no backend and no model call.

## Sections

1. **Tokens** — type a sentence, watch it split into colored word-piece chips
   (technical mode adds token IDs and vocab-size context).
2. **Embeddings** — an interactive 2D map where similar words cluster; click
   two words to see their vectors and cosine similarity.
3. **Attention** — click any token in a sample sentence to see glowing arcs to
   the tokens it attends to; technical mode adds the Q/K/V computation and the
   full attention heatmap.
4. **Transformer stack** — send a token up through stacked blocks and watch
   its hidden-state vector transform layer by layer (residual/LayerNorm
   callouts in technical mode).
5. **Generation** — a live probability bar chart over candidate next tokens
   with a temperature slider; roll the weighted die to sample and extend the
   prompt.

## Stack

React 18 · Vite · TypeScript · Tailwind CSS 4 · Framer Motion · Recharts

## Run it

```sh
npm install
npm run dev     # dev server
npm run build   # type-check + production build
```
