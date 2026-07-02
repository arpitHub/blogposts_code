// Mock vector store for the RAG section. Twelve document chunks live at fixed
// positions in the same kind of 2D "map of meaning" the Embeddings section
// uses. Per-query vector / BM25 / re-rank scores are hand-crafted so the demo
// can honestly show why hybrid retrieval and re-ranking exist: keyword-bait
// documents score high on BM25 but get demoted by the re-ranker.

export interface Doc {
  id: string
  label: string // short label drawn on the map
  text: string // chunk text shown in the assembled prompt
  x: number // position in the embedding map, 0..1
  y: number
}

export const DOCS: Doc[] = [
  { id: 'rayleigh', label: 'Rayleigh', text: 'Rayleigh scattering: air molecules scatter short (blue) wavelengths of sunlight much more strongly than long ones.', x: 0.16, y: 0.22 },
  { id: 'sunset', label: 'sunsets', text: 'At sunset, light crosses more atmosphere, so blue light is scattered away and the remaining reds dominate.', x: 0.26, y: 0.14 },
  { id: 'ozone', label: 'ozone', text: 'The ozone layer absorbs most ultraviolet radiation before it reaches the ground.', x: 0.1, y: 0.38 },
  { id: 'attention', label: 'self-attention', text: 'Transformers process all tokens in parallel and use self-attention to decide which tokens matter to each other.', x: 0.78, y: 0.2 },
  { id: 'bert', label: 'BERT', text: 'BERT is pre-trained with masked language modeling: hide a token, predict it from both directions of context.', x: 0.88, y: 0.32 },
  { id: 'gpus', label: 'GPUs', text: 'GPUs accelerate the huge matrix multiplications inside neural networks.', x: 0.7, y: 0.36 },
  { id: 'naples', label: 'Naples 1889', text: 'The modern pizza margherita was created in Naples in 1889, honoring Queen Margherita with tomato, mozzarella and basil.', x: 0.42, y: 0.78 },
  { id: 'flatbread', label: 'flatbreads', text: 'Topped flatbreads date back to antiquity across the Mediterranean, long before the modern pizza.', x: 0.54, y: 0.86 },
  { id: 'sourdough', label: 'sourdough', text: 'Sourdough bread rises using wild yeast and lactic-acid bacteria cultures.', x: 0.32, y: 0.9 },
  { id: 'bluemangroup', label: 'Blue Man Group', text: 'Blue Man Group is a performance-art company founded in 1987, known for its bald, blue-painted performers.', x: 0.48, y: 0.46 },
  { id: 'skydiving', label: 'skydiving', text: 'Skydiving requires a certified parachute rig and extensive safety training.', x: 0.34, y: 0.54 },
  { id: 'gridtransformer', label: 'power grid', text: 'Electrical transformers step voltage up or down between sections of the power grid.', x: 0.9, y: 0.6 },
]

export interface Query {
  id: string
  text: string
  x: number // where the embedded query lands on the map
  y: number
  vec: Record<string, number> // cosine similarity to each doc (default 0.1)
  bm25: Record<string, number> // keyword-overlap score (default 0.02)
  rerank: Record<string, number> // cross-encoder relevance (default 0.03)
  answer: string // grounded answer, [1][2] cite the assembled chunks
}

export const QUERIES: Query[] = [
  {
    id: 'sky',
    text: 'Why is the sky blue?',
    x: 0.22,
    y: 0.3,
    vec: { rayleigh: 0.92, sunset: 0.84, ozone: 0.61, skydiving: 0.34, bluemangroup: 0.22 },
    bm25: { bluemangroup: 0.85, skydiving: 0.78, rayleigh: 0.55, sunset: 0.3, ozone: 0.22 },
    rerank: { rayleigh: 0.97, sunset: 0.9, ozone: 0.52, skydiving: 0.08, bluemangroup: 0.04 },
    answer:
      'Air molecules scatter short blue wavelengths of sunlight far more than long ones — Rayleigh scattering — so scattered blue light reaches your eyes from every direction [1]. At sunset the path through air is longer, the blue is scattered away, and reds remain [2].',
  },
  {
    id: 'transformers',
    text: 'How do transformers work?',
    x: 0.76,
    y: 0.28,
    vec: { attention: 0.93, bert: 0.8, gpus: 0.62, gridtransformer: 0.31 },
    bm25: { gridtransformer: 0.9, attention: 0.6, bert: 0.35, gpus: 0.2 },
    rerank: { attention: 0.96, bert: 0.85, gpus: 0.55, gridtransformer: 0.06 },
    answer:
      'Transformers read every token at once and use self-attention to decide which tokens matter to each other [1]. Variants like BERT learn language by masking tokens and predicting them from surrounding context [2].',
  },
  {
    id: 'pizza',
    text: 'Who invented pizza?',
    x: 0.46,
    y: 0.7,
    vec: { naples: 0.91, flatbread: 0.82, sourdough: 0.48, bluemangroup: 0.12 },
    bm25: { naples: 0.8, flatbread: 0.45, sourdough: 0.3 },
    rerank: { naples: 0.97, flatbread: 0.88, sourdough: 0.25 },
    answer:
      'The pizza margherita as we know it was created in Naples in 1889 [1] — but no single person invented pizza: topped flatbreads had been eaten around the Mediterranean since antiquity [2].',
  },
]

export interface RankedDoc extends Doc {
  vec: number
  bm25: number
  first: number // first-stage retrieval score
  rerank: number
}

export const TOP_K = 4 // first-stage candidates
export const KEEP = 3 // chunks that make it into the prompt
export const RERANK_FLOOR = 0.2 // re-ranked docs below this relevance are dropped

/** First-stage retrieval: pure vector or hybrid (weighted vector + BM25). */
export function retrieve(query: Query, hybrid: boolean): RankedDoc[] {
  return DOCS.map((d) => {
    const vec = query.vec[d.id] ?? 0.1
    const bm25 = query.bm25[d.id] ?? 0.02
    return {
      ...d,
      vec,
      bm25,
      first: hybrid ? 0.55 * vec + 0.45 * bm25 : vec,
      rerank: query.rerank[d.id] ?? 0.03,
    }
  })
    .sort((a, b) => b.first - a.first)
    .slice(0, TOP_K)
}

/** Second stage: reorder the candidates by cross-encoder relevance. */
export function rerank(candidates: RankedDoc[]): RankedDoc[] {
  return [...candidates].sort((a, b) => b.rerank - a.rerank)
}
