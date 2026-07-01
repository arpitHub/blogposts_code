// Mocked next-token prediction. A small lookup table covers fun prompts;
// anything else falls back to deterministic pseudo-random logits seeded by
// the last word, so the demo always produces a stable, plausible-looking
// distribution without a real model.

import { tokenId } from './tokenize'

export interface Candidate {
  token: string
  logit: number
}

const TABLE: Record<string, Candidate[]> = {
  the: [
    { token: 'mat', logit: 3.4 }, { token: 'floor', logit: 3.0 },
    { token: 'chair', logit: 2.7 }, { token: 'roof', logit: 2.2 },
    { token: 'table', logit: 2.0 }, { token: 'moon', logit: 1.1 },
    { token: 'keyboard', logit: 0.8 }, { token: 'idea', logit: 0.2 },
  ],
  cat: [
    { token: 'sat', logit: 3.2 }, { token: 'jumped', logit: 2.6 },
    { token: 'slept', logit: 2.4 }, { token: 'ran', logit: 2.1 },
    { token: 'purred', logit: 1.9 }, { token: 'is', logit: 1.4 },
    { token: 'meowed', logit: 1.2 }, { token: 'computed', logit: -0.5 },
  ],
  on: [
    { token: 'the', logit: 3.8 }, { token: 'a', logit: 2.9 },
    { token: 'my', logit: 2.2 }, { token: 'top', logit: 1.8 },
    { token: 'its', logit: 1.3 }, { token: 'fire', logit: 0.6 },
    { token: 'purpose', logit: 0.4 }, { token: 'Mars', logit: -0.2 },
  ],
  is: [
    { token: 'a', logit: 3.3 }, { token: 'the', logit: 2.9 },
    { token: 'not', logit: 2.5 }, { token: 'very', logit: 2.1 },
    { token: 'about', logit: 1.5 }, { token: 'here', logit: 1.2 },
    { token: 'everything', logit: 0.5 }, { token: 'soup', logit: -0.8 },
  ],
  i: [
    { token: 'think', logit: 3.1 }, { token: 'was', logit: 2.8 },
    { token: 'have', logit: 2.7 }, { token: 'love', logit: 2.3 },
    { token: 'want', logit: 2.1 }, { token: 'never', logit: 1.4 },
    { token: 'refuse', logit: 0.7 }, { token: 'tokenize', logit: -0.6 },
  ],
  models: [
    { token: 'are', logit: 3.3 }, { token: 'can', logit: 2.8 },
    { token: 'learn', logit: 2.5 }, { token: 'predict', logit: 2.3 },
    { token: 'work', logit: 1.8 }, { token: 'hallucinate', logit: 1.2 },
    { token: 'dream', logit: 0.4 }, { token: 'dance', logit: -0.4 },
  ],
  sat: [
    { token: 'on', logit: 3.6 }, { token: 'down', logit: 2.8 },
    { token: 'quietly', logit: 1.9 }, { token: 'beside', logit: 1.7 },
    { token: 'there', logit: 1.4 }, { token: 'still', logit: 1.1 },
    { token: 'up', logit: 0.9 }, { token: 'majestically', logit: 0.3 },
  ],
}

const FALLBACK_TOKENS = [
  'the', 'and', 'is', 'was', 'to', 'of', 'a', 'that', 'it', 'very',
  'quite', 'really', 'suddenly', 'quietly', 'here', 'there',
]

/** Deterministic logits derived from the seed word, so unknown prompts still work. */
function fallback(seed: string): Candidate[] {
  const base = tokenId(seed)
  return FALLBACK_TOKENS.map((token, i) => {
    // A little multiplicative hashing to spread logits across [-1, 3.5].
    const h = Math.abs(Math.imul(base + i * 2654435761, 40503)) % 1000
    return { token, logit: Math.round((h / 1000) * 45 - 10) / 10 }
  })
    .sort((a, b) => b.logit - a.logit)
    .slice(0, 8)
}

export function candidatesFor(prompt: string): Candidate[] {
  const words = prompt.toLowerCase().match(/[a-z']+/g) ?? []
  const last = words[words.length - 1] ?? 'the'
  const cands = TABLE[last] ?? fallback(last)
  return [...cands].sort((a, b) => b.logit - a.logit)
}

/** softmax(logit / T) over the candidate set. */
export function applyTemperature(cands: Candidate[], temperature: number): Array<Candidate & { p: number }> {
  const t = Math.max(0.05, temperature)
  const m = Math.max(...cands.map((c) => c.logit))
  const es = cands.map((c) => Math.exp((c.logit - m) / t))
  const s = es.reduce((a, b) => a + b, 0)
  return cands.map((c, i) => ({ ...c, p: es[i] / s }))
}

export function sample(dist: Array<Candidate & { p: number }>): number {
  let r = Math.random()
  for (let i = 0; i < dist.length; i++) {
    r -= dist[i].p
    if (r <= 0) return i
  }
  return dist.length - 1
}
