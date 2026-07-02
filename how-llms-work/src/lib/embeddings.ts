// Toy word embeddings. Each word gets a hand-crafted 8-dimensional feature
// vector; the 2D scatter positions are a fixed linear projection of those
// vectors, so points that are close on screen genuinely have high cosine
// similarity — the map is honest, just tiny.

export const DIMS = [
  'royalty', 'masculine', 'animate', 'techy', 'edible', 'positive', 'big', 'abstract',
] as const

export interface WordVec {
  word: string
  cluster: 'royalty' | 'animals' | 'food' | 'tech' | 'feelings'
  v: number[] // 8 dims, each in [-1, 1]
  x: number // projected 2D position, filled in below
  y: number
}

const RAW: Array<Omit<WordVec, 'x' | 'y'>> = [
  { word: 'king', cluster: 'royalty', v: [0.95, 0.5, 0.8, 0, 0, 0.1, 0.6, 0.1] },
  { word: 'queen', cluster: 'royalty', v: [0.95, -0.5, 0.8, 0, 0, 0.2, 0.5, 0.1] },
  { word: 'prince', cluster: 'royalty', v: [0.8, 0.5, 0.8, 0, 0, 0.2, 0.4, 0.1] },
  { word: 'princess', cluster: 'royalty', v: [0.8, -0.5, 0.8, 0, 0, 0.3, 0.35, 0.1] },
  { word: 'crown', cluster: 'royalty', v: [0.85, 0, 0, 0, 0, 0.1, 0.2, 0.2] },
  { word: 'dog', cluster: 'animals', v: [0, 0.1, 0.95, 0, 0, 0.5, 0.4, 0] },
  { word: 'cat', cluster: 'animals', v: [0, -0.1, 0.95, 0, 0, 0.3, 0.3, 0] },
  { word: 'puppy', cluster: 'animals', v: [0, 0.1, 0.95, 0, 0, 0.8, 0.15, 0] },
  { word: 'kitten', cluster: 'animals', v: [0, -0.1, 0.95, 0, 0, 0.8, 0.1, 0] },
  { word: 'wolf', cluster: 'animals', v: [0, 0.2, 0.9, 0, 0, -0.3, 0.5, 0] },
  { word: 'apple', cluster: 'food', v: [0, 0, 0, 0, 0.95, 0.4, 0.15, 0] },
  { word: 'banana', cluster: 'food', v: [0, 0.05, 0, 0, 0.9, 0.35, 0.3, 0] },
  { word: 'bread', cluster: 'food', v: [0, 0, 0, 0, 0.85, 0.15, 0.35, 0.05] },
  { word: 'cheese', cluster: 'food', v: [0, -0.05, 0, 0, 0.88, 0.25, 0.2, 0.05] },
  { word: 'pizza', cluster: 'food', v: [0, 0, 0, 0.05, 0.95, 0.65, 0.4, 0] },
  { word: 'computer', cluster: 'tech', v: [0, 0, 0, 0.95, 0, 0, 0.5, 0.2] },
  { word: 'robot', cluster: 'tech', v: [0, 0.1, 0.3, 0.9, 0, 0, 0.5, 0.1] },
  { word: 'software', cluster: 'tech', v: [0, 0, 0, 0.95, 0, 0, 0.2, 0.6] },
  { word: 'algorithm', cluster: 'tech', v: [0, 0, 0, 0.9, 0, 0, 0.1, 0.9] },
  { word: 'data', cluster: 'tech', v: [0, 0, 0, 0.85, 0, 0, 0.2, 0.8] },
  { word: 'happy', cluster: 'feelings', v: [0, 0, 0.2, 0, 0, 0.95, 0, 0.8] },
  { word: 'joy', cluster: 'feelings', v: [0, 0, 0.1, 0, 0, 0.95, 0, 0.9] },
  { word: 'sad', cluster: 'feelings', v: [0, 0, 0.2, 0, 0, -0.9, 0, 0.8] },
  { word: 'angry', cluster: 'feelings', v: [0, 0.1, 0.3, 0, 0, -0.85, 0, 0.7] },
]

// Two fixed projection directions (a stand-in for t-SNE/PCA axes).
const P1 = [0.9, 0.1, -0.3, -0.8, 0.5, 0.0, 0.1, -0.2]
const P2 = [0.2, 0.6, 0.8, -0.4, -0.7, 0.5, 0.2, 0.1]

function dot(a: number[], b: number[]): number {
  return a.reduce((s, ai, i) => s + ai * b[i], 0)
}

export function cosine(a: number[], b: number[]): number {
  const na = Math.sqrt(dot(a, a))
  const nb = Math.sqrt(dot(b, b))
  if (na === 0 || nb === 0) return 0
  return dot(a, b) / (na * nb)
}

function project(): WordVec[] {
  const pts = RAW.map((w) => ({ ...w, x: dot(w.v, P1), y: dot(w.v, P2) }))
  const xs = pts.map((p) => p.x)
  const ys = pts.map((p) => p.y)
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)]
  // Normalize into [0.06, 0.94] so labels don't clip at the plot edges.
  const out = pts.map((p) => ({
    ...p,
    x: 0.06 + (0.88 * (p.x - x0)) / (x1 - x0),
    y: 0.06 + (0.88 * (p.y - y0)) / (y1 - y0),
  }))
  // Near-duplicate vectors land on the same pixel; nudge overlapping points
  // apart so every dot and label stays readable. Clusters survive the nudge.
  const MIN_D = 0.075
  for (let iter = 0; iter < 80; iter++) {
    for (let i = 0; i < out.length; i++) {
      for (let j = i + 1; j < out.length; j++) {
        const dx = out[j].x - out[i].x
        const dy = out[j].y - out[i].y
        const d = Math.hypot(dx, dy)
        if (d >= MIN_D) continue
        const angle = d < 1e-6 ? (i * 2.4 + j) : Math.atan2(dy, dx)
        const push = (MIN_D - d) / 2
        const ux = d < 1e-6 ? Math.cos(angle) : dx / d
        const uy = d < 1e-6 ? Math.sin(angle) : dy / d
        out[i].x -= ux * push
        out[i].y -= uy * push
        out[j].x += ux * push
        out[j].y += uy * push
      }
    }
  }
  for (const p of out) {
    p.x = Math.min(0.93, Math.max(0.07, p.x))
    p.y = Math.min(0.95, Math.max(0.07, p.y))
  }
  return out
}

export const WORDS: WordVec[] = project()

export const CLUSTERS = ['royalty', 'animals', 'food', 'tech', 'feelings'] as const
