// Hand-crafted attention pattern for one sample sentence. The weights are
// invented, but they encode the relationships a trained head plausibly finds
// (e.g. "it" resolving to "ball"), which is what the visualization teaches.

export const SENTENCE = [
  'The', 'robot', 'picked', 'up', 'the', 'ball', 'because', 'it', 'was', 'heavy',
] as const

export const N = SENTENCE.length

// boosts[i] lists (j, strength) pairs: token i pays extra attention to token j.
const BOOSTS: Record<number, Array<[number, number]>> = {
  0: [[1, 2.0]], // The -> robot (determiner looks at its noun)
  1: [[0, 1.2], [2, 1.6]], // robot -> The, picked
  2: [[1, 2.2], [3, 1.8], [5, 1.4]], // picked -> robot, up, ball
  3: [[2, 2.4]], // up -> picked (phrasal verb)
  4: [[5, 2.2]], // the -> ball
  5: [[2, 1.6], [4, 1.4]], // ball -> picked, the
  6: [[2, 1.0], [9, 1.2]], // because -> picked, heavy
  7: [[5, 2.6], [1, 1.1]], // it -> ball (coreference!), robot
  8: [[7, 1.8], [9, 1.6]], // was -> it, heavy
  9: [[5, 2.2], [7, 1.9], [8, 1.0]], // heavy -> ball, it, was
}

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs)
  const es = xs.map((x) => Math.exp(x - m))
  const s = es.reduce((a, b) => a + b, 0)
  return es.map((e) => e / s)
}

/** scores[i][j] = raw (pre-softmax) compatibility of query i with key j. */
export const SCORES: number[][] = Array.from({ length: N }, (_, i) =>
  Array.from({ length: N }, (_, j) => {
    let s = -1.0 // small base score for every pair
    if (i === j) s += 1.2 // tokens mildly attend to themselves
    if (Math.abs(i - j) === 1) s += 0.5 // and to their neighbors
    for (const [tj, boost] of BOOSTS[i] ?? []) if (tj === j) s += boost
    return Math.round(s * 10) / 10
  }),
)

/** weights[i][j] = softmax over row i — what token i attends to. */
export const WEIGHTS: number[][] = SCORES.map(softmax)
