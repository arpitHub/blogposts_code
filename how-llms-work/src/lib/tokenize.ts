// A toy word-piece tokenizer. It is not a real BPE implementation, but it
// mimics the behavior people notice in real tokenizers: common words stay
// whole, rare/long words split at morpheme-ish boundaries, and punctuation
// becomes its own token.

export interface Token {
  text: string
  id: number
  wordIndex: number // which whitespace-separated word this piece came from
  pieceIndex: number // position of this piece within its word
}

export const VOCAB_SIZE = 50257

const COMMON = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be',
  'been', 'it', 'this', 'that', 'these', 'those', 'of', 'to', 'in', 'on',
  'for', 'with', 'as', 'at', 'by', 'from', 'up', 'about', 'into', 'over',
  'after', 'before', 'how', 'why', 'what', 'when', 'where', 'who', 'can',
  'will', 'would', 'could', 'should', 'not', 'no', 'yes', 'so', 'if', 'out',
  'all', 'any', 'each', 'many', 'some', 'then', 'them', 'they', 'their',
  'there', 'she', 'he', 'her', 'him', 'his', 'you', 'your', 'we', 'our',
  'i', 'my', 'me', 'do', 'does', 'did', 'has', 'have', 'had', 'very',
  'really', 'just', 'like', 'time', 'people', 'way', 'day', 'new', 'old',
  'see', 'get', 'make', 'go', 'know', 'take', 'think', 'come', 'give',
  'look', 'use', 'find', 'want', 'tell', 'ask', 'work', 'works', 'word',
  'words', 'world', 'model', 'models', 'water', 'house', 'money', 'story',
  'never', 'always', 'under', 'again', 'still', 'because', 'other', 'first',
  'little', 'good', 'great', 'right', 'even', 'back', 'only', 'much',
])

// Ordered longest-first so the greedy split prefers bigger suffixes
// ("generative" -> "gener" + "ative", not "generat" + "ive").
const SUFFIXES = [
  'ization', 'ically', 'ations', 'ingly', 'ation', 'ative', 'ously',
  'ments', 'ness', 'ment', 'able', 'ible', 'tion', 'sion', 'ally', 'ical',
  'ling', 'ings', 'ing', 'ers', 'est', 'ful', 'ity', 'ive', 'ize', 'ous',
  'ies', 'ial', 'ish', 'ed', 'er', 'ly', 'al', 'en', 'es', 's',
]

function splitWord(w: string, isStem = false): string[] {
  const lw = w.toLowerCase()
  if (COMMON.has(lw) || lw.length <= 4) return [w]
  for (const suf of SUFFIXES) {
    // Once we're inside a stem, only peel real morphemes — stripping short
    // suffixes again produces junk like "surpri" + "s" + "ingly".
    if (isStem && suf.length < 3) continue
    if (lw.endsWith(suf) && lw.length - suf.length >= 3) {
      const stem = w.slice(0, w.length - suf.length)
      return [...splitWord(stem, true), w.slice(w.length - suf.length)]
    }
  }
  if (w.length > 8) {
    const mid = Math.ceil(w.length / 2)
    return [w.slice(0, mid), w.slice(mid)]
  }
  return [w]
}

/** Deterministic FNV-1a hash into the toy vocab range, so token IDs are stable. */
export function tokenId(piece: string): number {
  let h = 2166136261
  for (const ch of piece) {
    h ^= ch.codePointAt(0)!
    h = Math.imul(h, 16777619)
  }
  return Math.abs(h) % VOCAB_SIZE
}

export function tokenize(text: string): Token[] {
  const tokens: Token[] = []
  const words = text.split(/\s+/).filter(Boolean)
  words.forEach((word, wordIndex) => {
    // Peel punctuation off the edges of the word into standalone tokens.
    const parts = word.match(/[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]/g) ?? []
    let pieceIndex = 0
    for (const part of parts) {
      const pieces = /^[A-Za-z]/.test(part) ? splitWord(part) : [part]
      for (const piece of pieces) {
        tokens.push({ text: piece, id: tokenId(piece.toLowerCase()), wordIndex, pieceIndex })
        pieceIndex++
      }
    }
  })
  return tokens
}
