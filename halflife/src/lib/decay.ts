/**
 * Deterministic decay model shared by every "jar" in the app.
 *
 * Each particle gets a unit lifetime u = -log2(random()): the number of
 * half-lives it survives. Its real decay time is u * halfLife, so the same
 * random draw can be replayed at any half-life and scrubbed to any time —
 * the jar is a pure function of (lifetimes, halfLife, t).
 */

/** mulberry32 — tiny seeded PRNG so a jar re-randomizes only on reset */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Unit lifetimes in half-lives: P(u > x) = 2^-x */
export function makeUnitLifetimes(n: number, seed: number): Float64Array {
  const rand = mulberry32(seed)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    // avoid log2(0)
    out[i] = -Math.log2(1 - rand() * 0.999999)
  }
  return out
}

/** Fraction of parents still surviving at time t (measured, not theoretical) */
export function survivingFraction(
  lifetimes: Float64Array,
  halfLife: number,
  t: number,
): number {
  let alive = 0
  for (let i = 0; i < lifetimes.length; i++) {
    if (lifetimes[i] * halfLife > t) alive++
  }
  return alive / lifetimes.length
}

export function survivingCount(
  lifetimes: Float64Array,
  halfLife: number,
  t: number,
): number {
  let alive = 0
  for (let i = 0; i < lifetimes.length; i++) {
    if (lifetimes[i] * halfLife > t) alive++
  }
  return alive
}

/** Theoretical N(t)/N0 = 2^(-t/T) = e^(-λt) */
export function theoreticalFraction(halfLife: number, t: number): number {
  return Math.pow(2, -t / halfLife)
}

export interface ParticlePos {
  x: number
  y: number
}

/**
 * Jittered grid packing inside a rectangle — stable per (n, seed) so
 * particles don't jump between renders.
 */
export function packParticles(
  n: number,
  x0: number,
  y0: number,
  w: number,
  h: number,
  seed: number,
): ParticlePos[] {
  const rand = mulberry32(seed)
  const cols = Math.ceil(Math.sqrt((n * w) / h))
  const rows = Math.ceil(n / cols)
  const cw = w / cols
  const ch = h / rows
  const jitter = Math.min(cw, ch) * 0.22
  const out: ParticlePos[] = []
  for (let i = 0; i < n; i++) {
    const c = i % cols
    const r = Math.floor(i / cols)
    out.push({
      x: x0 + (c + 0.5) * cw + (rand() * 2 - 1) * jitter,
      y: y0 + (r + 0.5) * ch + (rand() * 2 - 1) * jitter,
    })
  }
  return out
}

export const LN2 = Math.LN2
