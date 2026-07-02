// Colors validated against the dark surface (#1a1a19): all clear 3:1 contrast,
// worst adjacent-pair CVD deltaE 41.3.
export const INK = {
  primary: '#ffffff',
  secondary: '#c3c2b7',
  muted: '#898781',
  grid: '#2c2c2a',
  axis: '#383835',
  surface: '#1a1a19',
  page: '#0d0d0d',
} as const

// Fixed categorical order — token chips cycle through these in every section.
export const TOKEN_COLORS = ['#3987e5', '#199e70', '#c98500', '#9085e9', '#e66767'] as const

// "This token" — the focused/selected token is always this blue, in every section.
export const FOCUS = '#3987e5'

// Sequential blue ramp, ordered dark -> light so low values recede toward the
// dark surface and high values glow (used by the attention heatmap).
export const SEQ_BLUE_DARK_TO_LIGHT = [
  '#0d366b', '#104281', '#184f95', '#1c5cab', '#256abf', '#2a78d6',
  '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb',
] as const

export function tokenColor(i: number): string {
  return TOKEN_COLORS[i % TOKEN_COLORS.length]
}

/** Map a 0..1 value onto the sequential ramp (low = recedes, high = glows). */
export function seqColor(v: number): string {
  const ramp = SEQ_BLUE_DARK_TO_LIGHT
  const t = Math.min(1, Math.max(0, v))
  return ramp[Math.round(t * (ramp.length - 1))]
}
