/** Human-friendly duration from seconds — spans 24 orders of magnitude */
export function fmtSeconds(s: number): string {
  if (s < 1e-6) return `${round(s * 1e9)} ns`
  if (s < 1e-3) return `${round(s * 1e6)} µs`
  if (s < 1) return `${round(s * 1e3)} ms`
  if (s < 60) return `${round(s)} s`
  if (s < 3600) return `${round(s / 60)} min`
  if (s < 86400) return `${round(s / 3600)} hours`
  const days = s / 86400
  if (days < 365.25) return `${round(days)} days`
  return fmtYears(days / 365.25)
}

/** Human-friendly duration from years */
export function fmtYears(y: number): string {
  if (y < 1e4) return `${Math.round(y).toLocaleString()} years`
  if (y < 1e6) return `${round(y / 1e3)} thousand years`
  if (y < 1e9) return `${round(y / 1e6)} million years`
  return `${round(y / 1e9)} billion years`
}

/** Compact axis-tick form: 5 kyr / 66 Myr / 4.55 Gyr */
export function fmtYearsShort(y: number): string {
  if (y < 1e3) return `${Math.round(y)} yr`
  if (y < 1e6) return `${round(y / 1e3)} kyr`
  if (y < 1e9) return `${round(y / 1e6)} Myr`
  return `${round(y / 1e9)} Gyr`
}

function round(v: number): string {
  if (v >= 100) return String(Math.round(v))
  if (v >= 10) return String(Math.round(v * 10) / 10)
  return String(Math.round(v * 100) / 100)
}

export function pct(v: number): string {
  return `${Math.round(v * 100)}%`
}
