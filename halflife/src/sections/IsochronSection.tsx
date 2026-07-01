import { useMemo, useRef, useState } from 'react'
import { Section } from '../components/Section'
import { Btn, EqCard, LegendItem, SliderRow, Stat } from '../components/ui'
import { D, useDepth } from '../lib/depth'
import { useClock } from '../lib/useClock'

/**
 * Patterson's Pb–Pb isochron, rebuilt as a physical simulation.
 *
 * Every sample starts at the same primordial lead composition (Canyon
 * Diablo troilite) and grows radiogenic Pb at a rate set by its own U/Pb
 * ratio μ. At any elapsed time all samples are colinear; the line pivots
 * around the primordial point and its slope encodes the age.
 */

const L238 = Math.LN2 / 4.468 // per Gyr
const L235 = Math.LN2 / 0.704
const U_RATIO = 137.88 // ²³⁸U/²³⁵U today
const T_TRUE = 4.55 // Gyr

const PRIMORDIAL = { x: 9.307, y: 10.294 }

// measured ²⁰⁶Pb/²⁰⁴Pb of Patterson's five meteorites + modern Earth sediment
// lx/ly/anchor: label placement, hand-set to avoid collisions when settled
const SAMPLES: {
  name: string
  xm: number
  earth?: boolean
  lx: number
  ly: number
  anchor: 'start' | 'end'
}[] = [
  { name: 'Canyon Diablo (iron)', xm: 9.46, lx: 10, ly: -8, anchor: 'start' },
  { name: 'Henbury (iron)', xm: 9.55, lx: 12, ly: 8, anchor: 'start' },
  { name: 'Forest City (stone)', xm: 19.27, lx: -10, ly: -12, anchor: 'end' },
  { name: 'Modoc (stone)', xm: 19.48, lx: 14, ly: 16, anchor: 'start' },
  { name: 'Nuevo Laredo (stone)', xm: 50.28, lx: -10, ly: -10, anchor: 'end' },
  {
    name: 'Earth (ocean sediment)',
    xm: 19.0,
    earth: true,
    lx: -14,
    ly: 22,
    anchor: 'end',
  },
]

/** radiogenic growth factors at elapsed time t (of a T_TRUE-old system) */
function growth(t: number) {
  const g238 = Math.exp(L238 * T_TRUE) - Math.exp(L238 * (T_TRUE - t))
  const g235 = Math.exp(L235 * T_TRUE) - Math.exp(L235 * (T_TRUE - t))
  return { g238, g235 }
}

/** invert slope -> age via bisection on s(t) = (e^λ₂₃₅ᵗ−1)/(137.88(e^λ₂₃₈ᵗ−1)) */
function ageFromSlope(slope: number): number {
  let lo = 0.001
  let hi = 10
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2
    const s =
      (Math.exp(L235 * mid) - 1) / (U_RATIO * (Math.exp(L238 * mid) - 1))
    if (s < slope) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

// chart geometry
const W = 620
const H = 430
const M = { l: 58, r: 20, t: 16, b: 46 }
const X_MAX = 55
const Y_MAX = 40
const sx = (v: number) => M.l + (v / X_MAX) * (W - M.l - M.r)
const sy = (v: number) => H - M.b - (v / Y_MAX) * (H - M.t - M.b)

export function IsochronSection() {
  const depth = useDepth()
  const { time, setTime, running, setRunning, reset } = useClock(0.65, T_TRUE)
  const [offsets, setOffsets] = useState<number[]>(SAMPLES.map(() => 0))
  const dragIdx = useRef<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const mus = useMemo(() => {
    const { g238 } = growth(T_TRUE)
    return SAMPLES.map((s) => (s.xm - PRIMORDIAL.x) / g238)
  }, [])

  const { g238, g235 } = growth(time)
  const pts = SAMPLES.map((s, i) => ({
    ...s,
    x: PRIMORDIAL.x + mus[i] * g238,
    y: PRIMORDIAL.y + (mus[i] / U_RATIO) * g235 + offsets[i],
  }))

  // least-squares fit through the (possibly dragged) points
  const fit = useMemo(() => {
    const n = pts.length
    const mx = pts.reduce((a, p) => a + p.x, 0) / n
    const my = pts.reduce((a, p) => a + p.y, 0) / n
    let sxy = 0
    let sxx = 0
    for (const p of pts) {
      sxy += (p.x - mx) * (p.y - my)
      sxx += (p.x - mx) * (p.x - mx)
    }
    const slope = sxx > 1e-9 ? sxy / sxx : 0
    const icpt = my - slope * mx
    const rms = Math.sqrt(
      pts.reduce((a, p) => a + (p.y - (icpt + slope * p.x)) ** 2, 0) / n,
    )
    return { slope, icpt, rms }
  }, [pts])

  const disturbed = offsets.some((o) => Math.abs(o) > 0.15)
  const settled = time >= T_TRUE
  // Undisturbed mid-animation, the slope→age inversion must use the
  // ²³⁵U/²³⁸U ratio of that epoch, which lands exactly on the elapsed time —
  // so show it directly. Once disturbed (a today-measurement scenario),
  // invert with the present-day ratio.
  const age = disturbed ? (fit.slope > 0 ? ageFromSlope(fit.slope) : 0) : time

  const onDrag = (clientY: number) => {
    const i = dragIdx.current
    const svg = svgRef.current
    if (i === null || !svg) return
    const rect = svg.getBoundingClientRect()
    const vy = ((clientY - rect.top) / rect.height) * H
    const targetY = ((H - M.b - vy) / (H - M.t - M.b)) * Y_MAX
    const baseY = PRIMORDIAL.y + (mus[i] / U_RATIO) * growth(time).g235
    setOffsets((prev) => {
      const next = [...prev]
      next[i] = Math.max(-8, Math.min(8, targetY - baseY))
      return next
    })
  }

  const handlePlay = () => {
    if (settled && !running) {
      setTime(0)
      setOffsets(SAMPLES.map(() => 0))
      setRunning(true)
    } else setRunning(!running)
  }

  return (
    <Section id="isochron" kicker="05 · The measurement" title="How Patterson dated the Earth">
      <div className="grid gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div>
          <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1">
            <LegendItem color="#f5b846" label="meteorite sample" />
            <LegendItem color="#3987e5" label="Earth (ocean sediment)" />
            <LegendItem color="#c98500" label="best-fit isochron" dashed />
          </div>
          <svg
            ref={svgRef}
            viewBox={`0 0 ${W} ${H}`}
            className="h-auto w-full touch-none rounded-xl border border-line bg-panel"
            role="img"
            aria-label="Lead-lead isochron plot of meteorite samples"
            onPointerMove={(e) => onDrag(e.clientY)}
            onPointerUp={() => (dragIdx.current = null)}
            onPointerLeave={() => (dragIdx.current = null)}
          >
            {/* gridlines */}
            {[10, 20, 30, 40, 50].map((v) => (
              <line key={`gx${v}`} x1={sx(v)} y1={sy(0)} x2={sx(v)} y2={M.t} stroke="#1c1c26" />
            ))}
            {[10, 20, 30].map((v) => (
              <line key={`gy${v}`} x1={M.l} y1={sy(v)} x2={W - M.r} y2={sy(v)} stroke="#1c1c26" />
            ))}
            {/* axes */}
            <line x1={M.l} y1={sy(0)} x2={W - M.r} y2={sy(0)} stroke="#34343f" />
            <line x1={M.l} y1={sy(0)} x2={M.l} y2={M.t} stroke="#34343f" />
            {[0, 10, 20, 30, 40, 50].map((v) => (
              <text key={`tx${v}`} x={sx(v)} y={sy(0) + 16} textAnchor="middle" fontSize="10" fill="#8a8894">
                {v}
              </text>
            ))}
            {[10, 20, 30, 40].map((v) => (
              <text key={`ty${v}`} x={M.l - 8} y={sy(v) + 3} textAnchor="end" fontSize="10" fill="#8a8894">
                {v}
              </text>
            ))}
            <text x={(M.l + W - M.r) / 2} y={H - 8} textAnchor="middle" fontSize="11" fill="#8a8894">
              <D b="lead made by slow uranium →" t="²⁰⁶Pb / ²⁰⁴Pb" />
            </text>
            <text
              x={16} y={(M.t + H - M.b) / 2}
              textAnchor="middle" fontSize="11" fill="#8a8894"
              transform={`rotate(-90 16 ${(M.t + H - M.b) / 2})`}
            >
              <D b="lead made by fast uranium →" t="²⁰⁷Pb / ²⁰⁴Pb" />
            </text>

            {/* best-fit line */}
            {fit.slope > 0 && (
              <line
                x1={sx(0)}
                y1={sy(fit.icpt)}
                x2={sx(X_MAX)}
                y2={sy(fit.icpt + fit.slope * X_MAX)}
                stroke="#c98500"
                strokeWidth="2"
                strokeDasharray={disturbed ? '5 5' : 'none'}
                opacity={time > 0.02 ? 1 : 0}
              />
            )}

            {/* residual sticks when disturbed */}
            {disturbed &&
              pts.map((p, i) =>
                Math.abs(offsets[i]) > 0.15 ? (
                  <line
                    key={`res${i}`}
                    x1={sx(p.x)}
                    y1={sy(p.y)}
                    x2={sx(p.x)}
                    y2={sy(fit.icpt + fit.slope * p.x)}
                    stroke="#e66767"
                    strokeWidth="1.5"
                    strokeDasharray="3 3"
                  />
                ) : null,
              )}

            {/* primordial point */}
            <circle cx={sx(PRIMORDIAL.x)} cy={sy(PRIMORDIAL.y)} r={3.5} fill="#8a8894" />
            <text
              x={sx(PRIMORDIAL.x) - 6}
              y={sy(PRIMORDIAL.y) + 16}
              fontSize="9"
              fill="#8a8894"
              textAnchor="end"
            >
              primordial lead
            </text>

            {/* samples */}
            {pts.map((p, i) => (
              <g key={p.name}>
                <circle
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r={14}
                  fill="transparent"
                  className="cursor-grab"
                  onPointerDown={(e) => {
                    dragIdx.current = i
                    ;(e.target as Element).setPointerCapture(e.pointerId)
                  }}
                />
                <circle
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r={6}
                  fill={p.earth ? '#3987e5' : '#f5b846'}
                  stroke="#0a0a0f"
                  strokeWidth="2"
                  pointerEvents="none"
                />
                {settled && (
                  <text
                    x={sx(p.x) + p.lx}
                    y={sy(p.y) + p.ly}
                    fontSize="9"
                    fill="#b9b7c0"
                    textAnchor={p.anchor}
                  >
                    {p.name}
                  </text>
                )}
              </g>
            ))}
          </svg>
          <p className="mt-2 text-xs text-ink-3">
            After the points settle, drag one up or down — see what a disturbed
            sample does to the line.
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <p className="max-w-prose text-[15px] leading-relaxed text-ink-2">
            <D
              b={
                <>
                  Every meteorite was born from the same cloud, at the same
                  moment, but with different amounts of uranium mixed in. As
                  eons pass, uranium turns to lead — rocks with more uranium
                  drift further. Press play: because they all share one
                  birthday, they stay on one straight line, and the line’s{' '}
                  <strong>tilt is the age</strong>. Five space rocks and the
                  Earth itself agree on 4.55 billion years.
                </>
              }
              t={
                <>
                  Each sample’s Pb composition evolves with its own μ =
                  ²³⁸U/²⁰⁴Pb, but ²⁰⁷Pb and ²⁰⁶Pb grow in a ratio fixed only by
                  time. Samples of common age are therefore colinear, pivoting
                  about primordial Pb — no assumption about initial lead needed.
                  Scatter, not slope, is the failure signal: drag a point to
                  simulate open-system behaviour and watch the RMS blow up
                  while the line stops meaning anything.
                </>
              }
            />
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <Btn primary onClick={handlePlay}>
              {running ? '❚❚ Pause' : settled ? '↺ Replay 4.55 Gyr' : '▶ Play'}
            </Btn>
            <Btn
              onClick={() => {
                reset()
                setOffsets(SAMPLES.map(() => 0))
              }}
            >
              ↺ Reset
            </Btn>
          </div>
          <SliderRow
            label={<D b="Time since the solar system formed" t="Elapsed time t" />}
            min={0}
            max={T_TRUE}
            step={0.01}
            value={time}
            onChange={(v) => {
              setRunning(false)
              setTime(v)
              setOffsets(SAMPLES.map(() => 0))
            }}
            display={`${time.toFixed(2)} Gyr`}
          />

          <div className="grid grid-cols-2 gap-2">
            <Stat
              label="isochron slope"
              value={fit.slope > 0 ? fit.slope.toFixed(3) : '—'}
            />
            <Stat
              label={disturbed ? 'apparent age (untrustworthy)' : 'age from slope'}
              value={time > 0.02 ? `${age.toFixed(2)} Gyr` : '—'}
              accent={disturbed ? undefined : 'amber'}
            />
          </div>

          {disturbed && (
            <div className="rounded-lg border border-[#e66767]/40 bg-[#e66767]/10 px-4 py-3 text-sm text-ink-2">
              <D
                b={
                  <>
                    The points no longer line up — and that’s the built-in lie
                    detector. A real disturbed rock betrays itself exactly like
                    this, so geologists throw it out rather than get a wrong
                    age.
                  </>
                }
                t={
                  <>
                    RMS residual {fit.rms.toFixed(2)} — an errorchron.
                    Colinearity is the closed-system test: real labs reject
                    fits like this (MSWD ≫ 1) instead of quoting an age.
                  </>
                }
              />
            </div>
          )}

          {depth === 'technical' && (
            <EqCard
              note={
                <>
                  λ₂₃₅ ≫ λ₂₃₈, so early history grows ²⁰⁷Pb fast and the slope
                  is a sensitive chronometer. Patterson (1956): 4.55 ± 0.07 Gyr.
                </>
              }
            >
              slope = (¹⁄₁₃₇.₈₈) · (e<sup>λ₂₃₅t</sup> − 1)/(e<sup>λ₂₃₈t</sup> − 1)
            </EqCard>
          )}
        </div>
      </div>
    </Section>
  )
}
