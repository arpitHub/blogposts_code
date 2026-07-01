import { useMemo, useState } from 'react'
import { Section } from '../components/Section'
import { ParticleJar } from '../components/ParticleJar'
import { Btn } from '../components/ui'
import { D, useDepth } from '../lib/depth'
import { makeUnitLifetimes, survivingFraction } from '../lib/decay'
import { fmtYears, fmtYearsShort } from '../lib/format'

const N = 120
const LOG_MIN = 2 // 100 years
const LOG_MAX = 10.15 // ~14 billion years

interface Clock {
  id: string
  name: string
  pair: string
  halfLife: number // years
  seed: number
  use: string
  techNote: string
}

const CLOCKS: Clock[] = [
  {
    id: 'c14',
    name: 'Carbon-14',
    pair: '¹⁴C → ¹⁴N',
    halfLife: 5730,
    seed: 11,
    use: 'bones, wood, cloth — archaeology',
    techNote: 'usable to ~50 kyr (~9 half-lives); needs once-living material',
  },
  {
    id: 'kar',
    name: 'Potassium-Argon',
    pair: '⁴⁰K → ⁴⁰Ar',
    halfLife: 1.248e9,
    seed: 12,
    use: 'volcanic rock and ash layers',
    techNote: 'clock zeroes when lava degasses; brackets fossil beds',
  },
  {
    id: 'upb',
    name: 'Uranium-Lead',
    pair: '²³⁸U → ²⁰⁶Pb',
    halfLife: 4.468e9,
    seed: 13,
    use: 'the oldest rocks and meteorites',
    techNote: 'two chains (²³⁸U, ²³⁵U) in one zircon cross-check each other',
  },
]

const PRESETS: { label: string; years: number }[] = [
  { label: 'Egyptian mummy', years: 3000 },
  { label: 'Cave paintings', years: 40000 },
  { label: 'T. rex', years: 6.8e7 },
  { label: 'Oldest zircon', years: 4.4e9 },
]

export function ClocksSection() {
  const depth = useDepth()
  const [logT, setLogT] = useState(3.5)
  const time = Math.pow(10, logT)

  const lifetimes = useMemo(
    () => CLOCKS.map((c) => makeUnitLifetimes(N, c.seed)),
    [],
  )

  return (
    <Section id="clocks" kicker="04 · Choosing a clock" title="Different clocks for different timescales">
      <p className="mb-6 max-w-3xl text-[15px] leading-relaxed text-ink-2">
        <D
          b={
            <>
              Same physics, three speeds. Drag the time slider: carbon-14 burns
              through its jar while uranium barely notices. A clock is only
              useful while it’s still ticking — you’d never time a marathon
              with a stopwatch that dies after a minute, or a sprint with a
              calendar. So you pick the atom to match the age of the thing
              you’re dating.
            </>
          }
          t={
            <>
              A radiometric clock resolves ages between roughly 0.1 and 10
              half-lives: earlier, too little daughter has accumulated to
              measure; later, too little parent survives. Drag the (log) time
              axis and watch each system sweep through its usable window —
              ¹⁴C saturates by ~50 kyr, while ²³⁸U→²⁰⁶Pb is still mid-curve at
              the age of the Earth.
            </>
          }
        />
      </p>

      {/* shared time control */}
      <div className="mb-6 rounded-xl border border-line bg-panel px-5 py-4">
        <div className="mb-1.5 flex items-baseline justify-between">
          <span className="text-sm text-ink-2">
            <D b="Time since the clock started" t="Elapsed time (log scale)" />
          </span>
          <span className="text-lg font-semibold tabular-nums text-amber-glow">
            {fmtYears(time)}
          </span>
        </div>
        <input
          type="range"
          className="hl-range"
          min={LOG_MIN}
          max={LOG_MAX}
          step={0.01}
          value={logT}
          onChange={(e) => setLogT(Number(e.target.value))}
          aria-label="Elapsed time, logarithmic"
        />
        <div className="mt-1 flex justify-between text-[10px] text-ink-3">
          <span>100 yr</span>
          <span>10 kyr</span>
          <span>1 Myr</span>
          <span>100 Myr</span>
          <span>10 Gyr</span>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <Btn key={p.label} onClick={() => setLogT(Math.log10(p.years))}>
              {p.label} · {fmtYearsShort(p.years)}
            </Btn>
          ))}
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-3">
        {CLOCKS.map((c, i) => {
          const frac = survivingFraction(lifetimes[i], c.halfLife, time)
          const halfLives = time / c.halfLife
          const inWindow = halfLives >= 0.1 && halfLives <= 10
          return (
            <div key={c.id} className="rounded-xl border border-line bg-panel p-4">
              <div className="mb-1 flex items-baseline justify-between">
                <h3 className="font-semibold text-ink">{c.name}</h3>
                <span className="font-mono text-xs text-ink-3">{c.pair}</span>
              </div>
              <div className="mb-2 text-xs text-ink-3">
                half-life {fmtYears(c.halfLife)} · {c.use}
              </div>
              <ParticleJar
                lifetimes={lifetimes[i]}
                halfLife={c.halfLife}
                time={time}
                seed={c.seed}
                compact
              />
              <div className="mt-3">
                <div className="flex justify-between text-xs text-ink-3">
                  <span>parent left</span>
                  <span
                    className={`font-semibold tabular-nums ${
                      frac < 0.01 ? 'text-ink-3' : 'text-amber-glow'
                    }`}
                  >
                    {frac < 0.01 && frac > 0
                      ? '<1%'
                      : `${Math.round(frac * 100)}%`}
                  </span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-line">
                  <div
                    className="h-full rounded-full bg-amber-series transition-all duration-300"
                    style={{ width: `${frac * 100}%` }}
                  />
                </div>
                <div
                  className={`mt-2 text-xs ${
                    inWindow ? 'text-blue-glow' : 'text-ink-3'
                  }`}
                >
                  {inWindow ? (
                    <D b="✓ clock is readable here" t={`✓ in range (${halfLives.toFixed(halfLives < 1 ? 2 : 1)} half-lives)`} />
                  ) : halfLives < 0.1 ? (
                    <D b="too early — barely any change yet" t={`below resolution (${halfLives.toExponential(1)} t½)`} />
                  ) : (
                    <D b="used up — the clock has stopped" t={`saturated (${Math.round(halfLives)} t½ elapsed)`} />
                  )}
                </div>
                {depth === 'technical' && (
                  <div className="mt-2 border-t border-line pt-2 text-xs leading-relaxed text-ink-3">
                    {c.techNote}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </Section>
  )
}
