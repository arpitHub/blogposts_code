import { useState } from 'react'
import { motion } from 'framer-motion'
import { Section } from '../components/Section'
import { D, useDepth } from '../lib/depth'
import { fmtYears } from '../lib/format'

interface Marker {
  yearsAgo: number
  label: string
  method: string
  color: string
}

const MARKERS: Marker[] = [
  { yearsAgo: 5000, label: 'Written human history', method: 'historical records, tree rings, ¹⁴C', color: '#3987e5' },
  { yearsAgo: 300000, label: 'First Homo sapiens', method: '⁴⁰Ar/³⁹Ar on volcanic ash (Jebel Irhoud)', color: '#3987e5' },
  { yearsAgo: 6.6e7, label: 'End of the dinosaurs', method: '⁴⁰Ar/³⁹Ar on the impact layer', color: '#c98500' },
  { yearsAgo: 3.8e9, label: 'First life on Earth', method: 'U–Pb on host rock; C-isotope traces', color: '#c98500' },
  { yearsAgo: 4.404e9, label: 'Oldest known mineral', method: 'U–Pb on a Jack Hills zircon', color: '#c98500' },
  { yearsAgo: 4.55e9, label: 'Age of the Earth', method: 'Pb–Pb meteorite isochron (Patterson)', color: '#f5b846' },
  { yearsAgo: 13.8e9, label: 'The Big Bang', method: 'CMB (Planck) + expansion rate', color: '#7db5f0' },
]

const NOW = 5000 // slider anchored so "written history" is visible on linear too
const MAX = 13.8e9

export function DeepTimeSection() {
  const depth = useDepth()
  // technical mode unlocks a log axis; beginner is pure linear scale
  const [logScale, setLogScale] = useState(false)
  const useLog = depth === 'technical' && logScale

  const pos = (yearsAgo: number) => {
    if (useLog) {
      const lo = Math.log10(NOW)
      const hi = Math.log10(MAX)
      return ((Math.log10(Math.max(yearsAgo, NOW)) - lo) / (hi - lo)) * 100
    }
    return (yearsAgo / MAX) * 100
  }

  return (
    <Section id="deeptime" kicker="07 · The payoff" title="All of it, on one scale">
      <p className="mb-6 max-w-3xl text-[15px] leading-relaxed text-ink-2">
        <D
          b={
            <>
              This is what those atomic clocks bought us: a map of time itself.
              All of written history — every pharaoh, war, and invention — is
              the thinnest sliver on the far right. Life, the dinosaurs, the
              Earth, the universe: each dated by counting atoms. Try to find
              human history without zooming.
            </>
          }
          t={
            <>
              Every marker below is anchored by a dated measurement, not a
              guess — the method is named on each. On a linear axis the last
              5,000 years collapse to a hairline against 13.8 Gyr; flip to log
              scale to give recent events room. That compression <em>is</em> the
              point of deep time.
            </>
          }
        />
      </p>

      {depth === 'technical' && (
        <div className="mb-5 flex items-center gap-3">
          <span className="text-sm text-ink-2">Axis:</span>
          <div className="flex rounded-md border border-line bg-panel p-1">
            {(
              [
                [false, 'Linear'],
                [true, 'Logarithmic'],
              ] as const
            ).map(([v, label]) => (
              <button
                key={label}
                onClick={() => setLogScale(v)}
                className={`rounded px-3 py-1 text-xs font-medium transition-colors ${
                  logScale === v ? 'bg-amber-glow text-void' : 'text-ink-2 hover:text-ink'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="text-xs text-ink-3">
            {useLog ? 'each step = ×10 in age' : 'true proportional scale'}
          </span>
        </div>
      )}

      <div className="rounded-xl border border-line bg-panel p-5 sm:p-8">
        {/* the scale bar */}
        <div className="relative h-3 rounded-full bg-gradient-to-r from-[#12233a] via-[#3d2f12] to-[#f5b846]/40">
          {MARKERS.map((m) => (
            <div
              key={m.label}
              className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-ink/40"
              style={{ left: `${pos(m.yearsAgo)}%` }}
            />
          ))}
        </div>
        <div className="mt-1 flex justify-between text-[10px] text-ink-3">
          <span>{useLog ? '5,000 yr ago' : 'today'}</span>
          <span>13.8 billion years ago</span>
        </div>

        {/* stacked marker cards — greedy lane packing so close markers
            never overlap horizontally (matters most on the log axis) */}
        <div className="relative mt-8" style={{ minHeight: 340 }}>
          {(() => {
            const CARD_PCT = 15 // approx card width as % of track, for collision test
            const laneRight: number[] = []
            return MARKERS.map((m) => {
              const clamped = Math.min(Math.max(pos(m.yearsAgo), 6), 94)
              const leftEdge = clamped - CARD_PCT / 2
              let lane = 0
              while (lane < laneRight.length && laneRight[lane] > leftEdge - 1) lane++
              laneRight[lane] = clamped + CARD_PCT / 2
              return { m, clamped, lane }
            })
          })().map(({ m, clamped, lane }, i) => {
            const top = lane * 66
            return (
              <motion.div
                key={m.label}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.06 }}
                className="absolute -translate-x-1/2"
                style={{ left: `${clamped}%`, top }}
              >
                <div className="flex flex-col items-center">
                  <div
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ background: m.color, boxShadow: `0 0 8px ${m.color}` }}
                  />
                  <div className="mt-1 w-40 rounded-lg border border-line bg-surface px-2.5 py-1.5 text-center">
                    <div className="text-xs font-semibold text-ink">{m.label}</div>
                    <div className="text-[11px] tabular-nums text-amber-glow">
                      {fmtYears(m.yearsAgo)}
                    </div>
                    {depth === 'technical' && (
                      <div className="mt-0.5 text-[10px] leading-tight text-ink-3">
                        {m.method}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>

        {!useLog && depth === 'beginner' && (
          <p className="mt-4 text-center text-xs text-ink-3">
            Notice how everything human crowds into the last flicker on the
            right. That squeeze is deep time — and radiometric dating is how we
            can see it at all.
          </p>
        )}
      </div>
    </Section>
  )
}
