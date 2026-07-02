import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Section } from '../components/Section'
import { Btn, EqCard } from '../components/ui'
import { D, useDepth } from '../lib/depth'
import { CHAINS, dwellSeconds } from '../lib/chains'
import { fmtSeconds } from '../lib/format'

const COLS = 5
const CELL_W = 132
const CELL_H = 104
const PAD_X = 60
const PAD_Y = 46
const HOP_MS = 450

/** serpentine layout: left→right, then right→left on the next row */
function nodePos(i: number) {
  const row = Math.floor(i / COLS)
  const colRaw = i % COLS
  const col = row % 2 === 0 ? colRaw : COLS - 1 - colRaw
  return { x: PAD_X + col * CELL_W, y: PAD_Y + row * CELL_H }
}

export function ChainsSection() {
  const depth = useDepth()
  const [chainId, setChainId] = useState('u238')
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const chain = CHAINS.find((c) => c.id === chainId)!
  const step = chain.steps[idx]
  const isStable = step.halfLifeS === undefined
  const dwell = isStable ? 0 : dwellSeconds(step.halfLifeS!)

  // advance the token after the current node's (log-mapped) waiting time
  useEffect(() => {
    if (!playing) return
    if (isStable) {
      setPlaying(false)
      return
    }
    const t = setTimeout(() => setIdx((i) => i + 1), dwell * 1000 + HOP_MS)
    return () => clearTimeout(t)
  }, [playing, idx, chainId, isStable, dwell])

  const selectChain = (id: string) => {
    setChainId(id)
    setIdx(0)
    setPlaying(true)
  }

  const rows = Math.ceil(chain.steps.length / COLS)
  const width = PAD_X * 2 + (COLS - 1) * CELL_W
  const height = PAD_Y * 2 + (rows - 1) * CELL_H + 16
  const pos = nodePos(idx)

  const { minStep, maxStep } = useMemo(() => {
    const timed = chain.steps.filter((s) => s.halfLifeS !== undefined)
    return {
      minStep: timed.reduce((a, b) => (a.halfLifeS! < b.halfLifeS! ? a : b)),
      maxStep: timed.reduce((a, b) => (a.halfLifeS! > b.halfLifeS! ? a : b)),
    }
  }, [chain])

  return (
    <Section id="chains" kicker="03 · The machinery" title="It's not one flip — it's a relay race">
      <p className="mb-6 max-w-3xl text-[15px] leading-relaxed text-ink-2">
        <D
          b={
            <>
              A uranium atom doesn’t become lead in one hop. It passes the baton
              through a dozen short-lived atoms — some hold it for ages, some
              for less than a heartbeat. Press play and follow one atom’s whole
              journey. The clock is set almost entirely by the very first,
              slowest runner.
            </>
          }
          t={
            <>
              Within one chain, half-lives span ~23 orders of magnitude — from{' '}
              {fmtSeconds(minStep.halfLifeS!)} ({minStep.isotope}) to{' '}
              {fmtSeconds(maxStep.halfLifeS!)} ({maxStep.isotope}). Because
              every intermediate is negligibly short-lived next to the parent,
              the chain is in secular equilibrium and behaves as a single
              parent→daughter clock with the parent’s λ. Waiting times below are
              log-compressed to stay watchable.
            </>
          }
        />
      </p>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        {CHAINS.map((c) => (
          <button
            key={c.id}
            role="tab"
            aria-selected={chainId === c.id}
            onClick={() => selectChain(c.id)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              chainId === c.id
                ? 'bg-amber-glow text-void'
                : 'border border-line bg-panel text-ink-2 hover:text-ink'
            }`}
          >
            {c.name}
          </button>
        ))}
        <span className="ml-1 hidden text-xs text-ink-3 sm:inline">{chain.note}</span>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,8fr)_minmax(0,4fr)]">
        <div className="overflow-x-auto rounded-xl border border-line bg-panel p-2">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="h-auto w-full min-w-[560px]"
            role="img"
            aria-label={`Decay chain from ${chain.steps[0].isotope} to ${chain.end}`}
          >
            {/* edges */}
            {chain.steps.slice(0, -1).map((s, i) => {
              const a = nodePos(i)
              const b = nodePos(i + 1)
              const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
              const done = i < idx
              return (
                <g key={i}>
                  <line
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={done ? '#3987e5' : '#232330'}
                    strokeWidth={done ? 1.8 : 1.2}
                  />
                  <text
                    x={mid.x + (a.y === b.y ? 0 : 12)}
                    y={mid.y + (a.y === b.y ? -7 : 4)}
                    textAnchor="middle"
                    fontSize="11"
                    fill={s.mode === 'α' ? '#c98500' : '#3987e5'}
                  >
                    {s.mode}
                  </text>
                </g>
              )
            })}

            {/* nodes */}
            {chain.steps.map((s, i) => {
              const p = nodePos(i)
              const state = i < idx ? 'past' : i === idx ? 'now' : 'future'
              const stable = s.halfLifeS === undefined
              return (
                <g
                  key={s.isotope}
                  onClick={() => {
                    setIdx(i)
                    setPlaying(!stable)
                  }}
                  className="cursor-pointer"
                  role="button"
                  aria-label={`Jump to ${s.isotope}`}
                >
                  <circle
                    cx={p.x} cy={p.y} r={17}
                    fill={
                      state === 'now'
                        ? '#f5b846'
                        : state === 'past'
                          ? '#1d3f66'
                          : stable
                            ? '#12121a'
                            : '#12121a'
                    }
                    stroke={
                      state === 'now'
                        ? '#c98500'
                        : state === 'past' || stable
                          ? '#3987e5'
                          : '#34343f'
                    }
                    strokeWidth={state === 'now' ? 2 : 1.2}
                    opacity={state === 'future' && !stable ? 0.85 : 1}
                  />
                  {/* dwell progress ring on the active node */}
                  {state === 'now' && !stable && playing && (
                    <motion.circle
                      key={`ring-${chainId}-${i}`}
                      cx={p.x} cy={p.y} r={22}
                      fill="none"
                      stroke="#f5b846"
                      strokeWidth={2.5}
                      strokeDasharray={2 * Math.PI * 22}
                      initial={{ strokeDashoffset: 2 * Math.PI * 22 }}
                      animate={{ strokeDashoffset: 0 }}
                      transition={{ duration: dwell, ease: 'linear', delay: HOP_MS / 1000 }}
                      transform={`rotate(-90 ${p.x} ${p.y})`}
                    />
                  )}
                  <text
                    x={p.x} y={p.y - 26}
                    textAnchor="middle" fontSize="12" fontWeight="600"
                    fill={state === 'now' ? '#f5b846' : stable ? '#7db5f0' : '#b9b7c0'}
                  >
                    {s.isotope}
                  </text>
                  <text
                    x={p.x} y={p.y + 36}
                    textAnchor="middle" fontSize="10"
                    fill="#8a8894"
                  >
                    {stable ? 'stable' : fmtSeconds(s.halfLifeS!)}
                  </text>
                </g>
              )
            })}

            {/* travelling atom */}
            <motion.circle
              r={7}
              fill="#f5b846"
              stroke="#0a0a0f"
              strokeWidth={2}
              initial={{ cx: nodePos(0).x, cy: nodePos(0).y }}
              animate={{ cx: pos.x, cy: pos.y }}
              transition={{ duration: HOP_MS / 1000, ease: 'easeInOut' }}
            />
          </svg>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-2">
            <Btn primary onClick={() => setPlaying(!playing)} disabled={isStable && !playing}>
              {playing ? '❚❚ Pause' : '▶ Play'}
            </Btn>
            <Btn
              onClick={() => {
                setIdx(0)
                setPlaying(false)
              }}
            >
              ↺ Restart
            </Btn>
          </div>

          <div className="rounded-lg border border-line bg-panel px-4 py-3">
            <div className="text-[11px] uppercase tracking-wide text-ink-3">
              current isotope
            </div>
            <div className="text-2xl font-bold text-amber-glow">{step.isotope}</div>
            <div className="mt-1 text-sm text-ink-2">
              {isStable ? (
                <>
                  Stable. The baton stops here — this lead atom will outlast the
                  stars.
                </>
              ) : (
                <>
                  waits <strong className="text-ink">{fmtSeconds(step.halfLifeS!)}</strong>{' '}
                  (half-life), then {step.mode === 'α' ? 'spits out an α particle' : 'converts via β decay'}
                </>
              )}
            </div>
          </div>

          {depth === 'technical' ? (
            <EqCard
              note={
                <>
                  α emission: A−4, Z−2 (banked helium — Rutherford’s clock).
                  β⁻ emission: Z+1, A unchanged. Bi-212/Bi-214 branchings are
                  simplified to the dominant path.
                </>
              }
            >
              t½ spans {fmtSeconds(minStep.halfLifeS!)} → {fmtSeconds(maxStep.halfLifeS!)}{' '}
              in one chain
            </EqCard>
          ) : (
            <p className="text-sm leading-relaxed text-ink-3">
              Click any atom in the chain to drop the baton there. Notice the
              wait times: the first runner holds it for billions of years, one
              in the middle for a fraction of a millisecond — yet every uranium
              atom that starts the race finishes at the same stable lead.
            </p>
          )}
        </div>
      </div>
    </Section>
  )
}
