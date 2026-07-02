import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Section from '../components/Section'
import { Depth, useDepth } from '../context/DepthContext'
import { SENTENCE, N, SCORES, WEIGHTS } from '../lib/attention'
import { FOCUS, INK, seqColor } from '../lib/palette'

const W = 720
const ARC_H = 150
const TOK_H = 44

function tokenX(i: number): number {
  return ((i + 0.5) / N) * W
}

function ArcDiagram({ selected, onSelect }: { selected: number; onSelect: (i: number) => void }) {
  const weights = WEIGHTS[selected]
  return (
    <svg
      viewBox={`0 0 ${W} ${ARC_H + TOK_H + 22}`}
      className="h-auto w-full"
      role="img"
      aria-label={`Attention arcs from the token "${SENTENCE[selected]}" to the tokens it attends to`}
    >
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* arcs from the selected token to every other token */}
      <AnimatePresence mode="popLayout">
        {weights.map((w, j) => {
          if (j === selected || w < 0.02) return null
          const x1 = tokenX(selected)
          const x2 = tokenX(j)
          const lift = Math.min(ARC_H - 10, 40 + Math.abs(x2 - x1) * 0.35)
          const d = `M ${x1} ${ARC_H} Q ${(x1 + x2) / 2} ${ARC_H - lift} ${x2} ${ARC_H}`
          return (
            <motion.path
              key={`${selected}-${j}`}
              d={d}
              fill="none"
              stroke={FOCUS}
              strokeLinecap="round"
              filter="url(#glow)"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: 0.15 + w * 1.6 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.45, ease: 'easeOut' }}
              style={{ strokeWidth: 1 + w * 9 }}
            />
          )
        })}
      </AnimatePresence>

      {/* token chips */}
      {SENTENCE.map((tok, i) => {
        const w = weights[i]
        const isSel = i === selected
        const x = tokenX(i)
        return (
          <g key={i} transform={`translate(${x}, ${ARC_H + 6})`} onClick={() => onSelect(i)} className="cursor-pointer">
            <motion.rect
              x={-W / N / 2 + 4}
              width={W / N - 8}
              height={TOK_H - 12}
              rx={7}
              initial={false}
              animate={{
                fill: isSel ? FOCUS : `rgba(57, 135, 229, ${(0.12 + w * 1.2) * (w > 0.03 ? 1 : 0.25)})`,
                stroke: isSel ? FOCUS : w > 0.1 ? FOCUS : INK.axis,
              }}
              strokeWidth={1.5}
            />
            <text
              y={TOK_H / 2 - 1}
              textAnchor="middle"
              fontSize={13}
              fontFamily="ui-monospace, monospace"
              fill={isSel ? '#fff' : w > 0.15 ? '#fff' : INK.secondary}
            >
              {tok}
            </text>
            {!isSel && w >= 0.1 && (
              <text y={TOK_H + 8} textAnchor="middle" fontSize={10} fill={INK.muted} fontFamily="ui-monospace, monospace">
                {(w * 100).toFixed(0)}%
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

function QKVPanel({ selected }: { selected: number }) {
  const scores = SCORES[selected]
  const weights = WEIGHTS[selected]
  const top = [...weights.keys()].sort((a, b) => weights[b] - weights[a]).slice(0, 4)
  return (
    <div className="rounded-2xl border border-hairline bg-surface p-4">
      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-3">
        Behind the arcs: query · key · value
      </p>
      <p className="mb-3 font-mono text-sm text-ink-2">
        Attention(Q, K, V) = softmax(QKᵀ / √d<sub>k</sub>) · V
      </p>
      <p className="mb-2 text-sm leading-relaxed text-ink-2">
        The selected token’s <span className="text-tok-blue">query</span> vector is dotted with every
        token’s <span className="text-tok-blue">key</span> vector, producing one score per pair.
        Softmax turns the scores into weights that sum to 1, which mix the{' '}
        <span className="text-tok-blue">value</span> vectors into a new representation.
      </p>
      <table className="w-full text-left font-mono text-xs">
        <thead>
          <tr className="text-ink-3">
            <th className="py-1 pr-2 font-normal">key j</th>
            <th className="py-1 pr-2 font-normal">q·kⱼ/√d</th>
            <th className="py-1 font-normal">softmax → weight</th>
          </tr>
        </thead>
        <tbody>
          {top.map((j) => (
            <tr key={j} className="border-t border-hairline text-ink-2">
              <td className="py-1 pr-2">{SENTENCE[j]}</td>
              <td className="py-1 pr-2 tabular-nums">{scores[j].toFixed(1)}</td>
              <td className="py-1 tabular-nums">{weights[j].toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Heatmap({ selected, onSelect }: { selected: number; onSelect: (i: number) => void }) {
  const [hover, setHover] = useState<[number, number] | null>(null)
  return (
    <div className="rounded-2xl border border-hairline bg-surface p-4">
      <p className="mb-3 text-xs font-medium uppercase tracking-wider text-ink-3">
        The full attention matrix — rows are queries, columns are keys
      </p>
      <div className="overflow-x-auto">
        <div className="inline-grid gap-px" style={{ gridTemplateColumns: `72px repeat(${N}, 26px)` }}>
          <span />
          {SENTENCE.map((t, j) => (
            <span key={j} className="truncate pb-1 text-center font-mono text-[9px] text-ink-3" title={t}>
              {t.length > 4 ? t.slice(0, 3) + '…' : t}
            </span>
          ))}
          {SENTENCE.map((rowTok, i) => (
            <div key={i} className="contents">
              <button
                onClick={() => onSelect(i)}
                className={`truncate pr-2 text-right font-mono text-[10px] leading-[26px] transition-colors ${
                  i === selected ? 'text-tok-blue' : 'text-ink-3 hover:text-ink-2'
                }`}
              >
                {rowTok}
              </button>
              {WEIGHTS[i].map((w, j) => (
                <button
                  key={j}
                  onClick={() => onSelect(i)}
                  onMouseEnter={() => setHover([i, j])}
                  onMouseLeave={() => setHover(null)}
                  aria-label={`${rowTok} attends to ${SENTENCE[j]} with weight ${w.toFixed(2)}`}
                  className="relative h-[26px] w-[26px] rounded-[3px]"
                  style={{
                    background: seqColor(Math.min(1, w * 1.6)),
                    outline: i === selected ? `1px solid ${INK.secondary}` : 'none',
                    outlineOffset: -1,
                  }}
                >
                  {hover && hover[0] === i && hover[1] === j && (
                    <span className="pointer-events-none absolute -top-9 left-1/2 z-10 -translate-x-1/2 whitespace-nowrap rounded-md border border-hairline bg-page px-2 py-1 font-mono text-[10px] text-ink shadow-lg">
                      {rowTok} → {SENTENCE[j]}: {w.toFixed(2)}
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>
      <p className="mt-2 text-xs text-ink-3">
        Brighter = more attention · each row sums to 1 · click a row to select that query token above.
      </p>
    </div>
  )
}

export default function Attention() {
  const { depth } = useDepth()
  const [selected, setSelected] = useState(7) // "it" — the fun one

  return (
    <Section
      id="attention"
      kicker="Step 3 · Attention"
      title="Every word looks around the sentence"
      lede={
        <Depth
          b={
            <>
              Here's the trick that made modern AI work: as the model reads, each word gets to look at
              every other word and decide which ones matter to it. Click any token below — the glowing
              lines show where it's looking. Try{' '}
              <button
                onClick={() => setSelected(7)}
                className="font-mono text-tok-blue underline decoration-dotted underline-offset-2"
              >
                “it”
              </button>
              : the model figures out that “it” means the ball, not the robot.
            </>
          }
          t={
            <>
              Self-attention: every token emits a query, key, and value vector (linear projections of
              its embedding). Token i’s new representation is a weighted sum of all value vectors,
              with weights softmax(q<sub>i</sub>·k<sub>j</sub>/√d<sub>k</sub>). Click a token to
              inspect its row of the attention matrix. Note how “it” places ~half its weight on
              “ball” — coreference resolved by dot products.
            </>
          }
        />
      }
    >
      <div className="rounded-2xl border border-hairline bg-surface p-4 sm:p-6">
        <ArcDiagram selected={selected} onSelect={setSelected} />
        <p className="mt-3 text-center text-sm text-ink-3">
          <span className="font-mono text-tok-blue">{SENTENCE[selected]}</span>{' '}
          <Depth
            b="is looking at the highlighted words — thicker line, more importance."
            t={`· row ${selected} of the attention matrix · line width ∝ softmax weight`}
          />
        </p>
      </div>

      {depth === 'technical' && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 grid gap-6 lg:grid-cols-2"
        >
          <QKVPanel selected={selected} />
          <Heatmap selected={selected} onSelect={setSelected} />
        </motion.div>
      )}

      {depth === 'beginner' && (
        <p className="mt-6 max-w-2xl text-sm leading-relaxed text-ink-3">
          Real models do this dozens of times in parallel (different “heads” looking for different
          kinds of relationships — grammar, names, tone) for every word at once. Flip to{' '}
          <span className="text-ink-2">Technical</span> to see the actual matrix.
        </p>
      )}
    </Section>
  )
}
