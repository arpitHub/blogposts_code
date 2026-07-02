import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import Section from '../components/Section'
import { Depth, useDepth } from '../context/DepthContext'
import { WORDS, DIMS, cosine, type WordVec } from '../lib/embeddings'
import { FOCUS, INK, TOKEN_COLORS } from '../lib/palette'

const CLUSTER_COLOR: Record<WordVec['cluster'], string> = {
  royalty: TOKEN_COLORS[3], // violet
  animals: TOKEN_COLORS[1], // aqua
  food: TOKEN_COLORS[2], // yellow
  tech: TOKEN_COLORS[0], // blue
  feelings: TOKEN_COLORS[4], // red
}

const W = 640
const H = 420

function VectorStrip({ word, color }: { word: WordVec; color: string }) {
  const { depth } = useDepth()
  return (
    <div>
      <p className="mb-1 flex items-center gap-2 text-sm">
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
        <span className="font-mono font-medium text-ink">{word.word}</span>
      </p>
      <div className="flex items-end gap-1" aria-hidden="true">
        {word.v.map((val, i) => (
          <motion.div
            key={i}
            className="w-6 rounded-t-sm"
            style={{ background: color, opacity: 0.9 }}
            initial={false}
            animate={{ height: 4 + Math.abs(val) * 44, opacity: val < 0 ? 0.45 : 0.9 }}
            transition={{ type: 'spring', stiffness: 300, damping: 26 }}
            title={`${DIMS[i]}: ${val}`}
          />
        ))}
      </div>
      {depth === 'technical' && (
        <div className="mt-1 flex gap-1">
          {word.v.map((val, i) => (
            <span key={i} className="w-6 text-center font-mono text-[9px] text-ink-3" title={DIMS[i]}>
              {val.toFixed(1).replace('0.', '.').replace('-0', '-')}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Embeddings() {
  const { depth } = useDepth()
  const [selected, setSelected] = useState<string[]>(['king', 'queen'])
  const [hovered, setHovered] = useState<string | null>(null)

  const pick = (word: string) => {
    setSelected((prev) => {
      if (prev.includes(word)) return prev.filter((w) => w !== word)
      return [...prev.slice(-1), word] // keep at most 2, newest last
    })
  }

  const [a, b] = useMemo(() => {
    const find = (w?: string) => WORDS.find((x) => x.word === w)
    return [find(selected[0]), find(selected[1])]
  }, [selected])

  const sim = a && b ? cosine(a.v, b.v) : null
  const simLabel =
    sim === null ? null : sim > 0.7 ? 'very close in meaning' : sim > 0.45 ? 'related' : sim > 0.15 ? 'loosely related' : 'unrelated'

  return (
    <Section
      id="embeddings"
      kicker="Step 2 · Embeddings"
      title="Each token becomes a point on a map of meaning"
      lede={
        <Depth
          b="A token ID is just a number — it says nothing about meaning. So the model looks the token up in a giant table and gets back a list of numbers: its coordinates on a map where similar words live near each other. Click two words below and see how close they are."
          t={
            <>
              Each token ID indexes a learned embedding matrix, yielding a dense vector — typically{' '}
              <span className="font-mono">768</span> to <span className="font-mono">12,288</span>{' '}
              dimensions in production models. This demo uses 8 interpretable dimensions, projected to
              2D (the way t-SNE/UMAP plots flatten real embeddings). Select two words to compute their
              cosine similarity.
            </>
          }
        />
      }
    >
      <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
        <div className="rounded-2xl border border-hairline bg-surface p-4">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="h-auto w-full"
            role="img"
            aria-label="Scatterplot of word embeddings projected to 2D, with similar words clustered together"
          >
            {/* hairline grid */}
            {[0.25, 0.5, 0.75].map((t) => (
              <g key={t}>
                <line x1={t * W} y1={0} x2={t * W} y2={H} stroke={INK.grid} strokeWidth={1} />
                <line x1={0} y1={t * H} x2={W} y2={t * H} stroke={INK.grid} strokeWidth={1} />
              </g>
            ))}

            {/* similarity line between the two selected words */}
            {a && b && (
              <motion.line
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                x1={a.x * W}
                y1={a.y * H}
                x2={b.x * W}
                y2={b.y * H}
                stroke={FOCUS}
                strokeWidth={2}
                strokeDasharray="6 4"
              />
            )}

            {WORDS.map((w, wi) => {
              const isSel = selected.includes(w.word)
              const isHov = hovered === w.word
              const c = CLUSTER_COLOR[w.cluster]
              return (
                <g
                  key={w.word}
                  transform={`translate(${w.x * W}, ${w.y * H})`}
                  onClick={() => pick(w.word)}
                  onMouseEnter={() => setHovered(w.word)}
                  onMouseLeave={() => setHovered(null)}
                  className="cursor-pointer"
                >
                  {/* oversized invisible hit target */}
                  <circle r={16} fill="transparent" />
                  {isSel && (
                    <motion.circle
                      r={11}
                      fill="none"
                      stroke={FOCUS}
                      strokeWidth={2}
                      initial={{ scale: 0.4, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                    />
                  )}
                  <motion.circle
                    r={isHov || isSel ? 7 : 5.5}
                    fill={c}
                    stroke={INK.surface}
                    strokeWidth={2}
                    initial={{ scale: 0 }}
                    whileInView={{ scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ type: 'spring', stiffness: 300, damping: 20, delay: (wi % 8) * 0.05 }}
                  />
                  <text
                    y={-12}
                    textAnchor="middle"
                    fontSize={11}
                    fill={isSel || isHov ? INK.primary : INK.secondary}
                    fontFamily="ui-monospace, monospace"
                  >
                    {w.word}
                  </text>
                </g>
              )
            })}
          </svg>
          <p className="mt-2 px-1 text-xs text-ink-3">
            <Depth
              b="One dot per word · colors mark neighborhoods (animals, food, royalty, tech, feelings). Click any two dots."
              t="2D projection of 8-dim toy embeddings · axes are arbitrary projection directions, only distances matter. Click any two points."
            />
          </p>
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-2xl border border-hairline bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-ink-3">
              <Depth b="Under the hood: the numbers" t="The actual vectors" />
            </p>
            <div className="flex flex-col gap-4">
              {a && <VectorStrip word={a} color={CLUSTER_COLOR[a.cluster]} />}
              {b && <VectorStrip word={b} color={CLUSTER_COLOR[b.cluster]} />}
              {!a && !b && <p className="text-sm text-ink-3">Select words on the map…</p>}
            </div>
            {depth === 'technical' && (
              <p className="mt-3 text-[11px] leading-relaxed text-ink-3">
                8 toy dims: {DIMS.join(' · ')}
              </p>
            )}
          </div>

          {sim !== null && a && b && (
            <motion.div
              key={`${a.word}-${b.word}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl border border-hairline bg-surface p-4"
            >
              <p className="mb-1 text-xs font-medium uppercase tracking-wider text-ink-3">
                <Depth b="Similarity" t="Cosine similarity" />
              </p>
              <p className="text-3xl font-bold tabular-nums text-ink">{sim.toFixed(2)}</p>
              <p className="mt-1 text-sm text-ink-2">
                <Depth
                  b={
                    <>
                      <span className="font-mono">{a.word}</span> and{' '}
                      <span className="font-mono">{b.word}</span> are {simLabel}.
                    </>
                  }
                  t={
                    <>
                      cos(θ) = (a·b) / (‖a‖‖b‖) between{' '}
                      <span className="font-mono">{a.word}</span> and{' '}
                      <span className="font-mono">{b.word}</span> · 1 = same direction, 0 =
                      orthogonal.
                    </>
                  }
                />
              </p>
            </motion.div>
          )}
        </div>
      </div>
    </Section>
  )
}
