import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import Section from '../components/Section'
import { Depth, useDepth } from '../context/DepthContext'
import { INK } from '../lib/palette'

const LAYERS = 6
const DIMS = 8

// Deterministic pseudo-random stream so the vector's journey is stable.
function mulberry(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// vectorStates[l] = the token's 8-dim vector after layer l (index 0 = raw embedding).
function buildJourney(): number[][] {
  const rand = mulberry(42)
  const states: number[][] = []
  let v = Array.from({ length: DIMS }, () => rand() * 2 - 1)
  states.push([...v])
  for (let l = 0; l < LAYERS; l++) {
    // residual update: v <- normalize(v + delta), like x + Attn(x) + FFN(x)
    v = v.map((x) => x + (rand() * 2 - 1) * 0.9)
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) / Math.sqrt(DIMS)
    v = v.map((x) => x / norm)
    states.push([...v])
  }
  return states
}

// The token's color warms from muted gray to focus blue as it rises.
function layerColor(l: number): string {
  const t = l / LAYERS
  const from = [137, 135, 129] // ink muted
  const to = [57, 135, 229] // focus blue
  const c = from.map((f, i) => Math.round(f + (to[i] - f) * t))
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}

const BEGINNER_CAPTIONS = [
  'Raw embedding — just “which token is this?”',
  'Picks up spelling and neighboring words',
  'Learns its grammatical role in the sentence',
  'Links up with related words far away',
  'Absorbs the topic and tone of the passage',
  'Weighs what matters for what comes next',
  'A representation of this token in full context',
]

const TECH_CAPTIONS = [
  'x₀ = embedding + positional encoding',
  'x₁ = x₀ + Attn(LN(x₀)) + FFN(LN(·))',
  'x₂ — early layers: syntax, local structure',
  'x₃ — middle layers: semantic relations',
  'x₄ — long-range dependencies resolved',
  'x₅ — late layers: task/next-token features',
  'x₆ → LN → unembedding → logits',
]

export default function TransformerStack() {
  const { depth } = useDepth()
  const [layer, setLayer] = useState(0)
  const [playing, setPlaying] = useState(false)
  const journey = useMemo(buildJourney, [])
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (!playing) return
    timer.current = window.setInterval(() => {
      setLayer((l) => {
        if (l >= LAYERS) {
          setPlaying(false)
          return l
        }
        return l + 1
      })
    }, 850)
    return () => {
      if (timer.current) window.clearInterval(timer.current)
    }
  }, [playing])

  const run = () => {
    setLayer(0)
    // brief reset frame before the climb starts
    window.setTimeout(() => setPlaying(true), 60)
  }

  const v = journey[layer]
  const captions = depth === 'beginner' ? BEGINNER_CAPTIONS : TECH_CAPTIONS

  return (
    <Section
      id="stack"
      kicker="Step 4 · The Transformer stack"
      title="Do it again. And again. And again."
      lede={
        <Depth
          b="One round of attention gives each word a rough sense of its context. The transformer repeats the recipe — look around, then think it over — layer after layer. Each pass refines the word's internal picture. Press play and follow one token up the tower, or drag the slider."
          t={
            <>
              A transformer is N identical blocks: multi-head attention, then a position-wise
              feed-forward network, each wrapped in a residual connection and layer norm. GPT-2 small
              has 12 layers; the largest models exceed 100. Watch a single token’s hidden state (8 toy
              dims here, 768+ in practice) transform at each block.
            </>
          }
        />
      }
    >
      <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
        {/* the tower */}
        <div className="rounded-2xl border border-hairline bg-surface p-5">
          <div className="mb-4 flex items-center gap-3">
            <button
              onClick={run}
              className="rounded-full bg-tok-blue px-4 py-1.5 text-sm font-medium text-white transition-opacity hover:opacity-90"
            >
              {playing ? 'Rising…' : '▶ Send a token through'}
            </button>
            <input
              type="range"
              min={0}
              max={LAYERS}
              value={layer}
              onChange={(e) => {
                setPlaying(false)
                setLayer(Number(e.target.value))
              }}
              className="w-40"
              aria-label="Layer"
            />
            <span className="font-mono text-xs text-ink-3">
              {layer === 0 ? 'input' : `layer ${layer}/${LAYERS}`}
            </span>
          </div>

          <div className="relative flex flex-col-reverse gap-2">
            {Array.from({ length: LAYERS }, (_, i) => {
              const active = layer === i + 1
              const passed = layer > i + 1
              return (
                <div key={i} className="relative">
                  <motion.div
                    className="rounded-xl border p-3"
                    initial={false}
                    animate={{
                      borderColor: active ? layerColor(i + 1) : 'rgba(255,255,255,0.1)',
                      backgroundColor: active ? 'rgba(57,135,229,0.08)' : 'rgba(35,35,34,1)',
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <span className={`text-xs font-medium ${active || passed ? 'text-ink' : 'text-ink-3'}`}>
                        <Depth b={`Block ${i + 1}`} t={`Block ${i + 1} / ${LAYERS}`} />
                      </span>
                      {depth === 'technical' && (
                        <span className="font-mono text-[10px] text-ink-3">+residual · LayerNorm ×2</span>
                      )}
                    </div>
                    <div className="mt-2 flex gap-2">
                      <span
                        className={`rounded-md px-2 py-0.5 text-[11px] ${
                          active ? 'bg-tok-blue/25 text-ink' : 'bg-page text-ink-3'
                        }`}
                      >
                        <Depth b="look around" t="multi-head attention" />
                      </span>
                      <span
                        className={`rounded-md px-2 py-0.5 text-[11px] ${
                          active ? 'bg-tok-blue/25 text-ink' : 'bg-page text-ink-3'
                        }`}
                      >
                        <Depth b="think it over" t="feed-forward (4× width)" />
                      </span>
                    </div>
                  </motion.div>
                  {/* residual skip arrow, technical only */}
                  {depth === 'technical' && (
                    <svg
                      className="absolute -left-5 top-0 h-full w-4"
                      viewBox="0 0 16 60"
                      preserveAspectRatio="none"
                      aria-hidden="true"
                    >
                      <path d="M12 58 C 2 45, 2 15, 12 2" fill="none" stroke={INK.axis} strokeWidth={1.5} />
                      <path d="M8 8 L12 2 L14 9" fill="none" stroke={INK.axis} strokeWidth={1.5} />
                    </svg>
                  )}
                </div>
              )
            })}

            {/* the traveling token */}
            <motion.div
              className="pointer-events-none absolute -right-3 z-10 flex h-7 w-7 items-center justify-center rounded-full text-[10px] font-bold text-white shadow-lg"
              initial={false}
              animate={{
                bottom: `${(layer / LAYERS) * 92}%`,
                backgroundColor: layerColor(layer),
                scale: playing ? [1, 1.25, 1] : 1,
              }}
              transition={{ type: 'spring', stiffness: 120, damping: 16 }}
            >
              tok
            </motion.div>
          </div>
          <p className="mt-3 text-xs text-ink-3">
            <Depth
              b="The dot is one token's internal picture — watch its color shift as each block refines it."
              t="Real stacks: GPT-2 12 blocks · GPT-3 96 · Llama-3-70B 80. Same block, different weights."
            />
          </p>
        </div>

        {/* the vector inspector */}
        <div className="flex flex-col gap-4">
          <div className="rounded-2xl border border-hairline bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-ink-3">
              <Depth b="The token's numbers, live" t={`hidden state h${layer} (toy 8-dim)`} />
            </p>
            <div className="flex h-28 items-center gap-1.5">
              {v.map((val, i) => {
                // After the residual updates, dims range roughly [-2.5, 2.5];
                // scale so a bar never escapes its card.
                const mag = Math.min(1, Math.abs(val) / 2.2)
                const h = 6 + mag * 44
                return (
                  <motion.div
                    key={i}
                    className="w-7 rounded-sm"
                    initial={false}
                    animate={{
                      height: h,
                      y: val < 0 ? h / 2 : -h / 2,
                      backgroundColor: layerColor(layer),
                    }}
                    transition={{ type: 'spring', stiffness: 260, damping: 24 }}
                  />
                )
              })}
            </div>
            <div className="mt-1 border-t border-hairline" aria-hidden="true" />
            <p className="mt-2 text-[11px] text-ink-3">
              <Depth b="Bars above/below the line = each dimension's value" t="8 of d_model dims · sign around zero" />
            </p>
          </div>

          <motion.div
            key={`${layer}-${depth}`}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-hairline bg-surface p-4"
          >
            <p className="text-sm leading-relaxed text-ink-2">
              <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: layerColor(layer) }} />
              {captions[layer]}
            </p>
          </motion.div>
        </div>
      </div>
    </Section>
  )
}
