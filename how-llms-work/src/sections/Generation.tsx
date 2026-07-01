import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import Section from '../components/Section'
import { Depth, useDepth } from '../context/DepthContext'
import { applyTemperature, candidatesFor, sample } from '../lib/generation'
import { FOCUS, INK } from '../lib/palette'

const HIGHLIGHT = '#cde2fb' // lightest step of the same blue ramp — sampled bar

export default function Generation() {
  const { depth } = useDepth()
  const [prompt, setPrompt] = useState('The cat sat on the')
  const [temperature, setTemperature] = useState(0.8)
  const [sampled, setSampled] = useState<string | null>(null)

  const dist = useMemo(
    () => applyTemperature(candidatesFor(prompt), temperature),
    [prompt, temperature],
  )

  const doSample = () => {
    const i = sample(dist)
    const tok = dist[i].token
    setSampled(tok)
    setPrompt((p) => `${p.trimEnd()} ${tok}`)
  }

  const data = dist.map((d) => ({ ...d, pct: d.p * 100 }))

  return (
    <Section
      id="generation"
      kicker="Step 5 · Generation"
      title="It's all a guess about the next token"
      lede={
        <Depth
          b="After all that reading, the model does one deceptively simple thing: it scores every token it knows on how well it would come next, then rolls a weighted die. Then it does it again for the token after that. Type a prompt, drag the temperature, and roll the die yourself."
          t={
            <>
              The final hidden state is multiplied by the unembedding matrix to give one logit per
              vocab entry; softmax(logits/T) is the sampling distribution. Temperature T rescales the
              logits: T→0 approaches argmax (greedy), T&gt;1 flattens toward uniform. (Distribution
              below is mocked — a lookup on the last word — but the softmax and sampling are real.)
            </>
          }
        />
      }
    >
      <div className="rounded-2xl border border-hairline bg-surface p-5 sm:p-6">
        <label htmlFor="gen-input" className="mb-2 block text-xs font-medium uppercase tracking-wider text-ink-3">
          Prompt
        </label>
        <div className="flex flex-col gap-3 sm:flex-row">
          <input
            id="gen-input"
            type="text"
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value)
              setSampled(null)
            }}
            spellCheck={false}
            className="min-w-0 flex-1 rounded-lg border border-hairline bg-page px-4 py-3 text-base text-ink outline-none transition-colors focus:border-tok-blue"
            placeholder="Type a prompt…"
          />
          <button
            onClick={doSample}
            className="shrink-0 rounded-lg bg-tok-blue px-5 py-3 text-sm font-medium text-white transition-opacity hover:opacity-90"
          >
            <Depth b="🎲 Roll the die" t="Sample next token" />
          </button>
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-4">
          <label htmlFor="temp" className="text-sm text-ink-2">
            Temperature
          </label>
          <input
            id="temp"
            type="range"
            min={0.1}
            max={2}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            className="w-48"
          />
          <span className="font-mono text-sm tabular-nums text-ink">{temperature.toFixed(2)}</span>
          <span className="text-xs text-ink-3">
            <Depth
              b={temperature < 0.5 ? 'careful — almost always picks the favorite' : temperature > 1.3 ? 'wild — long shots get real chances' : 'balanced'}
              t={temperature < 0.5 ? 'near-greedy: p mass concentrates on argmax' : temperature > 1.3 ? 'high entropy: distribution flattens' : 'softmax(logits / T)'}
            />
          </span>
        </div>

        <div className="mt-6 h-72 w-full" aria-label="Probability of each candidate next token">
          <ResponsiveContainer>
            <BarChart data={data} margin={{ top: 24, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid vertical={false} stroke={INK.grid} />
              <XAxis
                dataKey="token"
                tickLine={false}
                axisLine={{ stroke: INK.axis }}
                tick={{ fill: INK.secondary, fontSize: 12, fontFamily: 'ui-monospace, monospace' }}
                interval={0}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tick={{ fill: INK.muted, fontSize: 11 }}
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                width={40}
              />
              <Tooltip
                cursor={{ fill: 'rgba(255,255,255,0.04)' }}
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null
                  const d = payload[0].payload as (typeof data)[number]
                  return (
                    <div className="rounded-md border border-hairline bg-page px-3 py-2 font-mono text-xs text-ink shadow-lg">
                      <p className="font-semibold">“{d.token}”</p>
                      <p className="text-ink-2">p = {(d.p * 100).toFixed(1)}%</p>
                      {depth === 'technical' && <p className="text-ink-3">logit = {d.logit.toFixed(1)}</p>}
                    </div>
                  )
                }}
              />
              <Bar dataKey="pct" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={350}>
                {data.map((d) => (
                  <Cell
                    key={d.token}
                    fill={d.token === sampled ? HIGHLIGHT : FOCUS}
                    stroke={d.token === sampled ? '#fff' : 'none'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-2 flex items-center justify-between gap-4">
          <p className="text-xs text-ink-3">
            <Depth
              b="Each bar = one candidate word's chance of being picked next. Hover for exact odds."
              t="softmax over the top-8 mocked logits · hover a bar for p and logit."
            />
          </p>
          {sampled && (
            <motion.p
              key={prompt}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-sm text-ink-2"
            >
              picked <span className="rounded bg-tok-blue/20 px-1.5 py-0.5 font-mono text-ink">{sampled}</span>
              <Depth b=" — and now it starts over on the longer prompt." t=" — appended; the next forward pass conditions on it." />
            </motion.p>
          )}
        </div>
      </div>

      <p className="mx-auto mt-12 max-w-2xl text-center text-sm leading-relaxed text-ink-3">
        <Depth
          b="That's the whole loop: chop into tokens, place them on the map of meaning, let every word look around, refine it layer by layer, guess the next token — then repeat with one more token. Everything an LLM writes comes from this loop running very, very fast."
          t="tokenize → embed → N × (attention + FFN) → logits → sample → append → repeat. Autoregressive decoding, one token per forward pass (modulo KV-caching and speculative tricks)."
        />
      </p>
    </Section>
  )
}
