import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Section from '../components/Section'
import { Depth, useDepth } from '../context/DepthContext'
import { DOCS, KEEP, QUERIES, RERANK_FLOOR, rerank, retrieve, type Query } from '../lib/rag'
import { FOCUS, INK, TOKEN_COLORS } from '../lib/palette'

const DOC_COLOR = TOKEN_COLORS[1] // aqua — documents are one entity type
const W = 640
const H = 340

type Phase = 'idle' | 'embed' | 'search' | 'rerank' | 'assemble' | 'generate'
const ORDER: Phase[] = ['idle', 'embed', 'search', 'rerank', 'assemble', 'generate']

function after(phase: Phase, target: Phase): boolean {
  return ORDER.indexOf(phase) >= ORDER.indexOf(target)
}

function VectorMap({ query, phase, retrievedIds, candidateIds }: {
  query: Query
  phase: Phase
  retrievedIds: string[]
  candidateIds: string[]
}) {
  const qx = query.x * W
  const qy = query.y * H
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label="The query being embedded into the vector store and matched against nearby documents"
    >
      <defs>
        <filter id="rag-glow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {[0.25, 0.5, 0.75].map((t) => (
        <g key={t}>
          <line x1={t * W} y1={0} x2={t * W} y2={H} stroke={INK.grid} strokeWidth={1} />
          <line x1={0} y1={t * H} x2={W} y2={t * H} stroke={INK.grid} strokeWidth={1} />
        </g>
      ))}

      {/* similarity lines out to the first-stage candidates */}
      <AnimatePresence>
        {after(phase, 'search') &&
          candidateIds.map((id) => {
            const d = DOCS.find((x) => x.id === id)!
            const kept = retrievedIds.includes(id) && after(phase, 'assemble')
            return (
              <motion.line
                key={`${query.id}-${id}`}
                x1={qx}
                y1={qy}
                x2={d.x * W}
                y2={d.y * H}
                stroke={FOCUS}
                strokeWidth={kept ? 2.5 : 1.5}
                strokeDasharray="5 5"
                filter="url(#rag-glow)"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: kept ? 0.9 : 0.4 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
              />
            )
          })}
      </AnimatePresence>

      {/* document chunks */}
      {DOCS.map((d) => {
        const isCandidate = after(phase, 'search') && candidateIds.includes(d.id)
        const isKept = after(phase, 'assemble') && retrievedIds.includes(d.id)
        return (
          <g key={d.id} transform={`translate(${d.x * W}, ${d.y * H})`}>
            {isKept && (
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
              r={isCandidate ? 7 : 5.5}
              fill={DOC_COLOR}
              stroke={INK.surface}
              strokeWidth={2}
              animate={{ opacity: !after(phase, 'search') || isCandidate ? 1 : 0.35 }}
            />
            <text
              y={-11}
              textAnchor="middle"
              fontSize={10.5}
              fill={isCandidate ? INK.primary : INK.muted}
              fontFamily="ui-monospace, monospace"
            >
              {d.label}
            </text>
          </g>
        )
      })}

      {/* the embedded query flies in from the top-left corner */}
      {after(phase, 'embed') && (
        <motion.g
          initial={{ x: 30, y: 24, scale: 0.4, opacity: 0 }}
          animate={{ x: qx, y: qy, scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 90, damping: 16 }}
        >
          {phase === 'search' && (
            <motion.circle
              r={12}
              fill="none"
              stroke={FOCUS}
              strokeWidth={1.5}
              initial={{ scale: 0.3, opacity: 0.9 }}
              animate={{ scale: 4.5, opacity: 0 }}
              transition={{ duration: 1.1, repeat: Infinity, ease: 'easeOut' }}
            />
          )}
          <circle r={7.5} fill={FOCUS} filter="url(#rag-glow)" />
          <text y={22} textAnchor="middle" fontSize={11} fill={INK.primary} fontFamily="ui-monospace, monospace">
            query
          </text>
        </motion.g>
      )}
    </svg>
  )
}

export default function Rag() {
  const { depth } = useDepth()
  const [query, setQuery] = useState<Query>(QUERIES[0])
  const [hybrid, setHybrid] = useState(false)
  const [phase, setPhase] = useState<Phase>('idle')
  const timers = useRef<number[]>([])

  const candidates = useMemo(() => retrieve(query, depth === 'technical' && hybrid), [query, hybrid, depth])
  const reranked = useMemo(
    () => (depth === 'technical' ? rerank(candidates) : candidates),
    [candidates, depth],
  )
  // The re-ranker both reorders and drops anything below the relevance floor,
  // so keyword bait that survived first-stage hybrid scoring never reaches the prompt.
  const kept =
    depth === 'technical'
      ? reranked.filter((d) => d.rerank >= RERANK_FLOOR).slice(0, KEEP)
      : reranked.slice(0, KEEP)
  const showRerank = depth === 'technical' && after(phase, 'rerank')
  const results = showRerank ? reranked : candidates

  const clearTimers = () => {
    timers.current.forEach(window.clearTimeout)
    timers.current = []
  }

  const run = () => {
    clearTimers()
    setPhase('embed')
    const steps: Array<[Phase, number]> =
      depth === 'technical'
        ? [['search', 800], ['rerank', 2200], ['assemble', 3600], ['generate', 4800]]
        : [['search', 800], ['assemble', 2400], ['generate', 3600]]
    for (const [p, t] of steps) timers.current.push(window.setTimeout(() => setPhase(p), t))
  }

  // Changing the query, mode, or retrieval strategy resets the pipeline.
  useEffect(() => {
    clearTimers()
    setPhase('idle')
  }, [query, hybrid, depth])
  useEffect(() => clearTimers, [])

  return (
    <Section
      id="rag"
      kicker="Step 6 · Retrieval-augmented generation"
      title="Open-book beats closed-book"
      lede={
        <Depth
          b="Everything so far happens from memory — a closed-book exam. RAG lets the model look things up first: your question becomes a point on the map of meaning from Step 2, the nearest documents are pulled off the shelf, and they're pasted into the prompt before the model answers. Pick a question and run it."
          t={
            <>
              The query is embedded into the same vector space as pre-chunked documents; nearest
              neighbors by cosine similarity come back as context. Toggle{' '}
              <span className="font-mono">hybrid</span> to mix in BM25 keyword scores — then watch the
              cross-encoder re-ranker rescue precision by demoting keyword bait the first stage let
              through.
            </>
          }
        />
      }
    >
      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="rounded-2xl border border-hairline bg-surface p-4">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {QUERIES.map((q) => (
              <button
                key={q.id}
                onClick={() => setQuery(q)}
                className={`rounded-full border px-3 py-1 text-xs transition-colors ${
                  query.id === q.id
                    ? 'border-tok-blue bg-tok-blue/15 text-ink'
                    : 'border-hairline text-ink-3 hover:border-tok-blue hover:text-ink-2'
                }`}
              >
                {q.text}
              </button>
            ))}
          </div>

          <VectorMap
            query={query}
            phase={phase}
            retrievedIds={kept.map((d) => d.id)}
            candidateIds={candidates.map((d) => d.id)}
          />

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <button
              onClick={run}
              className="rounded-full bg-tok-blue px-4 py-1.5 text-sm font-medium text-white transition-opacity hover:opacity-90"
            >
              {phase === 'idle' ? '🔍 Retrieve & answer' : '↻ Run again'}
            </button>
            {depth === 'technical' && (
              <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-2">
                <span className={hybrid ? 'text-ink-3' : 'text-ink'}>vector only</span>
                <button
                  role="switch"
                  aria-checked={hybrid}
                  onClick={() => setHybrid((h) => !h)}
                  className={`relative h-5 w-9 rounded-full transition-colors ${hybrid ? 'bg-tok-blue' : 'bg-axis'}`}
                >
                  <span
                    className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${hybrid ? 'left-[18px]' : 'left-0.5'}`}
                  />
                </button>
                <span className={hybrid ? 'text-ink' : 'text-ink-3'}>hybrid (+BM25)</span>
              </label>
            )}
          </div>
          <p className="mt-2 text-xs text-ink-3">
            <Depth
              b="Teal dots are documents in the library · the blue dot is your question landing among them."
              t={`${DOCS.length} chunks in the store · first stage keeps top-${candidates.length} · re-ranker keeps ≤${KEEP} with relevance ≥ ${RERANK_FLOOR}.`}
            />
          </p>
        </div>

        <div className="flex flex-col gap-4">
          {/* ranked results */}
          <div className="rounded-2xl border border-hairline bg-surface p-4">
            <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-3">
              {showRerank ? (
                <Depth b="Best matches" t="After re-ranking (cross-encoder)" />
              ) : (
                <Depth b="Best matches" t={hybrid ? 'First stage · 0.55·vec + 0.45·bm25' : 'First stage · cosine similarity'} />
              )}
            </p>
            {after(phase, 'search') ? (
              <ul className="flex flex-col gap-1.5">
                {results.map((d) => {
                  const dropped = showRerank && !kept.some((k) => k.id === d.id)
                  return (
                    <motion.li
                      key={d.id}
                      layout
                      transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                      className={`flex items-center gap-2 rounded-lg border border-hairline px-2.5 py-1.5 text-xs ${
                        dropped ? 'opacity-40' : ''
                      }`}
                    >
                      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: DOC_COLOR }} />
                      <span className={`flex-1 truncate font-mono ${dropped ? 'line-through' : 'text-ink-2'}`}>
                        {d.label}
                      </span>
                      {depth === 'technical' ? (
                        <span className="font-mono tabular-nums text-ink-3">
                          {showRerank
                            ? `r=${d.rerank.toFixed(2)}`
                            : hybrid
                              ? `v=${d.vec.toFixed(2)} b=${d.bm25.toFixed(2)}`
                              : d.vec.toFixed(2)}
                        </span>
                      ) : (
                        <span className="tabular-nums text-ink-3">{Math.round(d.first * 100)}%</span>
                      )}
                    </motion.li>
                  )
                })}
              </ul>
            ) : (
              <p className="text-sm text-ink-3">Run a query to search the library…</p>
            )}
          </div>

          {/* prompt assembly + answer */}
          <div className="rounded-2xl border border-hairline bg-surface p-4">
            <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-3">
              <Depth b="What the model actually reads" t="Assembled prompt → generation" />
            </p>
            {after(phase, 'assemble') ? (
              <div className="flex flex-col gap-1.5">
                {kept.map((d, i) => (
                  <motion.div
                    key={d.id}
                    initial={{ opacity: 0, x: -28 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.18 }}
                    className="rounded-lg border-l-2 bg-page px-2.5 py-1.5 text-[11px] leading-snug text-ink-2"
                    style={{ borderColor: DOC_COLOR }}
                  >
                    <span className="mr-1 font-mono text-ink-3">[{i + 1}]</span>
                    {d.text}
                  </motion.div>
                ))}
                <motion.div
                  initial={{ opacity: 0, x: -28 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: KEEP * 0.18 }}
                  className="rounded-lg border-l-2 border-tok-blue bg-page px-2.5 py-1.5 text-[11px] text-ink"
                >
                  <span className="mr-1 font-mono text-ink-3">Q:</span>
                  {query.text}
                </motion.div>
                {phase === 'generate' && (
                  <motion.p
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-1 rounded-lg bg-tok-blue/10 px-2.5 py-2 text-xs leading-relaxed text-ink"
                  >
                    {query.answer}
                  </motion.p>
                )}
              </div>
            ) : (
              <p className="text-sm text-ink-3">
                <Depth
                  b="Retrieved passages will stack up here, in front of your question."
                  t="context chunks + user query → single forward prompt."
                />
              </p>
            )}
          </div>
        </div>
      </div>
    </Section>
  )
}
