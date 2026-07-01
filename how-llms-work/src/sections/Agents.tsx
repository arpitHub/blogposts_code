import { useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Section from '../components/Section'
import { Depth, useDepth } from '../context/DepthContext'
import {
  AGENT_ANSWER, AGENT_QUERY, AGENT_STEPS, STAGES, TOOLS, type Stage, type ToolName,
} from '../lib/agent'
import { FOCUS, INK, TOKEN_COLORS } from '../lib/palette'

const W = 720
const H = 330

// Node centers in the flowchart.
const POS = {
  query: { x: 90, y: 60 },
  plan: { x: 250, y: 60 },
  act: { x: 470, y: 60 },
  observe: { x: 470, y: 240 },
  decide: { x: 250, y: 240 },
  answer: { x: 610, y: 285 },
} as const

type NodeId = keyof typeof POS

const EDGES: Record<string, string> = {
  'query-plan': 'M 150 60 L 202 60',
  'plan-act': 'M 298 60 L 404 60',
  'act-observe': 'M 470 84 L 470 216',
  'observe-decide': 'M 412 240 L 304 240',
  'decide-plan': 'M 202 232 C 128 210, 128 90, 202 66', // the loop
  'decide-answer': 'M 268 264 C 330 322, 460 330, 540 300',
}

const TOTAL = AGENT_STEPS.length * STAGES.length // pointer TOTAL+1 = final answer

interface Cursor {
  iter: number
  stage: Stage | 'start' | 'answer'
  node: NodeId
  edge: string | null
}

function locate(pointer: number): Cursor {
  if (pointer <= 0) return { iter: 0, stage: 'start', node: 'query', edge: null }
  if (pointer > TOTAL) return { iter: AGENT_STEPS.length - 1, stage: 'answer', node: 'answer', edge: 'decide-answer' }
  const iter = Math.floor((pointer - 1) / STAGES.length)
  const stage = STAGES[(pointer - 1) % STAGES.length]
  const node: NodeId = stage === 'act' ? 'act' : stage
  const edge =
    stage === 'plan'
      ? iter === 0
        ? 'query-plan'
        : 'decide-plan'
      : stage === 'act'
        ? 'plan-act'
        : stage === 'observe'
          ? 'act-observe'
          : 'observe-decide'
  return { iter, stage, node, edge }
}

function FlowNode({ id, label, sub, active, w = 96 }: {
  id: NodeId
  label: string
  sub?: string
  active: boolean
  w?: number
}) {
  const p = POS[id]
  const h = sub ? 48 : 40
  return (
    <g transform={`translate(${p.x}, ${p.y})`}>
      <motion.rect
        x={-w / 2}
        y={-h / 2}
        width={w}
        height={h}
        rx={10}
        initial={false}
        animate={{
          fill: active ? 'rgba(57,135,229,0.14)' : '#232322',
          stroke: active ? FOCUS : 'rgba(255,255,255,0.12)',
        }}
        strokeWidth={1.5}
        filter={active ? 'url(#agent-glow)' : undefined}
      />
      <text
        y={sub ? -3 : 4}
        textAnchor="middle"
        fontSize={13}
        fontWeight={600}
        fill={active ? INK.primary : INK.secondary}
      >
        {label}
      </text>
      {sub && (
        <text y={15} textAnchor="middle" fontSize={10} fill={active ? INK.secondary : INK.muted} fontFamily="ui-monospace, monospace">
          {sub}
        </text>
      )}
    </g>
  )
}

function Flowchart({ cursor, tool }: { cursor: Cursor; tool: ToolName }) {
  const particle = POS[cursor.node]
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label="Agent loop flowchart: plan, call a tool, observe the result, then decide to loop again or answer"
    >
      <defs>
        <filter id="agent-glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 8 4 L 0 8 z" fill={INK.axis} />
        </marker>
        <marker id="arrow-hot" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 8 4 L 0 8 z" fill={FOCUS} />
        </marker>
      </defs>

      {/* edges — the one just traversed lights up */}
      {Object.entries(EDGES).map(([id, d]) => {
        const hot = cursor.edge === id
        return (
          <g key={id}>
            <path d={d} fill="none" stroke={INK.axis} strokeWidth={1.5} markerEnd="url(#arrow)" />
            {hot && (
              <motion.path
                key={`${id}-${cursor.iter}`}
                d={d}
                fill="none"
                stroke={FOCUS}
                strokeWidth={2.5}
                markerEnd="url(#arrow-hot)"
                filter="url(#agent-glow)"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 0.9 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              />
            )}
          </g>
        )
      })}
      <text x={116} y={155} fontSize={10} fill={INK.muted} fontFamily="ui-monospace, monospace">
        loop ↺
      </text>
      <text x={380} y={318} fontSize={10} fill={INK.muted} fontFamily="ui-monospace, monospace">
        done → answer
      </text>

      <FlowNode id="query" label="query" active={cursor.stage === 'start'} w={104} />
      <FlowNode id="plan" label="plan" active={cursor.stage === 'plan'} />
      <FlowNode
        id="act"
        label="call tool"
        sub={`${TOOLS[tool].icon} ${TOOLS[tool].label}`}
        active={cursor.stage === 'act'}
        w={128}
      />
      <FlowNode id="observe" label="observe" active={cursor.stage === 'observe'} w={110} />
      <FlowNode id="decide" label="decide" active={cursor.stage === 'decide'} w={100} />
      <FlowNode id="answer" label="answer" active={cursor.stage === 'answer'} w={120} />

      {/* the traveling particle */}
      <motion.g
        initial={false}
        animate={{ x: particle.x, y: particle.y - 32 }}
        transition={{ type: 'spring', stiffness: 140, damping: 18 }}
      >
        <motion.circle
          r={6}
          fill={FOCUS}
          filter="url(#agent-glow)"
          animate={{ scale: [1, 1.25, 1] }}
          transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
        />
      </motion.g>
    </svg>
  )
}

function TraceLog({ pointer }: { pointer: number }) {
  const [open, setOpen] = useState(true)
  const scroller = useRef<HTMLDivElement>(null)
  const lines = useMemo(() => {
    const out: Array<{ kind: 'thought' | 'action' | 'observation' | 'final'; text: string }> = []
    AGENT_STEPS.forEach((s, i) => {
      const progress = pointer - 1 - i * STAGES.length // stages completed within iteration i
      if (progress >= 0) out.push({ kind: 'thought', text: s.thought })
      if (progress >= 1) out.push({ kind: 'action', text: s.action })
      if (progress >= 2) out.push({ kind: 'observation', text: s.observation })
      if (progress >= 3) out.push({ kind: 'thought', text: s.decide })
    })
    if (pointer > TOTAL) out.push({ kind: 'final', text: AGENT_ANSWER })
    return out
  }, [pointer])

  // Follow the newest trace entry, like a terminal.
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' })
  }, [lines.length, open])

  const KIND_STYLE = {
    thought: { label: 'Thought', color: INK.secondary },
    action: { label: 'Action', color: TOKEN_COLORS[2] },
    observation: { label: 'Observation', color: TOKEN_COLORS[1] },
    final: { label: 'Final Answer', color: FOCUS },
  } as const

  return (
    <div className="rounded-2xl border border-hairline bg-surface p-4">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-left text-xs font-medium uppercase tracking-wider text-ink-3 transition-colors hover:text-ink-2"
      >
        <span>ReAct trace · {lines.length} {lines.length === 1 ? 'entry' : 'entries'}</span>
        <span aria-hidden="true">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div
          ref={scroller}
          className="mt-3 flex max-h-64 flex-col gap-1.5 overflow-y-auto font-mono text-[11px] leading-relaxed"
        >
          {lines.length === 0 && <p className="text-ink-3">Step through the loop to build the trace…</p>}
          <AnimatePresence initial={false}>
            {lines.map((l, i) => (
              <motion.p
                key={i}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                className="text-ink-2"
              >
                <span style={{ color: KIND_STYLE[l.kind].color }}>{KIND_STYLE[l.kind].label}:</span>{' '}
                {l.text}
              </motion.p>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}

export default function Agents() {
  const { depth } = useDepth()
  const [pointer, setPointer] = useState(0)
  const cursor = locate(pointer)
  const step = AGENT_STEPS[cursor.iter]
  const done = pointer > TOTAL

  const nextLabel = (() => {
    if (pointer >= TOTAL + 1) return null
    const next = locate(pointer + 1)
    if (next.stage === 'answer') return '✓ Reply'
    if (next.stage === 'plan') return pointer === 0 ? '▶ Start' : '↺ Plan again'
    if (next.stage === 'act') return `Call ${TOOLS[AGENT_STEPS[next.iter].tool].label}`
    if (next.stage === 'observe') return 'Observe result'
    return 'Decide'
  })()

  const caption = (() => {
    switch (cursor.stage) {
      case 'start':
        return 'The question arrives. Instead of answering straight away, the agent starts a loop.'
      case 'plan':
        return step.plan
      case 'act':
        return `It picks the ${TOOLS[step.tool].label} and calls: ${step.action}`
      case 'observe':
        return `The tool comes back with: ${step.observation}`
      case 'decide':
        return step.decide
      case 'answer':
        return AGENT_ANSWER
    }
  })()

  return (
    <Section
      id="agents"
      kicker="Step 7 · Agents"
      title="Give the loop hands"
      lede={
        <Depth
          b="An agent is the generation loop from Step 5 with one upgrade: instead of only writing words, the model can take actions — search the web, run a calculation, execute code — look at what came back, and check its own work before replying. Step through a real(ish) example below."
          t={
            <>
              Agent = LLM + tool schema + a control loop. Each turn the model emits either a tool
              call or a final answer; tool output is appended to the context and the loop re-enters
              — the ReAct pattern (reason + act). Step through the loop and watch the
              thought/action/observation trace accumulate on the right.
            </>
          }
        />
      }
    >
      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="rounded-2xl border border-hairline bg-surface p-4 sm:p-5">
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-ink-3">Query</p>
          <p className="mb-4 text-sm text-ink">{AGENT_QUERY}</p>

          <Flowchart cursor={cursor} tool={step.tool} />

          <div className="mt-4 flex flex-wrap items-center gap-3">
            {nextLabel ? (
              <button
                onClick={() => setPointer((p) => p + 1)}
                className="rounded-full bg-tok-blue px-4 py-1.5 text-sm font-medium text-white transition-opacity hover:opacity-90"
              >
                {nextLabel}
              </button>
            ) : (
              <span className="rounded-full bg-tok-blue/15 px-4 py-1.5 text-sm font-medium text-tok-blue">
                ✓ Answered
              </span>
            )}
            {pointer > 0 && (
              <button
                onClick={() => setPointer(0)}
                className="rounded-full border border-hairline px-4 py-1.5 text-sm text-ink-3 transition-colors hover:border-tok-blue hover:text-ink-2"
              >
                Reset
              </button>
            )}
            <span className="ml-auto font-mono text-xs text-ink-3">
              {done ? 'done' : cursor.stage === 'start' ? 'ready' : `iteration ${cursor.iter + 1} / ${AGENT_STEPS.length}`}
            </span>
          </div>

          {/* the toolbox — the active iteration's tool lights up while calling */}
          <div className="mt-4 flex flex-wrap gap-2">
            {(Object.keys(TOOLS) as ToolName[]).map((t) => {
              const active = cursor.stage === 'act' && step.tool === t
              return (
                <span
                  key={t}
                  className={`rounded-lg border px-2.5 py-1 text-xs transition-colors ${
                    active ? 'border-tok-blue bg-tok-blue/15 text-ink' : 'border-hairline text-ink-3'
                  }`}
                >
                  {TOOLS[t].icon} {TOOLS[t].label}
                </span>
              )
            })}
            <span className="self-center text-[11px] text-ink-3">
              <Depth b="· its toolbox" t="· tools exposed via JSON schema" />
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          {depth === 'beginner' ? (
            <motion.div
              key={pointer}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl border border-hairline bg-surface p-4"
            >
              <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-3">
                What's happening
              </p>
              <p className="text-sm leading-relaxed text-ink-2">{caption}</p>
            </motion.div>
          ) : (
            <TraceLog pointer={pointer} />
          )}

          {done && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="rounded-2xl border border-hairline bg-surface p-4"
            >
              <p className="mb-1 text-xs font-medium uppercase tracking-wider text-ink-3">
                Final reply
              </p>
              <p className="rounded-lg bg-tok-blue/10 px-2.5 py-2 text-xs leading-relaxed text-ink">
                {AGENT_ANSWER}
              </p>
              <p className="mt-2 text-[11px] leading-relaxed text-ink-3">
                <Depth
                  b="Three tool calls, each one checked before moving on — that's the whole trick."
                  t="3 iterations · terminates when the model emits an answer instead of a tool call (real systems also cap steps and tokens)."
                />
              </p>
            </motion.div>
          )}
        </div>
      </div>
    </Section>
  )
}
