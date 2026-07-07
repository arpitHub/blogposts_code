import { useEffect, useRef, useState } from 'react'
import { TopCourtSVG, PlayerDot, Ball, MID_X, NET_Y, BASE_TOP, BASE_BOT, IN_X0, IN_X1, SRV } from './TopCourt'
import { Segmented } from '../ui'

// Animated patterns of play. Each step: who hits, ball from → to, where both
// players move, and a coaching note.

const PATTERNS = {
  serveplus1: {
    label: 'Serve + 1',
    blurb: 'The most common point-winning pattern in tennis: a wide serve drags the returner off court, and the very next ball goes behind them or into the space.',
    steps: [
      {
        note: 'Serve wide from the deuce side. The target is the corner of the service box — not an ace, a DISPLACEMENT. The returner must chase it.',
        ball: [[MID_X + 40, BASE_BOT + 16], [IN_X0 + 14, NET_Y - SRV + 10]],
        you: [MID_X + 40, BASE_BOT + 16], opp: [IN_X0 + 30, BASE_TOP - 10],
      },
      {
        note: 'The stretched return floats back weakly, usually cross-court. While it travels, you’ve already moved inside the baseline, hunting a forehand.',
        ball: [[IN_X0 + 20, NET_Y - SRV], [MID_X - 20, BASE_BOT - 60]],
        you: [MID_X - 4, BASE_BOT - 30], opp: [IN_X0 - 8, BASE_TOP + 30],
      },
      {
        note: 'The “+1”: drive the forehand into the open court, away from the recovering returner. Point over — and you only had to hit one groundstroke.',
        ball: [[MID_X - 20, BASE_BOT - 60], [IN_X1 - 22, BASE_TOP + 26]],
        you: [MID_X - 10, BASE_BOT - 50], opp: [MID_X - 60, BASE_TOP + 10],
      },
    ],
  },
  crossgrind: {
    label: 'Cross-court grind → line',
    blurb: 'The bread-and-butter rally pattern: trade cross-court balls (high margin, short recovery) until a short one arrives — then change direction down the line.',
    steps: [
      {
        note: 'Rally cross-court. The net is lower in the middle, the court is longer on the diagonal, and your recovery is a single step. You can do this all day.',
        ball: [[IN_X0 + 40, BASE_BOT - 20], [IN_X1 - 40, BASE_TOP + 20]],
        you: [IN_X0 + 40, BASE_BOT - 20], opp: [IN_X1 - 40, BASE_TOP + 20],
      },
      {
        note: 'They trade it back cross-court. Patience is the skill here — changing direction off a deep ball is how errors happen. Wait.',
        ball: [[IN_X1 - 40, BASE_TOP + 20], [IN_X0 + 44, BASE_BOT - 26]],
        you: [IN_X0 + 44, BASE_BOT - 26], opp: [IN_X1 - 46, BASE_TOP + 24],
      },
      {
        note: 'A SHORT ball lands mid-court — the trigger. Step in and take it early.',
        ball: [[IN_X1 - 46, BASE_TOP + 24], [IN_X0 + 60, NET_Y + 74]],
        you: [IN_X0 + 56, NET_Y + 80], opp: [IN_X1 - 40, BASE_TOP + 20],
      },
      {
        note: 'Down the line, behind their recovery. Direction change off a SHORT ball is low-risk; the same shot off a deep ball is a gift to your opponent. That timing distinction is the whole pattern.',
        ball: [[IN_X0 + 60, NET_Y + 74], [IN_X0 + 20, BASE_TOP + 18]],
        you: [IN_X0 + 60, NET_Y + 70], opp: [MID_X + 40, BASE_TOP + 14],
      },
    ],
  },
  approach: {
    label: 'Approach & volley',
    blurb: 'Turn a weak ball into net dominance: approach DOWN THE LINE (shorter recovery to cover the pass), then volley into the space you created.',
    steps: [
      {
        note: 'Your opponent leaves a ball short and sitting. Move through it — this is an invitation, not a rally ball.',
        ball: [[MID_X - 30, BASE_TOP + 30], [MID_X + 30, NET_Y + 80]],
        you: [MID_X + 26, NET_Y + 96], opp: [MID_X - 30, BASE_TOP + 30],
      },
      {
        note: 'Approach down the line and keep moving forward — hit and run in one motion. Down the line, because your net position then only needs one step to cover their straight pass.',
        ball: [[MID_X + 30, NET_Y + 80], [IN_X1 - 20, BASE_TOP + 16]],
        you: [MID_X + 44, NET_Y + 46], opp: [IN_X1 - 30, BASE_TOP + 12],
      },
      {
        note: 'They’re forced to pass from a corner. You’re set just right of center, shading the line. Their highest-percentage try is the cross-court dip…',
        ball: [[IN_X1 - 20, BASE_TOP + 16], [MID_X - 40, NET_Y + 40]],
        you: [MID_X + 30, NET_Y + 52], opp: [IN_X1 - 36, BASE_TOP + 24],
      },
      {
        note: '…which you cut off and angle away into the open court. First volley wins — because the approach shot chose the right line.',
        ball: [[MID_X - 30, NET_Y + 48], [IN_X0 + 26, BASE_TOP + 60]],
        you: [MID_X - 16, NET_Y + 46], opp: [IN_X1 - 50, BASE_TOP + 20],
      },
    ],
  },
  dropLob: {
    label: 'Drop shot → lob',
    blurb: 'The cruelest one-two in tennis: drag them sprinting forward with a drop shot, then float the next ball over their head while they’re still braking.',
    steps: [
      {
        note: 'From inside the baseline, off a slower ball: the drop shot. Disguise matters more than perfection — it should look like your drive until the last instant.',
        ball: [[MID_X - 20, BASE_BOT - 70], [MID_X - 50, NET_Y - 40]],
        you: [MID_X - 20, BASE_BOT - 70], opp: [MID_X + 20, BASE_TOP + 16],
      },
      {
        note: 'They sprint diagonally and barely scrape it back — a soft, rising ball with no depth. They are now standing ON the net.',
        ball: [[MID_X - 50, NET_Y - 36], [MID_X + 10, NET_Y + 60]],
        you: [MID_X + 16, NET_Y + 90], opp: [MID_X - 44, NET_Y - 26],
      },
      {
        note: 'The lob, over their backhand shoulder. They must turn, chase, and hit a desperate ball — or watch it bounce. Space created forward, exploited backward.',
        ball: [[MID_X + 10, NET_Y + 60], [IN_X1 - 40, BASE_TOP + 24]],
        you: [MID_X, NET_Y + 80], opp: [MID_X - 30, NET_Y - 60],
      },
    ],
  },
}

export default function PatternPlayer() {
  const [patternId, setPatternId] = useState('serveplus1')
  const [stepIdx, setStepIdx] = useState(0)
  const [anim, setAnim] = useState(1) // 0..1 ball progress within step
  const rafRef = useRef()
  const pattern = PATTERNS[patternId]
  const step = pattern.steps[stepIdx]

  // animate ball whenever step changes
  useEffect(() => {
    setAnim(0)
    const start = performance.now()
    const dur = 900
    const tick = (now) => {
      const t = Math.min((now - start) / dur, 1)
      setAnim(t)
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [patternId, stepIdx])

  const [from, to] = step.ball
  // slight arc on the ball path
  const bx = from[0] + (to[0] - from[0]) * anim
  const by = from[1] + (to[1] - from[1]) * anim - Math.sin(anim * Math.PI) * 26

  const selectPattern = (id) => { setPatternId(id); setStepIdx(0) }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,300px)_1fr]">
      <TopCourtSVG label={`Pattern: ${pattern.label}, step ${stepIdx + 1}`}>
        {/* previous steps' ball paths, numbered */}
        {pattern.steps.slice(0, stepIdx + 1).map((s, i) => {
          const [f, t] = s.ball
          const past = i < stepIdx
          return (
            <g key={i} opacity={past ? 0.4 : 0.9}>
              <line
                x1={f[0]} y1={f[1]}
                x2={past ? t[0] : f[0] + (t[0] - f[0]) * anim}
                y2={past ? t[1] : f[1] + (t[1] - f[1]) * anim}
                stroke="#dce65a" strokeWidth="2.5" strokeDasharray="6 5"
              />
              <circle cx={f[0]} cy={f[1]} r="9" fill="#0d1d15" opacity="0.55" />
              <text x={f[0]} y={f[1] + 3.5} textAnchor="middle" fontSize="10" fontWeight="bold" fill="#dce65a">{i + 1}</text>
            </g>
          )
        })}
        <PlayerDot x={step.opp[0]} y={step.opp[1]} color="#1f3f2e" label="opponent" />
        <PlayerDot x={step.you[0]} y={step.you[1]} label="you" />
        <Ball x={bx} y={by} />
      </TopCourtSVG>

      <div>
        <div className="mb-3 overflow-x-auto">
          <Segmented
            options={Object.entries(PATTERNS).map(([value, v]) => ({ value, label: v.label }))}
            value={patternId}
            onChange={selectPattern}
            size="sm"
          />
        </div>

        <p className="text-sm leading-relaxed text-court-700">{pattern.blurb}</p>

        <div className="mt-4 rounded-xl border border-line bg-court-50/50 px-4 py-3">
          <div className="text-xs font-bold uppercase tracking-wide text-clay-600">
            Shot {stepIdx + 1} of {pattern.steps.length}
          </div>
          <p className="mt-1.5 min-h-20 text-sm leading-relaxed text-court-900/90">{step.note}</p>
        </div>

        <div className="mt-3 flex items-center gap-2">
          <button
            onClick={() => setStepIdx((i) => Math.max(i - 1, 0))}
            disabled={stepIdx === 0}
            className="rounded-lg border border-line bg-white px-4 py-2 text-sm font-medium text-court-700 transition hover:border-clay-400 disabled:opacity-40"
          >
            ← Back
          </button>
          {stepIdx < pattern.steps.length - 1 ? (
            <button
              onClick={() => setStepIdx((i) => i + 1)}
              className="rounded-lg bg-clay-500 px-4 py-2 text-sm font-semibold text-white transition hover:bg-clay-600"
            >
              Next shot →
            </button>
          ) : (
            <button
              onClick={() => setStepIdx(0)}
              className="rounded-lg bg-court-800 px-4 py-2 text-sm font-semibold text-white transition hover:bg-court-700"
            >
              ↺ Replay pattern
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
