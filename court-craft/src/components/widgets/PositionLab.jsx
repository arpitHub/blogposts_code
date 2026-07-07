import { useRef, useState } from 'react'
import { TopCourtSVG, PlayerDot, MID_X, NET_Y, BASE_TOP, BASE_BOT, IN_X0, IN_X1, X0, CW, SRV, VIEW_W, VIEW_H, S } from './TopCourt'
import { Segmented } from '../ui'

// Drag "you" around the court and get live positioning feedback per scenario.

const SCENARIOS = {
  rally: {
    label: 'Baseline rally (singles)',
    ideal: { x: MID_X, y: BASE_BOT + 20 },
    others: [{ x: MID_X, y: BASE_TOP - 14, color: '#1f3f2e', label: 'opponent' }],
    intro: 'Neutral rally, opponent centered behind their baseline. Where do you wait for the next ball?',
    tips: 'Home base is a step BEHIND the center of the baseline — deep balls land in front of you (playable) instead of at your feet (misery).',
  },
  return: {
    label: 'Returning a big serve',
    ideal: { x: IN_X1 - 40, y: BASE_BOT + 34 },
    others: [
      { x: MID_X + 30, y: BASE_TOP - 14, color: '#1f3f2e', label: 'server' },
      { x: (MID_X + IN_X1) / 2, y: NET_Y + 40, color: 'transparent', label: '' },
    ],
    intro: 'Opponent is serving to your right-side (deuce) box, and they hit it big. Position yourself to return.',
    tips: 'Stand near the middle of the serve’s possible angles — around the singles sideline’s inner third — and BEHIND the baseline: extra depth buys reaction time against pace. Move in only for weaker second serves.',
  },
  net: {
    label: 'You approached the net',
    ideal: { x: MID_X + 40, y: NET_Y + 78 },
    others: [{ x: IN_X0 + 30, y: BASE_TOP - 14, color: '#1f3f2e', label: 'opponent' }],
    intro: 'You hit an approach shot down the right sideline and you’re following it in. Where do you set up for the first volley?',
    tips: 'Volley position is about 2–3 m inside the service line, shifted toward the side you approached down — you’re shadowing the ball, cutting off the straight passing shot and making them try the harder cross-court pass.',
  },
  doubles: {
    label: 'Doubles: partner serving',
    ideal: { x: (IN_X0 + MID_X) / 2 - 8, y: NET_Y + SRV / 2 },
    others: [
      { x: MID_X + CW / 3, y: BASE_BOT + 16, color: '#1f3f2e', label: 'partner (serving)' },
      { x: MID_X - 30, y: BASE_TOP - 14, color: '#5d5d5d', label: 'returner' },
      { x: IN_X1 - 40, y: NET_Y - SRV / 2, color: '#5d5d5d', label: 'their net player' },
    ],
    intro: 'Your partner serves from the right half. As the server’s partner, you start at the net — but where exactly?',
    tips: 'Middle of the service box on YOUR half: close enough to pick off weak returns, far enough to cover the lob over you. From here you poach toward the middle when the return floats.',
  },
}

function feedback(pos, sc) {
  const dx = pos.x - sc.ideal.x
  const dy = pos.y - sc.ideal.y
  const dist = Math.hypot(dx, dy) / S // meters
  let grade, msg
  if (dist < 1.2) {
    grade = 'good'
    msg = 'Textbook. ' + sc.tips
  } else if (dist < 3) {
    grade = 'close'
    const dir = Math.abs(dx) > Math.abs(dy)
      ? (dx > 0 ? 'a bit further left' : 'a bit further right')
      : (dy > 0 ? 'a bit further forward (toward the net)' : 'a bit further back')
    msg = `Workable, but shade ${dir}. ${sc.tips}`
  } else {
    grade = 'far'
    // zone-specific coaching
    if (pos.y > NET_Y + SRV && pos.y < BASE_BOT - 10 && sc.ideal.y > BASE_BOT - 40) {
      msg = 'You’re camped in no man’s land — balls will land at your feet all day. Commit: baseline or net, not between.'
    } else if (pos.y < NET_Y) {
      msg = 'You’ve wandered onto your opponent’s side of the net — that one’s actually against the rules mid-point!'
    } else {
      msg = 'A long way from the working zone. ' + sc.tips
    }
  }
  return { grade, dist, msg }
}

export default function PositionLab() {
  const [scId, setScId] = useState('rally')
  const [pos, setPos] = useState({ x: MID_X - 60, y: BASE_BOT - 80 })
  const [reveal, setReveal] = useState(false)
  const svgRef = useRef(null)
  const dragging = useRef(false)
  const sc = SCENARIOS[scId]
  const fb = feedback(pos, sc)

  const toSvg = (e) => {
    const rect = svgRef.current.getBoundingClientRect()
    const x = ((e.clientX - rect.left) / rect.width) * VIEW_W
    const y = ((e.clientY - rect.top) / rect.height) * VIEW_H
    return {
      x: Math.min(Math.max(x, X0 - 20), X0 + CW + 20),
      y: Math.min(Math.max(y, NET_Y - 30), VIEW_H - 14),
    }
  }

  const ringColor = { good: '#3f7a54', close: '#de7f5c', far: '#9a3a1f' }[fb.grade]

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,300px)_1fr]">
      <div
        ref={svgRef}
        className="touch-none"
        onPointerDown={(e) => { dragging.current = true; setPos(toSvg(e)) }}
        onPointerMove={(e) => { if (dragging.current) setPos(toSvg(e)) }}
        onPointerUp={() => { dragging.current = false }}
        onPointerLeave={() => { dragging.current = false }}
      >
        <TopCourtSVG label="Draggable positioning court">
          {sc.others.map((o, i) => o.color !== 'transparent' && (
            <PlayerDot key={i} x={o.x} y={o.y} color={o.color} label={o.label} />
          ))}
          {reveal && (
            <g className="transition-all duration-500">
              <circle cx={sc.ideal.x} cy={sc.ideal.y} r={1.2 * S} fill="#3f7a54" opacity="0.25" />
              <circle cx={sc.ideal.x} cy={sc.ideal.y} r="7" fill="#3f7a54" stroke="white" strokeWidth="2" />
            </g>
          )}
          {/* your draggable dot */}
          <circle cx={pos.x} cy={pos.y} r="22" fill={ringColor} opacity="0.25" />
          <PlayerDot x={pos.x} y={pos.y} label="you (drag me)" r={11} />
        </TopCourtSVG>
      </div>

      <div>
        <div className="mb-4 overflow-x-auto">
          <Segmented
            options={Object.entries(SCENARIOS).map(([value, v]) => ({ value, label: v.label }))}
            value={scId}
            onChange={(id) => { setScId(id); setReveal(false) }}
            size="sm"
          />
        </div>

        <p className="text-sm leading-relaxed text-court-800/90">{sc.intro}</p>

        <div className={`mt-3 rounded-xl border px-4 py-3 text-sm leading-relaxed transition-colors ${
          fb.grade === 'good' ? 'border-court-300 bg-court-50 text-court-900'
            : fb.grade === 'close' ? 'border-clay-200 bg-clay-50 text-court-900'
              : 'border-clay-300 bg-clay-100 text-clay-900'
        }`}>
          <b>
            {fb.grade === 'good' ? '✓ Great spot' : fb.grade === 'close' ? '≈ Nearly' : '✗ Rethink this one'}
            {' '}({fb.dist.toFixed(1)} m from ideal).
          </b>{' '}
          {fb.msg}
        </div>

        <button
          onClick={() => setReveal((v) => !v)}
          className="mt-3 rounded-lg border border-line bg-white px-3 py-1.5 text-xs font-medium text-court-700 transition hover:border-clay-400"
        >
          {reveal ? 'Hide' : 'Show'} the ideal zone
        </button>
      </div>
    </div>
  )
}
