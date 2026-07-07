import { useState } from 'react'
import { TopCourtSVG, PlayerDot, MID_X, NET_Y, BASE_TOP, BASE_BOT, IN_X0, IN_X1 } from './TopCourt'
import { Segmented } from '../ui'

// Where should you stand AFTER your shot? Not the center mark — the spot that
// bisects your opponent's possible angles.

const SHOTS = {
  crossL: {
    label: 'Deep to their left corner',
    opp: { x: IN_X0 + 18, y: BASE_TOP + 18 },
    caption: 'Your ball pushed them wide to the left side. Their fastest reply is the straight ball down that same sideline; the cross-court has much further to travel. Split the difference: the bisector sits a small step LEFT of your center mark — cheat toward the sideline the ball went to, but only a step.',
  },
  middle: {
    label: 'Deep down the middle',
    opp: { x: MID_X, y: BASE_TOP + 18 },
    caption: 'A deep ball through the middle gives them no angle at all — their cone is narrow and symmetric, so your recovery spot is exactly the center mark. This is why “when in trouble, go big up the middle” is such reliable advice.',
  },
  crossR: {
    label: 'Deep to their right corner',
    opp: { x: IN_X1 - 18, y: BASE_TOP + 18 },
    caption: 'Mirror image: with the opponent stretched wide right, the quick straight ball comes down your right sideline, so the bisector — and your recovery spot — shifts a step RIGHT of the center mark. Notice it never shifts far: the court is long, so the cone stays surprisingly narrow.',
  },
}

function bisectorPoint(opp) {
  // opponent's two extreme targets on your baseline
  const t1 = { x: IN_X0 + 10, y: BASE_BOT }
  const t2 = { x: IN_X1 - 10, y: BASE_BOT }
  // recovery x = where the angle bisector from opp crosses your baseline area
  const a1 = Math.atan2(t1.y - opp.y, t1.x - opp.x)
  const a2 = Math.atan2(t2.y - opp.y, t2.x - opp.x)
  const mid = (a1 + a2) / 2
  const dy = BASE_BOT - 12 - opp.y
  const x = opp.x + Math.cos(mid) * (dy / Math.sin(mid))
  return { x, y: BASE_BOT - 12 }
}

export default function RecoveryLab() {
  const [shotId, setShotId] = useState('crossL')
  const s = SHOTS[shotId]
  const rec = bisectorPoint(s.opp)
  const t1 = { x: IN_X0 + 10, y: BASE_BOT }
  const t2 = { x: IN_X1 - 10, y: BASE_BOT }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,300px)_1fr]">
      <TopCourtSVG label="Recovery position diagram">
        {/* opponent's possible-angle cone */}
        <path
          d={`M${s.opp.x},${s.opp.y} L${t1.x},${t1.y} L${t2.x},${t2.y} Z`}
          fill="#dce65a" opacity="0.18"
        />
        <line x1={s.opp.x} y1={s.opp.y} x2={t1.x} y2={t1.y} stroke="#dce65a" strokeWidth="1.5" strokeDasharray="5 5" opacity="0.8" />
        <line x1={s.opp.x} y1={s.opp.y} x2={t2.x} y2={t2.y} stroke="#dce65a" strokeWidth="1.5" strokeDasharray="5 5" opacity="0.8" />
        {/* bisector */}
        <line x1={s.opp.x} y1={s.opp.y} x2={rec.x} y2={rec.y} stroke="#cf5f38" strokeWidth="2.5" strokeDasharray="2 5" />
        {/* your shot that started it */}
        <line x1={MID_X} y1={BASE_BOT - 40} x2={s.opp.x} y2={s.opp.y} stroke="white" strokeWidth="2" opacity="0.5" strokeDasharray="7 6" />
        <PlayerDot x={s.opp.x} y={s.opp.y} color="#1f3f2e" label="opponent" />
        {/* recovery spot */}
        <g className="transition-all duration-700" transform={`translate(${rec.x} ${rec.y})`}>
          <circle r="16" fill="none" stroke="#cf5f38" strokeWidth="2.5" strokeDasharray="4 4">
            <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="9s" repeatCount="indefinite" />
          </circle>
          <circle r="10" fill="#cf5f38" stroke="white" strokeWidth="2.5" />
        </g>
        <text x={rec.x} y={rec.y + 32} textAnchor="middle" fontSize="11" fontWeight="bold" fill="white">recover here</text>
        {/* center mark reference */}
        <text x={MID_X} y={BASE_BOT + 22} textAnchor="middle" fontSize="10" fill="#faf7f2" opacity="0.85">center mark</text>
      </TopCourtSVG>

      <div>
        <div className="mb-4">
          <Segmented
            options={Object.entries(SHOTS).map(([value, v]) => ({ value, label: v.label }))}
            value={shotId}
            onChange={setShotId}
            size="sm"
          />
        </div>
        <p className="text-sm leading-relaxed text-court-800/90">
          The yellow cone is <strong>everywhere your opponent can plausibly hit next</strong>.
          Your recovery spot splits that cone in half — equal running distance to their best
          ball on either side.
        </p>
        <div className="mt-3 rounded-xl border border-line bg-court-50/50 px-4 py-3 text-sm leading-relaxed text-court-900/90">
          {s.caption}
        </div>
        <p className="mt-3 text-xs text-court-500">
          Rule of thumb: hit cross-court, recover just PAST the center mark toward your shot’s
          side; hit down the middle, recover to the mark itself.
        </p>
      </div>
    </div>
  )
}
