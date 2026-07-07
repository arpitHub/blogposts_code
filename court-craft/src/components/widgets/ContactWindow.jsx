import { useState } from 'react'
import { Link } from 'react-router-dom'
import { SliderRow } from '../ui'

/**
 * Zoomed-in look at the contact moment: slide from a flat drive to a heavy
 * topspin brush and watch the swing path through the ball change.
 */
export default function ContactWindow() {
  const [brush, setBrush] = useState(35) // 0 = dead flat, 100 = maximum brush

  const angle = (brush / 100) * 52 // swing-path angle at contact, degrees
  const rad = (angle * Math.PI) / 180
  const rpm = Math.round(brush * 32)
  const label = brush < 15 ? 'Flat drive' : brush < 45 ? 'Drive with shape' : brush < 75 ? 'Rally topspin' : 'Heavy topspin'

  // swing-path arrow through the contact point (140,110)
  const cx = 140
  const cy = 110
  const L = 78
  const x1 = cx - L * Math.cos(rad)
  const y1 = cy + L * Math.sin(rad)
  const x2 = cx + L * Math.cos(rad)
  const y2 = cy - L * Math.sin(rad)

  return (
    <div className="grid items-center gap-6 sm:grid-cols-[280px_1fr]">
      <svg viewBox="0 0 280 220" className="mx-auto w-full max-w-[280px]" role="img" aria-label="Swing path through the contact point">
        <rect x="0" y="0" width="280" height="220" rx="10" fill="#f6f9f6" />
        {/* target direction */}
        <g stroke="#8bb899" strokeWidth="1.5" fill="#8bb899">
          <line x1="150" y1="40" x2="240" y2="40" strokeDasharray="4 4" />
          <text x="243" y="44" fontSize="10" stroke="none" fill="#5d9770">to net</text>
        </g>

        {/* swing path arrow */}
        <g stroke="#cf5f38" strokeWidth="4" fill="#cf5f38" strokeLinecap="round">
          <line x1={x1} y1={y1} x2={x2} y2={y2} />
          <g transform={`translate(${x2} ${y2}) rotate(${-angle})`}>
            <path d="M0 0 l -12 -6 l 3 6 l -3 6 z" stroke="none" />
          </g>
        </g>

        {/* racket face (slightly closed as brush increases) */}
        <g transform={`translate(${cx - 26} ${cy}) rotate(${-4 - brush * 0.06})`}>
          <ellipse rx="10" ry="30" fill="none" stroke="#1f3f2e" strokeWidth="3" />
          <line x1="0" y1="30" x2="-6" y2="58" stroke="#1f3f2e" strokeWidth="3" strokeLinecap="round" />
        </g>

        {/* ball with spin arrows */}
        <g transform={`translate(${cx} ${cy})`}>
          <circle r="16" fill="#dce65a" stroke="#b3bd2d" strokeWidth="1.5" />
          <path d="M-16,0 A19,19 0 0 1 16,0" fill="none" stroke="#fff" strokeWidth="2" />
          {brush > 12 && (
            <g stroke="#7d311e" strokeWidth="2" fill="#7d311e" opacity={Math.min(1, (brush - 12) / 40)}>
              <path d="M -4 -22 A 22 22 0 0 1 15 -16" fill="none" />
              <path d="M 15 -16 l -1 -6 l 6 3 z" stroke="none" />
              <path d="M 4 22 A 22 22 0 0 1 -15 16" fill="none" />
              <path d="M -15 16 l 1 6 l -6 -3 z" stroke="none" />
            </g>
          )}
        </g>

        <text x="14" y="204" fontSize="11" fill="#3f7a54">
          swing path: {angle.toFixed(0)}° upward through contact
        </text>
      </svg>

      <div>
        <div className="mb-1 flex items-baseline gap-3">
          <span className="font-display text-xl font-bold text-court-950">{label}</span>
          <span className="text-sm text-court-500">≈ {rpm.toLocaleString()} rpm</span>
        </div>
        <p className="mb-4 text-sm leading-relaxed text-court-800/90">
          {brush < 15
            ? 'The racket travels level through contact — maximum pace, minimum net clearance. Great for putaways, risky as a rally default.'
            : brush < 45
              ? 'A slightly upward path adds shape to the ball without giving up much speed. This is where most solid club forehands live.'
              : brush < 75
                ? 'A strong low-to-high brush: the ball arcs well over the net and dips back in. Swing fast — the spin keeps it in.'
                : 'Steep brush, huge spin: the ball jumps off the court on the bounce. Costs some forward pace — perfect for high, heavy rally balls.'}
        </p>
        <SliderRow
          label="Brush up the back of the ball"
          value={brush} onChange={setBrush}
          min={0} max={100}
          leftHint="drive through it" rightHint="brush up it"
        />
        <p className="mt-3 text-xs text-court-500">
          Want the full physics? Fire these exact shots in the{' '}
          <Link to="/spin" className="font-medium text-clay-600 underline">Ball Flight Lab →</Link>
        </p>
      </div>
    </div>
  )
}
