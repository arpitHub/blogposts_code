import { useState } from 'react'
import { Segmented } from '../ui'

// Top-down view: net at the top, baseline at the bottom.
const STANCES = {
  neutral: {
    label: 'Neutral',
    feet: [
      { x: 120, y: 150, r: -80 }, // back (right) foot
      { x: 100, y: 105, r: -75 }, // front (left) foot steps toward net
    ],
    hips: { x1: 88, y1: 132, x2: 134, y2: 122 },
    blurb: 'The classic “step in” stance: front foot toward the net, weight transfers forward through the shot. Best for balls you have time to set up for — it naturally drives energy into the court.',
    pros: 'Easy weight transfer, great for approach shots and beginners learning the swing.',
    cons: 'Slower to recover from — you finish side-on and must reset.',
  },
  open: {
    label: 'Open',
    feet: [
      { x: 135, y: 130, r: -10 },
      { x: 88, y: 128, r: -25 },
    ],
    hips: { x1: 86, y1: 118, x2: 138, y2: 118 },
    blurb: 'Feet face the net; power comes from coiling the upper body against the hips and legs, then uncoiling — like wringing a towel. The default on the modern forehand, especially when rushed.',
    pros: 'Fast to set up and fast to recover — you’re already facing the court.',
    cons: 'Needs a real shoulder coil; without it the stance becomes an armsy slap.',
  },
  closed: {
    label: 'Closed',
    feet: [
      { x: 122, y: 152, r: -85 },
      { x: 88, y: 112, r: -60 }, // front foot crosses over
    ],
    hips: { x1: 90, y1: 136, x2: 128, y2: 118 },
    blurb: 'The front foot crosses beyond the ball line — usually forced, when you’re stretched wide or running around a backhand. Hip rotation is blocked, so the arm and shoulders do more of the work.',
    pros: 'Reaches balls the other stances can’t; disguises direction well.',
    cons: 'Blocks the hips — hardest stance to generate easy power from.',
  },
}

export default function StanceDiagram() {
  const [stance, setStance] = useState('neutral')
  const s = STANCES[stance]

  return (
    <div>
      <div className="mb-4">
        <Segmented
          options={Object.entries(STANCES).map(([value, v]) => ({ value, label: v.label }))}
          value={stance}
          onChange={setStance}
        />
      </div>

      <div className="grid items-center gap-6 sm:grid-cols-[240px_1fr]">
        <svg viewBox="0 0 224 190" className="mx-auto w-full max-w-[240px]" role="img" aria-label={`${s.label} stance, viewed from above`}>
          {/* court patch */}
          <rect x="0" y="0" width="224" height="190" rx="10" fill="#cb6d4a" />
          {/* net direction */}
          <line x1="12" y1="24" x2="212" y2="24" stroke="#faf7f2" strokeWidth="3" />
          <text x="112" y="16" textAnchor="middle" fontSize="11" fill="#faf7f2" fontWeight="bold">NET ↑</text>
          {/* baseline */}
          <line x1="12" y1="176" x2="212" y2="176" stroke="#faf7f2" strokeWidth="2" opacity="0.7" />
          <text x="112" y="188" textAnchor="middle" fontSize="9" fill="#faf7f2" opacity="0.9">baseline</text>
          {/* incoming ball path */}
          <line x1="196" y1="34" x2="150" y2="112" stroke="#dce65a" strokeWidth="2" strokeDasharray="4 4" />
          <circle cx="150" cy="112" r="5" fill="#dce65a" stroke="#b3bd2d" />
          {/* hips line */}
          <line {...s.hips} stroke="#faf7f2" strokeWidth="3" strokeLinecap="round" opacity="0.85">
            <animate attributeName="opacity" values="0.85;0.5;0.85" dur="2.5s" repeatCount="indefinite" />
          </line>
          {/* feet */}
          {s.feet.map((f, i) => (
            <g key={i} transform={`translate(${f.x} ${f.y}) rotate(${f.r})`} className="transition-transform duration-500">
              <ellipse rx="9" ry="16" fill="#1f3f2e" opacity={i === 0 ? 0.75 : 1} />
              <ellipse cx="0" cy="-11" rx="6" ry="5" fill="#1f3f2e" opacity={i === 0 ? 0.75 : 1} />
            </g>
          ))}
        </svg>

        <div>
          <p className="leading-relaxed text-court-900/90">{s.blurb}</p>
          <div className="mt-3 grid gap-2 text-sm">
            <div className="rounded-lg bg-court-50 px-3 py-2"><b className="text-court-600">+ </b>{s.pros}</div>
            <div className="rounded-lg bg-clay-50 px-3 py-2"><b className="text-clay-600">− </b>{s.cons}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
