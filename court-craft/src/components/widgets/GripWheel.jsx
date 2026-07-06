import { useState } from 'react'
import { Segmented } from '../ui'

// Handle cross-section, viewed from the butt cap of a racket held on edge
// (like a hammer). Bevels numbered 1-8 clockwise for a right-hander;
// bevel 1 is on top when the racket face is perpendicular to the ground.
const GRIPS = {
  continental: {
    label: 'Continental',
    bevel: 2,
    face: 22, // resulting face tilt at a neutral contact: + = open
    find: 'Hold the racket on edge like a hammer you’d drive a nail with. Your index knuckle and heel pad sit on bevel 2.',
    shots: ['Serve', 'Volley', 'Overhead', 'Slice', 'Defensive digs'],
    feel: 'The “neutral tool” grip — one grip for everything at the net and above your head. Face opens naturally, which is why slices float and volleys have underspin.',
    warn: 'Rally topspin is very hard from here — that’s what the next grips are for.',
  },
  eastern: {
    label: 'Eastern',
    bevel: 3,
    face: 0,
    find: 'Place your palm flat on the strings, slide down to the handle, and shake hands with it. Index knuckle on bevel 3.',
    shots: ['Flat(ish) forehand drives', 'Fast-court tennis', 'Easy transitions to net'],
    feel: 'The classic “shake hands” forehand. Face arrives square to the ball — direct, punchy, quick to switch to continental for volleys.',
    warn: 'Less natural topspin than semi-western; high, heavy balls get awkward.',
  },
  semiwestern: {
    label: 'Semi-Western',
    bevel: 4,
    face: -14,
    find: 'From eastern, rotate your hand one bevel further under the handle (clockwise for a righty). Index knuckle on bevel 4.',
    shots: ['Modern topspin forehand', 'Rally baseline tennis', 'High bouncing balls'],
    feel: 'The modern default. The face arrives slightly closed, so brushing up the ball for topspin happens almost by itself.',
    warn: 'Very low balls and volleys need a grip change — practice the switch.',
  },
  western: {
    label: 'Western',
    bevel: 5,
    face: -28,
    find: 'Rotate one more bevel under — your palm is nearly beneath the handle. Index knuckle on bevel 5.',
    shots: ['Extreme topspin', 'High-bouncing clay courts', 'Shoulder-height rally balls'],
    feel: 'Maximum spin machinery: the face arrives well closed and the ball gets a violent upward brush. Loved on clay.',
    warn: 'Low balls and fast courts punish it, and volleys require a big grip change. Not a beginner’s starting point.',
  },
  easternbh: {
    label: 'Eastern backhand',
    bevel: 1,
    face: 0,
    find: 'From continental, rotate to put your index knuckle on top of the handle — bevel 1. Feels like revving a motorcycle.',
    shots: ['One-handed backhand drive', 'Topspin backhands', 'Kick serves (advanced)'],
    feel: 'The one-hander’s power grip: face square-to-slightly-closed on the backhand side, letting you drive through the ball.',
    warn: 'On the forehand side this grip is useless — it’s a dedicated backhand tool.',
  },
}

const OCT_CENTER = 115
const OCT_R = 66

function octagonPoints() {
  const pts = []
  for (let k = 0; k < 8; k++) {
    const a = ((k * 45 + 22.5 - 90) * Math.PI) / 180
    pts.push([OCT_CENTER + OCT_R * Math.cos(a), OCT_CENTER + OCT_R * Math.sin(a)])
  }
  return pts
}

export default function GripWheel() {
  const [gripId, setGripId] = useState('continental')
  const g = GRIPS[gripId]
  const pts = octagonPoints()

  // bevel k (1-8): edge between vertices (k-2+8)%8 and (k-1)%8... derive: bevel 1 top edge
  const bevelEdge = (n) => {
    const i = (n - 1 + 7) % 8 // vertex index left of bevel center
    return [pts[i], pts[(i + 1) % 8]]
  }
  const bevelCenter = (n) => {
    const [p1, p2] = bevelEdge(n)
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
  }
  const [bc1, bc2] = bevelEdge(g.bevel)
  const bCenter = bevelCenter(g.bevel)
  const knuckleAngle = Math.atan2(bCenter[1] - OCT_CENTER, bCenter[0] - OCT_CENTER)

  return (
    <div>
      <div className="mb-4 overflow-x-auto">
        <Segmented
          options={Object.entries(GRIPS).map(([value, v]) => ({ value, label: v.label }))}
          value={gripId}
          onChange={setGripId}
          size="sm"
        />
      </div>

      <div className="grid items-center gap-6 md:grid-cols-[230px_170px_1fr]">
        {/* handle cross-section */}
        <svg viewBox="0 0 230 230" className="mx-auto w-full max-w-[230px]" role="img" aria-label={`Handle cross-section with the ${g.label} grip highlighted`}>
          <polygon
            points={pts.map((p) => p.join(',')).join(' ')}
            fill="#f4e9dc" stroke="#7d311e" strokeWidth="2.5"
          />
          {/* bevel numbers */}
          {Array.from({ length: 8 }, (_, k) => {
            const c = bevelCenter(k + 1)
            const lx = OCT_CENTER + (c[0] - OCT_CENTER) * 1.22
            const ly = OCT_CENTER + (c[1] - OCT_CENTER) * 1.22
            return (
              <text key={k} x={lx} y={ly + 4} textAnchor="middle" fontSize="13"
                fontWeight={k + 1 === g.bevel ? 'bold' : 'normal'}
                fill={k + 1 === g.bevel ? '#b94a26' : '#8b8378'}>
                {k + 1}
              </text>
            )
          })}
          {/* highlighted bevel edge */}
          <line
            x1={bc1[0]} y1={bc1[1]} x2={bc2[0]} y2={bc2[1]}
            stroke="#cf5f38" strokeWidth="6" strokeLinecap="round"
            className="transition-all duration-500"
          />
          {/* index-knuckle marker */}
          <g
            className="transition-transform duration-500"
            transform={`translate(${bCenter[0] + Math.cos(knuckleAngle) * 13}, ${bCenter[1] + Math.sin(knuckleAngle) * 13})`}
          >
            <circle r="9" fill="#cf5f38" stroke="white" strokeWidth="2.5" />
          </g>
          <text x={OCT_CENTER} y={OCT_CENTER - 4} textAnchor="middle" fontSize="11" fill="#7d311e">handle</text>
          <text x={OCT_CENTER} y={OCT_CENTER + 10} textAnchor="middle" fontSize="9" fill="#a89f92">(from the butt cap)</text>
          {/* legend for knuckle dot */}
          <circle cx="18" cy="216" r="6" fill="#cf5f38" stroke="white" strokeWidth="2" />
          <text x="30" y="220" fontSize="10.5" fill="#6b6157">index knuckle rests here</text>
        </svg>

        {/* resulting racket face at contact */}
        <svg viewBox="0 0 170 230" className="mx-auto w-full max-w-[170px]" role="img" aria-label="Resulting racket face angle at contact">
          <text x="85" y="20" textAnchor="middle" fontSize="11" fontWeight="bold" fill="#254e37">
            Face at contact
          </text>
          {/* incoming ball */}
          <circle cx="30" cy="115" r="8" fill="#dce65a" stroke="#b3bd2d" strokeWidth="1.5" />
          <line x1="42" y1="115" x2="62" y2="115" stroke="#8bb899" strokeWidth="2" strokeDasharray="3 3" />
          {/* face */}
          <g className="transition-transform duration-500" transform={`rotate(${-g.face} 100 115)`}>
            <line x1="100" y1="60" x2="100" y2="170" stroke="#1f3f2e" strokeWidth="6" strokeLinecap="round" />
            <line x1="100" y1="170" x2="112" y2="205" stroke="#b94a26" strokeWidth="4" strokeLinecap="round" />
          </g>
          <text x="85" y="222" textAnchor="middle" fontSize="11" fill={g.face > 8 ? '#3b6ea5' : g.face < -8 ? '#b94a26' : '#5d9770'}>
            {g.face > 8 ? `open ${g.face}° — slices & serves` : g.face < -8 ? `closed ${-g.face}° — topspin brush` : 'square — flat drives'}
          </text>
        </svg>

        {/* description */}
        <div>
          <h4 className="font-display text-lg font-bold text-court-950">{g.label}</h4>
          <p className="mt-1 text-sm leading-relaxed text-court-800/90">{g.find}</p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {g.shots.map((s) => (
              <span key={s} className="rounded-full bg-court-100 px-2.5 py-0.5 text-xs font-medium text-court-700">{s}</span>
            ))}
          </div>
          <p className="mt-3 text-sm leading-relaxed text-court-700">{g.feel}</p>
          <p className="mt-2 rounded-lg bg-clay-50 px-3 py-2 text-xs leading-relaxed text-clay-800">⚠ {g.warn}</p>
        </div>
      </div>
    </div>
  )
}
