import { useState } from 'react'

const SPOTS = {
  shoulder: {
    label: 'Shoulder', x: 138, y: 78,
    injury: 'Rotator cuff irritation / swimmer’s shoulder',
    why: 'Thousands of overhead serves with a small, fatigued rotator cuff doing the braking.',
    prevent: 'Band external rotations (2×15 before play), and finish every serve — cutting the follow-through short makes the cuff absorb the deceleration.',
  },
  elbow: {
    label: 'Elbow', x: 158, y: 118,
    injury: 'Tennis elbow (lateral epicondylitis)',
    why: 'Off-center contact and armsy, body-less swings send vibration into the forearm tendons. Stiff strings and tight grips finish the job.',
    prevent: 'Fix the technique cause (use the body, not the arm), soften your string setup, loosen grip pressure to 3/10 between shots, and do slow eccentric wrist curls.',
  },
  wrist: {
    label: 'Wrist', x: 172, y: 152,
    injury: 'Wrist strain / TFCC irritation',
    why: 'Late contact forces the wrist to save the shot; extreme grips add load.',
    prevent: 'Earlier preparation (contact out front), and wrist flexor/extensor strengthening with light dumbbells.',
  },
  back: {
    label: 'Lower back', x: 112, y: 148,
    injury: 'Lumbar strain / stress reactions',
    why: 'The serve’s arch-and-rotate under load, repeated hundreds of times — especially with a toss drifting behind the head.',
    prevent: 'Fix the toss (in front!), strengthen the core with dead bugs and side planks, and warm up rotation before the first serve, not during.',
  },
  knee: {
    label: 'Knee', x: 128, y: 232,
    injury: 'Jumper’s knee / patellar tendinopathy',
    why: 'Constant low, loaded positions and explosive push-offs on hard courts.',
    prevent: 'Split squats and slow step-downs build the tendon; landing softly (bent knees) on split steps spreads the load.',
  },
  ankle: {
    label: 'Ankle', x: 118, y: 308,
    injury: 'Lateral ankle sprain',
    why: 'The classic: backpedaling for an overhead or landing on a ball. Direction changes on a planted foot.',
    prevent: 'Single-leg balance work (30s each, eyes closed to progress), sidestep — never backpedal — for lobs, and clear stray balls off your court immediately.',
  },
  calf: {
    label: 'Calf / Achilles', x: 148, y: 278,
    injury: '“Tennis leg” (calf strain) & Achilles tendinopathy',
    why: 'Explosive first steps from a cold start — the classic over-40 first-set injury.',
    prevent: 'Daily calf raises (slow down-phase), and a REAL warm-up: two minutes of jogging and skipping before the first sprint, every session.',
  },
}

export default function InjuryMap() {
  const [spot, setSpot] = useState('elbow')
  const s = SPOTS[spot]

  return (
    <div className="grid items-center gap-6 sm:grid-cols-[240px_1fr]">
      <svg viewBox="0 0 260 360" className="mx-auto w-full max-w-[240px]" role="img" aria-label="Tennis injury body map">
        <rect x="0" y="0" width="260" height="360" rx="12" fill="#f6f9f6" />
        {/* simple player silhouette, racket arm raised */}
        <g stroke="#b8d5c0" strokeWidth="14" strokeLinecap="round" fill="none">
          <line x1="125" y1="70" x2="122" y2="160" />
          <line x1="122" y1="160" x2="112" y2="235" />
          <line x1="112" y1="235" x2="116" y2="310" />
          <line x1="122" y1="160" x2="138" y2="232" />
          <line x1="138" y1="232" x2="148" y2="305" />
          <line x1="125" y1="82" x2="158" y2="118" />
          <line x1="158" y1="118" x2="172" y2="150" />
          <line x1="125" y1="82" x2="96" y2="120" />
        </g>
        <circle cx="128" cy="42" r="17" fill="#b8d5c0" />
        {/* racket in raised hand */}
        <line x1="172" y1="150" x2="188" y2="168" stroke="#8bb899" strokeWidth="5" strokeLinecap="round" />
        <ellipse cx="198" cy="180" rx="12" ry="17" transform="rotate(40 198 180)" fill="none" stroke="#8bb899" strokeWidth="4" />

        {/* hotspots */}
        {Object.entries(SPOTS).map(([id, sp]) => (
          <g key={id} onClick={() => setSpot(id)} className="cursor-pointer">
            <circle cx={sp.x} cy={sp.y} r="13" fill={id === spot ? '#cf5f38' : '#faf7f2'} stroke={id === spot ? 'white' : '#cf5f38'} strokeWidth="2.5">
              {id === spot && <animate attributeName="r" values="13;15;13" dur="1.6s" repeatCount="indefinite" />}
            </circle>
            <text x={sp.x} y={sp.y + 4} textAnchor="middle" fontSize="12" fontWeight="bold" fill={id === spot ? 'white' : '#b94a26'}>
              {id === spot ? '●' : '+'}
            </text>
          </g>
        ))}
      </svg>

      <div>
        <div className="text-xs font-bold uppercase tracking-wide text-clay-600">{s.label}</div>
        <h4 className="font-display mt-0.5 text-lg font-bold text-court-950">{s.injury}</h4>
        <p className="mt-2 text-sm leading-relaxed text-court-800/90">
          <b className="text-court-900">Why tennis causes it: </b>{s.why}
        </p>
        <p className="mt-2 rounded-lg bg-court-50 px-3 py-2 text-sm leading-relaxed text-court-800">
          <b className="text-court-700">Prevention: </b>{s.prevent}
        </p>
        <p className="mt-3 text-xs text-court-500">Tap the markers to explore each hotspot.</p>
      </div>
    </div>
  )
}
