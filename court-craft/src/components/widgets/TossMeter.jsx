import { useState } from 'react'
import { SliderRow } from '../ui'

/**
 * Where should the toss go? Slide the toss forward/back and see what it does
 * to the serve. Side view: server on the left, net direction to the right.
 */
export default function TossMeter() {
  const [toss, setToss] = useState(30) // -100 (behind head) … +100 (far into court)

  // Toss x-position in SVG space: 0 = directly overhead (x=110)
  const ballX = 110 + toss * 0.28
  const ballY = 26

  let zone
  if (toss < -30) {
    zone = {
      name: 'Behind you',
      tone: 'bad',
      text: 'Your back must arch to reach it. All the energy goes up instead of forward — and your spine takes the load. (Advanced players toss slightly leftward-back on purpose for kick serves, but never this far.)',
    }
  } else if (toss < 15) {
    zone = {
      name: 'Directly overhead',
      tone: 'ok',
      text: 'Reachable, but neutral: you can hit up at the ball, yet there’s no forward momentum. Fine for consistent spin serves; you’re leaving free power unused on first serves.',
    }
  } else if (toss <= 70) {
    zone = {
      name: 'Into the court — the sweet spot',
      tone: 'good',
      text: 'The classic “1 o’clock” toss. Reaching for it pulls your whole body forward and up into the court — free pace, natural pronation, and you land inside the baseline ready for the next ball.',
    }
  } else {
    zone = {
      name: 'Too far forward',
      tone: 'bad',
      text: 'You’re chasing the ball — contact happens low and in front, the swing collapses into the net, and you fall forward off balance. Rein it back.',
    }
  }

  const toneCls = {
    good: 'bg-court-100 text-court-800 border-court-300',
    ok: 'bg-line text-court-700 border-court-200',
    bad: 'bg-clay-100 text-clay-800 border-clay-300',
  }[zone.tone]

  // lean of the server toward the ball
  const lean = Math.max(-14, Math.min(10, toss * 0.12))

  return (
    <div className="grid items-center gap-6 sm:grid-cols-[300px_1fr]">
      <svg viewBox="0 0 300 230" className="mx-auto w-full max-w-[300px]" role="img" aria-label="Serve toss position diagram">
        <rect x="0" y="0" width="300" height="230" rx="10" fill="#f6f9f6" />
        <rect x="0" y="205" width="300" height="25" fill="#cb6d4a" />
        <line x1="0" y1="205" x2="300" y2="205" stroke="#a34e2e" strokeWidth="2" />
        {/* baseline mark + net direction */}
        <line x1="110" y1="198" x2="110" y2="205" stroke="#faf7f2" strokeWidth="3" />
        <text x="265" y="200" fontSize="10" fill="#5d9770">net →</text>

        {/* toss zones on a dashed arc line */}
        <line x1="60" y1={ballY} x2="250" y2={ballY} stroke="#b8d5c0" strokeWidth="1.5" strokeDasharray="3 4" />
        <rect x="114" y={ballY - 9} width="46" height="18" rx="9" fill="#3f7a54" opacity="0.15" />

        {/* server: simplified, leaning toward toss */}
        <g transform={`rotate(${lean} 110 205)`}>
          <circle cx="106" cy="78" r="10" fill="#1f3f2e" />
          <line x1="106" y1="88" x2="108" y2="140" stroke="#1f3f2e" strokeWidth="5" strokeLinecap="round" />
          <polyline points="108,140 98,172 92,205" fill="none" stroke="#1f3f2e" strokeWidth="4" strokeLinecap="round" opacity="0.75" />
          <polyline points="108,140 118,172 124,205" fill="none" stroke="#1f3f2e" strokeWidth="4" strokeLinecap="round" />
          {/* toss arm pointing at ball */}
          <line x1="106" y1="96" x2={106 + (ballX - 106) * 0.35} y2="62" stroke="#1f3f2e" strokeWidth="4" strokeLinecap="round" />
          {/* racket arm in trophy */}
          <polyline points="106,96 88,86 84,62" fill="none" stroke="#1f3f2e" strokeWidth="4" strokeLinecap="round" />
          <line x1="84" y1="62" x2="79" y2="44" stroke="#b94a26" strokeWidth="3" strokeLinecap="round" />
          <ellipse cx="76" cy="34" rx="8" ry="13" transform="rotate(-15 76 34)" fill="none" stroke="#b94a26" strokeWidth="2.5" />
        </g>

        {/* ball + drop line */}
        <line x1={ballX} y1={ballY + 8} x2={ballX} y2="196" stroke="#dce65a" strokeWidth="1.5" strokeDasharray="2 4" />
        <circle cx={ballX} cy={ballY} r="7" fill="#dce65a" stroke="#b3bd2d" strokeWidth="1.5" />
      </svg>

      <div>
        <div className={`inline-block rounded-full border px-3 py-1 text-sm font-semibold ${toneCls}`}>
          {zone.name}
        </div>
        <p className="mt-3 min-h-20 text-sm leading-relaxed text-court-800/90">{zone.text}</p>
        <SliderRow
          label="Toss position"
          value={toss} onChange={setToss}
          min={-100} max={100}
          leftHint="behind your head" rightHint="deep into the court"
        />
      </div>
    </div>
  )
}
