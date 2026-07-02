import { useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Section } from '../components/Section'
import { D } from '../lib/depth'

const YEAR_MIN = 1896
const YEAR_MAX = 1956

interface Stop {
  year: number
  who: string
  title: string
  beginner: React.ReactNode
  technical: React.ReactNode
  scene: React.FC
}

const STOPS: Stop[] = [
  {
    year: 1896,
    who: 'Henri Becquerel',
    title: 'The fogged plates',
    beginner: (
      <>
        Becquerel wrapped photographic plates in thick black paper — no light
        could touch them — and left uranium salts on top, in a dark drawer.
        The plates fogged anyway. <strong>The surprise:</strong> the uranium
        was pouring out invisible energy all by itself, with no sunlight, no
        charging, no fuel. Nobody had ever seen matter do that.
      </>
    ),
    technical: (
      <>
        Becquerel found uranium salts exposed plates through opaque paper even
        after days in darkness, ruling out phosphorescence — emission needed no
        prior excitation and didn’t fade. The radiation also discharged
        electroscopes, giving the first quantitative handle: an ionization
        current proportional to the uranium content, independent of its
        chemical form. Energy was leaving individual atoms at a steady rate.
      </>
    ),
    scene: BecquerelScene,
  },
  {
    year: 1898,
    who: 'Marie & Pierre Curie',
    title: 'Something far stronger than uranium',
    beginner: (
      <>
        The Curies noticed pitchblende ore was <em>more</em> radioactive than
        the uranium in it could explain. So they boiled down literal tonnes of
        black rock in a leaky shed, chasing the extra glow — and cornered two
        new elements, polonium and radium. <strong>The surprise:</strong> a
        speck of radium glowed and warmed itself, endlessly, as if energy were
        free.
      </>
    ),
    technical: (
      <>
        Using the piezoelectric quartz electrometer, the Curies made
        radioactivity a measurement, not an adjective. Pitchblende’s activity
        exceeded its uranium share by ~4×; chemical fractionation traced the
        excess to radium, roughly a <strong>million times</strong> more active
        than uranium per gram. From ~1 tonne of residue they isolated ~0.1 g of
        radium chloride (1902). Radium’s steady self-heating (~100 cal/g/hr)
        posed the energy problem decay theory would soon solve.
      </>
    ),
    scene: CurieScene,
  },
  {
    year: 1905,
    who: 'Ernest Rutherford',
    title: 'Decay is a clock — read it from a rock',
    beginner: (
      <>
        Rutherford and Soddy realized radioactive atoms transform on a fixed
        “half-time” schedule, and the debris piles up. His leap: the helium
        trapped inside a uranium mineral is spent radiation — so counting it
        tells you how long the rock has been sitting there.{' '}
        <strong>The surprise:</strong> his first crude reading said the rock
        was ~500 million years old — older than physics then allowed the whole
        Sun to be.
      </>
    ),
    technical: (
      <>
        From the transformation law N(t) = N₀e<sup>−λt</sup> (Rutherford &
        Soddy, 1902), each α decay banks one helium atom in the mineral.
        Measuring the He/U ratio in fergusonite against uranium’s known
        α-production rate, Rutherford announced an age of ~500 Myr (1905) —
        a lower bound, since helium leaks. Boltwood’s Pb/U ratios (1907)
        soon gave 400–2,200 Myr, demolishing Kelvin’s ~20–100 Myr Earth.
      </>
    ),
    scene: RutherfordScene,
  },
  {
    year: 1956,
    who: 'Clair Patterson',
    title: 'The age of the Earth: 4.55 billion years',
    beginner: (
      <>
        Patterson measured lead trapped in meteorites — leftover construction
        rubble from the solar system — in a lab he scrubbed to fanatical
        cleanliness because everyday dust is full of lead.{' '}
        <strong>The surprise:</strong> five different space rocks, plus the
        Earth itself, all pointed at exactly the same birthday: 4.55 billion
        years ago. Different rocks, one answer.
      </>
    ),
    technical: (
      <>
        In the first ultraclean lab, Patterson measured Pb isotope ratios in
        five meteorites by mass spectrometry. On a ²⁰⁷Pb/²⁰⁴Pb vs ²⁰⁶Pb/²⁰⁴Pb
        plot they fell on one line — a Pb–Pb isochron — whose slope gives{' '}
        <strong>4.55 ± 0.07 Gyr</strong>. Modern ocean-sediment lead fell on
        the same line, tying Earth to the meteorites. Section 05 rebuilds this
        exact plot.
      </>
    ),
    scene: PattersonScene,
  },
]

export function DiscoverySection() {
  const [year, setYear] = useState(1896)
  const trackRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  // active stop = nearest stop at or before the playhead (first stop as floor)
  const active = STOPS.reduce((best, s) =>
    Math.abs(s.year - year) < Math.abs(best.year - year) ? s : best,
  )

  const yearToPct = (y: number) => ((y - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)) * 100

  const setFromPointer = (clientX: number) => {
    const el = trackRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
    setYear(Math.round(YEAR_MIN + frac * (YEAR_MAX - YEAR_MIN)))
  }

  return (
    <Section id="discovery" kicker="02 · The discovery" title="Sixty years from a fogged plate to the age of the Earth">
      {/* scrubbable timeline */}
      <div className="mb-8 select-none">
        <div
          ref={trackRef}
          className="relative h-16 cursor-ew-resize touch-none"
          onPointerDown={(e) => {
            dragging.current = true
            e.currentTarget.setPointerCapture(e.pointerId)
            setFromPointer(e.clientX)
          }}
          onPointerMove={(e) => dragging.current && setFromPointer(e.clientX)}
          onPointerUp={() => (dragging.current = false)}
          role="slider"
          aria-label="Timeline year"
          aria-valuemin={YEAR_MIN}
          aria-valuemax={YEAR_MAX}
          aria-valuenow={year}
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'ArrowRight') setYear((y) => Math.min(YEAR_MAX, y + 1))
            if (e.key === 'ArrowLeft') setYear((y) => Math.max(YEAR_MIN, y - 1))
          }}
        >
          {/* rail */}
          <div className="absolute left-0 right-0 top-8 h-0.5 bg-line" />
          {/* decade ticks */}
          {[1900, 1910, 1920, 1930, 1940, 1950].map((y) => (
            <div
              key={y}
              className="absolute top-7 h-2.5 w-px bg-axis"
              style={{ left: `${yearToPct(y)}%` }}
            >
              <span className="absolute -left-3 top-4 text-[10px] text-ink-3">
                {y}
              </span>
            </div>
          ))}
          {/* stops */}
          {STOPS.map((s) => (
            <button
              key={s.year}
              onClick={() => setYear(s.year)}
              className="group absolute top-8 -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${yearToPct(s.year)}%` }}
              aria-label={`${s.year}: ${s.who}`}
            >
              <span
                className={`block h-3.5 w-3.5 rounded-full border-2 transition-all ${
                  active.year === s.year
                    ? 'scale-125 border-amber-glow bg-amber-glow shadow-[0_0_10px_rgba(245,184,70,0.7)]'
                    : 'border-ink-3 bg-panel group-hover:border-ink'
                }`}
              />
              <span
                className={`absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap text-[11px] font-medium ${
                  active.year === s.year ? 'text-amber-glow' : 'text-ink-3'
                }`}
              >
                {s.year}
              </span>
            </button>
          ))}
          {/* playhead */}
          <div
            className="pointer-events-none absolute top-8 -translate-x-1/2"
            style={{ left: `${yearToPct(year)}%` }}
          >
            <div className="h-6 w-0.5 -translate-y-1/2 bg-ink/60" />
            <div className="mt-1 -translate-x-1/2 whitespace-nowrap rounded border border-line bg-panel px-1.5 py-0.5 text-[11px] tabular-nums text-ink absolute left-1/2">
              {year}
            </div>
          </div>
        </div>
        <p className="mt-6 text-xs text-ink-3">
          Drag along the timeline, or tap a stop.
        </p>
      </div>

      {/* active stop card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={active.year}
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -24 }}
          transition={{ duration: 0.3 }}
          className="grid gap-6 rounded-xl border border-line bg-panel p-5 sm:p-7 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]"
        >
          <div className="rounded-lg bg-surface p-3">
            <active.scene />
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-widest text-amber-glow">
              {active.year} · {active.who}
            </div>
            <h3 className="mb-3 mt-1 text-xl font-semibold text-ink">
              {active.title}
            </h3>
            <p className="text-[15px] leading-relaxed text-ink-2">
              <D b={active.beginner} t={active.technical} />
            </p>
          </div>
        </motion.div>
      </AnimatePresence>
    </Section>
  )
}

/* ---------- scenes: small animated SVG recreations ---------- */

function BecquerelScene() {
  return (
    <svg viewBox="0 0 300 190" className="h-auto w-full" role="img" aria-label="Uranium salts fogging a wrapped photographic plate">
      {/* drawer */}
      <rect x="18" y="20" width="264" height="150" rx="8" fill="#0d0d14" stroke="#34343f" />
      <text x="30" y="40" fill="#8a8894" fontSize="10">closed drawer — total darkness</text>
      {/* wrapped plate */}
      <rect x="60" y="95" width="180" height="58" rx="4" fill="#191925" stroke="#34343f" />
      <text x="150" y="147" fill="#8a8894" fontSize="9" textAnchor="middle">photographic plate, wrapped in black paper</text>
      {/* fog blooming on the plate */}
      <motion.ellipse
        cx="150" cy="118" rx="52" ry="18"
        fill="#f5b846"
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 0.32, 0.32] }}
        transition={{ duration: 4, times: [0, 0.7, 1], repeat: Infinity, repeatDelay: 1.2 }}
        style={{ filter: 'blur(6px)' }}
      />
      {/* uranium salt crystals */}
      <g>
        <polygon points="130,78 142,58 158,62 168,80 150,92" fill="#232330" stroke="#c98500" strokeWidth="1.2" />
        <polygon points="152,80 162,66 176,72 178,86 164,92" fill="#191925" stroke="#c98500" strokeWidth="1" />
        <text x="150" y="52" fill="#b9b7c0" fontSize="10" textAnchor="middle">uranium salts</text>
      </g>
      {/* invisible rays */}
      {[-30, -10, 10, 30].map((dx, i) => (
        <motion.line
          key={dx}
          x1={150 + dx * 0.4} y1={90} x2={150 + dx} y2={112}
          stroke="#f5b846" strokeWidth="1.5" strokeDasharray="3 3"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 0.8, 0] }}
          transition={{ duration: 1.6, delay: i * 0.3, repeat: Infinity }}
        />
      ))}
    </svg>
  )
}

function CurieScene() {
  return (
    <svg viewBox="0 0 300 190" className="h-auto w-full" role="img" aria-label="Tonnes of pitchblende reduced to a glowing vial of radium">
      {/* pitchblende mound */}
      <g>
        {[
          [50, 140, 26], [80, 130, 30], [110, 142, 24], [66, 152, 22], [95, 155, 26],
        ].map(([cx, cy, r], i) => (
          <circle key={i} cx={cx} cy={cy} r={r} fill="#14141c" stroke="#34343f" />
        ))}
        <text x="83" y="115" fill="#b9b7c0" fontSize="10" textAnchor="middle">~1 tonne of pitchblende</text>
      </g>
      {/* arrow */}
      <motion.path
        d="M 140 140 C 170 140 180 140 200 140"
        stroke="#8a8894" strokeWidth="1.5" fill="none" markerEnd="url(#arrowhead)"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.2, repeat: Infinity, repeatDelay: 2.4 }}
      />
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6" fill="none" stroke="#8a8894" />
        </marker>
      </defs>
      <text x="170" y="128" fill="#8a8894" fontSize="9" textAnchor="middle">4 years of boiling</text>
      {/* radium vial */}
      <g>
        <motion.circle
          cx="236" cy="132" r="26" fill="#7db5f0"
          animate={{ opacity: [0.12, 0.4, 0.12] }}
          transition={{ duration: 2.2, repeat: Infinity }}
          style={{ filter: 'blur(8px)' }}
        />
        <rect x="228" y="112" width="16" height="42" rx="5" fill="#0d0d14" stroke="#7db5f0" />
        <motion.rect
          x="231" y="130" width="10" height="20" rx="3" fill="#7db5f0"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2.2, repeat: Infinity }}
        />
        <text x="236" y="170" fill="#b9b7c0" fontSize="10" textAnchor="middle">0.1 g radium</text>
      </g>
      <text x="150" y="30" fill="#8a8894" fontSize="10" textAnchor="middle">activity ≈ 1,000,000 × uranium, gram for gram</text>
    </svg>
  )
}

function RutherfordScene() {
  return (
    <svg viewBox="0 0 300 190" className="h-auto w-full" role="img" aria-label="Alpha particles accumulating as helium inside a mineral">
      {/* mineral */}
      <polygon points="60,60 130,48 158,90 132,150 70,155 40,105" fill="#14141c" stroke="#34343f" strokeWidth="1.5" />
      <text x="98" y="30" fill="#b9b7c0" fontSize="10" textAnchor="middle">uranium mineral (fergusonite)</text>
      {/* U atoms inside */}
      {[
        [78, 85], [110, 72], [95, 120], [125, 105], [70, 125],
      ].map(([cx, cy], i) => (
        <g key={i}>
          <circle cx={cx} cy={cy} r={6} fill="#f5b846" opacity={0.9} />
          {/* alpha particle escaping */}
          <motion.circle
            r={3} fill="#7db5f0"
            initial={{ cx, cy, opacity: 0 }}
            animate={{ cx: cx + 18, cy: cy - 12, opacity: [0, 1, 0] }}
            transition={{ duration: 1.8, delay: i * 0.7, repeat: Infinity, repeatDelay: 1.6 }}
          />
        </g>
      ))}
      {/* helium tank gauge */}
      <g>
        <rect x="200" y="55" width="34" height="100" rx="6" fill="#0d0d14" stroke="#34343f" />
        <motion.rect
          x="204" width="26" rx="4" fill="#3987e5"
          initial={{ y: 151, height: 0 }}
          animate={{ y: 63, height: 88 }}
          transition={{ duration: 9, repeat: Infinity, ease: 'linear' }}
        />
        <text x="217" y="170" fill="#b9b7c0" fontSize="10" textAnchor="middle">trapped helium</text>
      </g>
      <text x="262" y="100" fill="#8a8894" fontSize="9" textAnchor="middle">He ÷ rate</text>
      <text x="262" y="112" fill="#8a8894" fontSize="9" textAnchor="middle">= age</text>
      <text x="262" y="130" fill="#f5b846" fontSize="11" textAnchor="middle" fontWeight="600">~500 Myr</text>
    </svg>
  )
}

function PattersonScene() {
  // mini preview of the section-5 isochron
  const pts = [
    [60, 138], [78, 128], [122, 96], [128, 99], [208, 46],
  ]
  return (
    <svg viewBox="0 0 300 190" className="h-auto w-full" role="img" aria-label="Five meteorite lead measurements falling on one line">
      {/* axes */}
      <line x1="45" y1="160" x2="270" y2="160" stroke="#34343f" />
      <line x1="45" y1="160" x2="45" y2="25" stroke="#34343f" />
      <text x="157" y="180" fill="#8a8894" fontSize="9" textAnchor="middle">²⁰⁶Pb / ²⁰⁴Pb</text>
      <text x="28" y="95" fill="#8a8894" fontSize="9" textAnchor="middle" transform="rotate(-90 28 95)">²⁰⁷Pb / ²⁰⁴Pb</text>
      {/* isochron line draws in */}
      <motion.line
        x1="50" y1="145" x2="230" y2="35"
        stroke="#c98500" strokeWidth="2"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.6, repeat: Infinity, repeatDelay: 2.8 }}
      />
      {pts.map(([cx, cy], i) => (
        <motion.circle
          key={i} cx={cx} cy={cy} r={5}
          fill="#f5b846" stroke="#0a0a0f" strokeWidth="1.5"
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 + i * 0.25, duration: 0.4, repeat: Infinity, repeatDelay: 3.6 }}
        />
      ))}
      <text x="215" y="24" fill="#f5b846" fontSize="12" fontWeight="600" textAnchor="middle">4.55 Gyr</text>
      <text x="150" y="18" fill="#8a8894" fontSize="9" textAnchor="start"></text>
    </svg>
  )
}
