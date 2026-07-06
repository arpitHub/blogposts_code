import { useEffect, useMemo, useRef, useState } from 'react'
import { simulateShot, classifySpin, COURT } from '../../lib/ballPhysics'
import { SliderRow } from '../ui'

// ---- shared side-view court geometry -------------------------------------
const W = 860
const H = 300
const GROUND = 258
const PPM = 28.5 // pixels per meter
const X0 = 48 // px of x=0 (hitter's baseline)
const sx = (m) => X0 + m * PPM
const sy = (m) => GROUND - m * PPM

const TONE_COLORS = {
  topspin: '#b94a26',
  flat: '#2e6142',
  slice: '#3b6ea5',
}

export function SideCourtSVG({ children, showZones = true }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" role="img" aria-label="Side view of a tennis court with ball trajectory">
      {/* sky */}
      <rect x="0" y="0" width={W} height={GROUND} fill="#f6f9f6" />
      {/* ground */}
      <rect x="0" y={GROUND} width={W} height={H - GROUND} fill="#cb6d4a" />
      <rect x="0" y={GROUND} width={W} height="3" fill="#a34e2e" />
      {/* valid landing zone */}
      {showZones && (
        <rect x={sx(COURT.netX)} y={GROUND - 4} width={(COURT.length - COURT.netX) * PPM} height="4" fill="#3f7a54" opacity="0.85" />
      )}
      {/* baselines + service line */}
      {[0, COURT.length].map((m) => (
        <g key={m}>
          <line x1={sx(m)} y1={GROUND - 7} x2={sx(m)} y2={GROUND} stroke="#7d311e" strokeWidth="3" />
        </g>
      ))}
      <line x1={sx(COURT.serviceLineX)} y1={GROUND - 5} x2={sx(COURT.serviceLineX)} y2={GROUND} stroke="#7d311e" strokeWidth="2" />
      {/* net */}
      <g>
        <line x1={sx(COURT.netX)} y1={sy(COURT.netHeight)} x2={sx(COURT.netX)} y2={GROUND} stroke="#1f3f2e" strokeWidth="3" />
        <line x1={sx(COURT.netX) - 5} y1={sy(COURT.netHeight)} x2={sx(COURT.netX) + 5} y2={sy(COURT.netHeight)} stroke="#1f3f2e" strokeWidth="3" />
      </g>
      {/* labels */}
      <text x={sx(0)} y={GROUND + 26} textAnchor="middle" fontSize="11" fill="#7d311e">You</text>
      <text x={sx(COURT.netX)} y={GROUND + 26} textAnchor="middle" fontSize="11" fill="#1f3f2e">Net (0.91 m)</text>
      <text x={sx(COURT.serviceLineX)} y={GROUND + 26} textAnchor="middle" fontSize="11" fill="#7d311e">Service line</text>
      <text x={sx(COURT.length)} y={GROUND + 26} textAnchor="middle" fontSize="11" fill="#7d311e">Baseline</text>
      {/* hitter */}
      <g stroke="#1f3f2e" strokeWidth="2.5" strokeLinecap="round" fill="none">
        <circle cx={sx(-0.55)} cy={sy(1.62)} r="7" fill="#1f3f2e" stroke="none" />
        <line x1={sx(-0.55)} y1={sy(1.5)} x2={sx(-0.6)} y2={sy(0.85)} />
        <line x1={sx(-0.6)} y1={sy(0.85)} x2={sx(-0.85)} y2={sy(0)} />
        <line x1={sx(-0.6)} y1={sy(0.85)} x2={sx(-0.35)} y2={sy(0)} />
        <line x1={sx(-0.55)} y1={sy(1.38)} x2={sx(-0.05)} y2={sy(1.02)} />
      </g>
      {children}
    </svg>
  )
}

function pathFrom(points) {
  return points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ')
}

const VERDICTS = {
  in: { label: 'IN — nice shot', cls: 'bg-court-500 text-white' },
  net: { label: 'NET — didn’t clear it', cls: 'bg-court-900 text-white' },
  long: { label: 'OUT — sailed long', cls: 'bg-clay-500 text-white' },
}

const PRESETS = [
  { name: 'Flat drive', speed: 100, path: 5, face: 6 },
  { name: 'Rally topspin', speed: 90, path: 30, face: 8 },
  { name: 'Heavy topspin', speed: 82, path: 45, face: 13 },
  { name: 'Defensive slice', speed: 70, path: -26, face: 14 },
  { name: 'Moonball', speed: 65, path: 36, face: 30 },
]

export default function TrajectoryLab() {
  const [speed, setSpeed] = useState(90)
  const [path, setPath] = useState(30) // swing path angle, deg; + = low-to-high
  const [face, setFace] = useState(8) // racket face openness, deg
  const [pinned, setPinned] = useState(null)
  const [clock, setClock] = useState(0)
  const [replayKey, setReplayKey] = useState(0)
  const rafRef = useRef()

  const rpm = Math.round(path * 70)
  const launchDeg = face + 0.15 * path
  const shot = useMemo(
    () => simulateShot({ speedKmh: speed, launchDeg, rpm }),
    [speed, launchDeg, rpm],
  )
  const spinClass = classifySpin(rpm)
  const tone = TONE_COLORS[spinClass.tone]

  // Replay the ball whenever the trajectory changes.
  useEffect(() => {
    const start = performance.now()
    const playRate = 0.55 // slow-mo
    const tick = (now) => {
      const t = ((now - start) / 1000) * playRate
      setClock(t)
      if (t < shot.duration + 0.4) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [shot, replayKey])

  const t = Math.min(clock, shot.duration)
  // find current ball point (points are time-ordered)
  const ball = useMemo(() => {
    const pts = shot.points
    let lo = 0
    let hi = pts.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (pts[mid].t < t) lo = mid + 1
      else hi = mid
    }
    return pts[lo]
  }, [shot, t])

  const revealed = useMemo(
    () => shot.points.filter((p) => p.t <= t),
    [shot, t],
  )
  const done = clock >= shot.duration

  const spinAngle = ((rpm / 60) * 2 * 360 * t) % 360 // exaggerated visual spin

  return (
    <div>
      {/* presets */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-court-500">Presets:</span>
        {PRESETS.map((p) => (
          <button
            key={p.name}
            onClick={() => { setSpeed(p.speed); setPath(p.path); setFace(p.face) }}
            className="rounded-full border border-line bg-court-50 px-3 py-1 text-xs font-medium text-court-700 transition hover:border-clay-400 hover:text-clay-600"
          >
            {p.name}
          </button>
        ))}
      </div>

      <SideCourtSVG>
        {/* pinned ghost */}
        {pinned && (
          <path d={pathFrom(pinned.points)} fill="none" stroke="#9aa5a0" strokeWidth="2" strokeDasharray="5 5" />
        )}
        {/* revealed trajectory */}
        <path d={pathFrom(revealed)} fill="none" stroke={tone} strokeWidth="2.5" />
        {/* full path faint once done */}
        {done && <path d={pathFrom(shot.points)} fill="none" stroke={tone} strokeWidth="2.5" opacity="0.35" />}
        {/* net clearance marker */}
        {done && shot.netClearance > 0 && (
          <g>
            <line
              x1={sx(COURT.netX)} y1={sy(COURT.netHeight)}
              x2={sx(COURT.netX)} y2={sy(COURT.netHeight + shot.netClearance)}
              stroke={tone} strokeWidth="1.5" strokeDasharray="3 3"
            />
            <text x={sx(COURT.netX) + 8} y={sy(COURT.netHeight + shot.netClearance / 2)} fontSize="11" fill={tone}>
              {shot.netClearance.toFixed(1)} m over
            </text>
          </g>
        )}
        {/* landing marker */}
        {done && shot.landingX != null && (
          <text x={sx(Math.min(shot.landingX, 26))} y={GROUND - 10} textAnchor="middle" fontSize="12" fill={tone} fontWeight="bold">
            ▼
          </text>
        )}
        {/* ball */}
        <g transform={`translate(${sx(ball.x)}, ${sy(ball.y)})`}>
          <g transform={`rotate(${rpm >= 0 ? spinAngle : -spinAngle})`}>
            <circle r="6.5" fill="#dce65a" stroke="#b3bd2d" strokeWidth="1" />
            <path d="M-6.5,0 A8,8 0 0 1 6.5,0" fill="none" stroke="#fff" strokeWidth="1.3" />
          </g>
        </g>
      </SideCourtSVG>

      {/* verdict + stats */}
      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <span className={`rounded-full px-3 py-1 text-xs font-bold tracking-wide ${VERDICTS[shot.verdict].cls} ${done ? '' : 'opacity-30'}`}>
          {done ? VERDICTS[shot.verdict].label : '…'}
        </span>
        <span className="font-semibold" style={{ color: tone }}>{spinClass.label}</span>
        <span className="text-court-600">Spin: <b>{Math.abs(rpm).toLocaleString()} rpm {rpm > 0 ? '(top)' : rpm < 0 ? '(back)' : ''}</b></span>
        {shot.netClearance != null && (
          <span className="text-court-600">Net clearance: <b>{shot.netClearance > 0 ? `${shot.netClearance.toFixed(2)} m` : 'hit the net'}</b></span>
        )}
        {shot.landingX != null && (
          <span className="text-court-600">
            Lands: <b>{shot.landingX <= COURT.length ? `${(COURT.length - shot.landingX).toFixed(1)} m inside baseline` : `${(shot.landingX - COURT.length).toFixed(1)} m long`}</b>
          </span>
        )}
        {shot.bounceApex > 0.02 && (
          <span className="text-court-600">Bounce height: <b>{shot.bounceApex.toFixed(1)} m</b></span>
        )}
      </div>

      {/* controls */}
      <div className="mt-5 grid gap-5 sm:grid-cols-3">
        <SliderRow
          label="Swing path" value={path} onChange={setPath}
          min={-30} max={45} step={1}
          format={(v) => `${v > 0 ? '+' : ''}${v}°`}
          leftHint="high → low (slice)" rightHint="low → high (topspin)"
        />
        <SliderRow
          label="Racket face at contact" value={face} onChange={setFace}
          min={0} max={32} step={1}
          format={(v) => `${v}° open`}
          leftHint="flat / closed" rightHint="open (aims higher)"
        />
        <SliderRow
          label="Shot speed" value={speed} onChange={setSpeed}
          min={55} max={120} step={1}
          format={(v) => `${v} km/h`}
          leftHint="push" rightHint="rip"
        />
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          onClick={() => setReplayKey((k) => k + 1)}
          className="rounded-lg bg-court-800 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-court-700"
        >
          ▶ Replay shot
        </button>
        <button
          onClick={() => setPinned(shot)}
          className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-court-700 transition hover:border-clay-400"
        >
          📌 Pin this shot as a ghost
        </button>
        {pinned && (
          <button
            onClick={() => setPinned(null)}
            className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-court-500 transition hover:border-clay-400"
          >
            Clear ghost
          </button>
        )}
      </div>
    </div>
  )
}

/** Static overlay comparing topspin / flat / slice at rally pace. */
export function SpinComparison() {
  const shots = useMemo(() => ([
    { name: 'Topspin', shot: simulateShot({ speedKmh: 88, launchDeg: 13, rpm: 2400 }), color: TONE_COLORS.topspin },
    { name: 'Flat', shot: simulateShot({ speedKmh: 92, launchDeg: 7, rpm: 200 }), color: TONE_COLORS.flat },
    { name: 'Slice', shot: simulateShot({ speedKmh: 72, launchDeg: 11, rpm: -1700 }), color: TONE_COLORS.slice },
  ]), [])

  return (
    <div>
      <SideCourtSVG>
        {shots.map(({ name, shot, color }) => (
          <g key={name}>
            <path d={pathFrom(shot.points)} fill="none" stroke={color} strokeWidth="2.5" opacity="0.9" />
          </g>
        ))}
      </SideCourtSVG>
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        {shots.map(({ name, shot, color }) => (
          <div key={name} className="rounded-xl border border-line bg-court-50/50 px-4 py-3">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
              <span className="font-semibold text-court-900">{name}</span>
            </div>
            <div className="mt-1 text-xs leading-relaxed text-court-600">
              Clears net by <b>{shot.netClearance.toFixed(1)} m</b> · bounces to <b>{shot.bounceApex.toFixed(1)} m</b>
              {name === 'Topspin' && ' — big margin, dips in, kicks up at your opponent.'}
              {name === 'Flat' && ' — fastest through the court, but the smallest margin for error.'}
              {name === 'Slice' && ' — floats deep and skids low, forcing the opponent to hit up.'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
