// Stick-figure rendering + pose interpolation for the stroke-technique
// explorers. Poses live in a 240x220 box; ground is y=205; the player hits
// toward the right (the net side).

export const FIG_W = 240
export const FIG_H = 220
export const GROUND_Y = 205

const lerp = (a, b, t) => a + (b - a) * t
const lerpPt = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)]

/** Interpolate between two poses. Joints present in both are lerped. */
export function lerpPose(a, b, t) {
  const out = {}
  for (const k of Object.keys(a)) {
    if (Array.isArray(a[k]) && Array.isArray(b[k])) out[k] = lerpPt(a[k], b[k], t)
    else out[k] = a[k]
  }
  return out
}

/**
 * Evaluate a keyframe sequence at s ∈ [0,1] (keyframes evenly spaced).
 * Returns { pose, index } where index = nearest keyframe.
 */
export function evalPhases(phases, s) {
  const n = phases.length
  const f = Math.min(Math.max(s, 0), 1) * (n - 1)
  const i = Math.min(Math.floor(f), n - 2)
  const t = f - i
  // ease within each segment so motion pauses subtly at keyframes
  const te = t * t * (3 - 2 * t)
  return {
    pose: lerpPose(phases[i].pose, phases[i + 1].pose, te),
    index: Math.round(f),
    segT: t,
  }
}

/** Sample the racket-tip path across the whole sequence (for the trail). */
export function racketTrail(phases, steps = 72) {
  const pts = []
  for (let k = 0; k <= steps; k++) {
    const { pose } = evalPhases(phases, k / steps)
    if (pose.racketTip) pts.push(pose.racketTip)
  }
  return pts
}

export function trailPath(pts) {
  return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')
}

function Racket({ wrist, tip, color = '#b94a26' }) {
  const dx = tip[0] - wrist[0]
  const dy = tip[1] - wrist[1]
  const len = Math.hypot(dx, dy) || 1
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI
  // head ellipse sits at the far 40% of the wrist→tip segment
  const cx = wrist[0] + dx * 0.78
  const cy = wrist[1] + dy * 0.78
  return (
    <g>
      <line
        x1={wrist[0]} y1={wrist[1]}
        x2={wrist[0] + dx * 0.62} y2={wrist[1] + dy * 0.62}
        stroke={color} strokeWidth="3" strokeLinecap="round"
      />
      <ellipse
        cx={cx} cy={cy} rx={len * 0.24} ry={len * 0.13}
        transform={`rotate(${angle} ${cx} ${cy})`}
        fill="none" stroke={color} strokeWidth="2.5"
      />
    </g>
  )
}

/**
 * Render a single pose. Expected joints:
 * head, neck, hip, elbowR, wristR, elbowL, wristL, kneeF, ankleF, kneeB, ankleB, racketTip
 * Optional: ball [x,y]
 */
export function StickFigure({ pose, color = '#1f3f2e', racketColor = '#b94a26', opacity = 1 }) {
  const p = pose
  const limb = { stroke: color, strokeWidth: 4, strokeLinecap: 'round', fill: 'none' }
  return (
    <g opacity={opacity}>
      {/* back arm drawn first so it sits behind the torso */}
      {p.elbowL && p.wristL && (
        <polyline points={`${p.neck} ${p.elbowL} ${p.wristL}`} {...limb} strokeWidth="3.5" opacity="0.55" />
      )}
      {/* legs */}
      <polyline points={`${p.hip} ${p.kneeB} ${p.ankleB}`} {...limb} opacity="0.75" />
      <polyline points={`${p.hip} ${p.kneeF} ${p.ankleF}`} {...limb} />
      {/* torso + head */}
      <line x1={p.hip[0]} y1={p.hip[1]} x2={p.neck[0]} y2={p.neck[1]} {...limb} strokeWidth="5" />
      <circle cx={p.head[0]} cy={p.head[1]} r="10" fill={color} />
      {/* racket arm */}
      {p.elbowR && p.wristR && (
        <polyline points={`${p.neck} ${p.elbowR} ${p.wristR}`} {...limb} />
      )}
      {p.wristR && p.racketTip && <Racket wrist={p.wristR} tip={p.racketTip} color={racketColor} />}
      {/* ball */}
      {p.ball && (
        <g>
          <circle cx={p.ball[0]} cy={p.ball[1]} r="5.5" fill="#dce65a" stroke="#b3bd2d" strokeWidth="1" />
        </g>
      )}
    </g>
  )
}

/** Court-floor backdrop shared by the figure panels. */
export function FigureStage({ children, width = FIG_W, label }) {
  return (
    <svg viewBox={`0 0 ${width} ${FIG_H}`} className="w-full select-none" role="img" aria-label={label}>
      <rect x="0" y="0" width={width} height={FIG_H} fill="#f6f9f6" rx="8" />
      <rect x="0" y={GROUND_Y} width={width} height={FIG_H - GROUND_Y} fill="#cb6d4a" />
      <line x1="0" y1={GROUND_Y} x2={width} y2={GROUND_Y} stroke="#a34e2e" strokeWidth="2" />
      {children}
    </svg>
  )
}
