import { useMemo } from 'react'
import { packParticles } from '../lib/decay'

/**
 * The app's core visual: a jar of atoms. Parent atoms glow amber; decayed
 * (daughter) atoms go dim blue. Pure function of (lifetimes, halfLife, time)
 * so it can be played *or* scrubbed. Reused in sections 1, 3 (colors) and 4.
 */

// jar geometry in viewBox units
const VB_W = 260
const VB_H = 300
const BODY = { x: 22, y: 46, w: 216, h: 240, r: 26 }
const PAD = 16

export function ParticleJar({
  lifetimes,
  halfLife,
  time,
  seed,
  parentLabel = 'parent',
  daughterLabel = 'daughter',
  compact = false,
}: {
  lifetimes: Float64Array
  halfLife: number
  time: number
  seed: number
  parentLabel?: string
  daughterLabel?: string
  compact?: boolean
}) {
  const positions = useMemo(
    () =>
      packParticles(
        lifetimes.length,
        BODY.x + PAD,
        BODY.y + PAD,
        BODY.w - PAD * 2,
        BODY.h - PAD * 2,
        seed + 1,
      ),
    [lifetimes.length, seed],
  )

  const r = compact ? 4.2 : 5.2

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      className="h-auto w-full"
      role="img"
      aria-label={`Jar of ${lifetimes.length} atoms: glowing amber circles are ${parentLabel} atoms, dim blue are ${daughterLabel} atoms`}
    >
      <defs>
        <radialGradient id={`jarGlow-${seed}`} cx="50%" cy="35%" r="75%">
          <stop offset="0%" stopColor="#f5b846" stopOpacity="0.07" />
          <stop offset="100%" stopColor="#f5b846" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* jar lid */}
      <rect
        x={VB_W / 2 - 56}
        y={24}
        width={112}
        height={20}
        rx={7}
        fill="#191925"
        stroke="#34343f"
      />
      {/* jar body */}
      <rect
        x={BODY.x}
        y={BODY.y}
        width={BODY.w}
        height={BODY.h}
        rx={BODY.r}
        fill="#0d0d14"
        stroke="#34343f"
        strokeWidth="1.5"
      />
      <rect
        x={BODY.x}
        y={BODY.y}
        width={BODY.w}
        height={BODY.h}
        rx={BODY.r}
        fill={`url(#jarGlow-${seed})`}
      />

      {positions.map((p, i) => {
        const alive = lifetimes[i] * halfLife > time
        return (
          <g key={i}>
            {/* soft halo, only while parent */}
            <circle
              className="hl-particle"
              cx={p.x}
              cy={p.y}
              r={r + 3.5}
              fill="#f5b846"
              opacity={alive ? 0.18 : 0}
            />
            <circle
              className="hl-particle"
              cx={p.x}
              cy={p.y}
              r={r}
              fill={alive ? '#f5b846' : '#1d3f66'}
              stroke={alive ? '#c98500' : '#3987e5'}
              strokeWidth={alive ? 0 : 0.8}
              strokeOpacity={0.55}
              opacity={alive ? 1 : 0.75}
            />
          </g>
        )
      })}
    </svg>
  )
}
