import { useEffect, useRef, useState } from 'react'
import { StickFigure, FigureStage, evalPhases, racketTrail, trailPath } from './figures'

function Panel({ title, phases, s, color }) {
  const { pose, index } = evalPhases(phases, s)
  const trail = racketTrail(phases)
  const doneCount = Math.round(s * (trail.length - 1))
  return (
    <div>
      <div className="mb-2 text-sm font-bold text-court-800">{title}</div>
      <FigureStage label={title}>
        <path d={trailPath(trail)} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="3 4" opacity="0.3" />
        <path d={trailPath(trail.slice(0, doneCount + 1))} fill="none" stroke={color} strokeWidth="2.5" opacity="0.8" />
        <StickFigure pose={pose} />
      </FigureStage>
      <div className="mt-1.5 min-h-14 text-xs leading-relaxed text-court-600">
        <b className="text-court-800">{phases[index].name}.</b> {phases[index].note}
      </div>
    </div>
  )
}

/**
 * Two stroke variants side by side (e.g. one-handed vs. two-handed backhand),
 * driven by one shared scrubber + play button.
 */
export default function VariantCompare({ left, right }) {
  const [s, setS] = useState(0)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef()

  useEffect(() => {
    if (!playing) return
    const start = performance.now()
    const dur = 2800
    const tick = (now) => {
      const t = Math.min((now - start) / dur, 1)
      setS(t)
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
      else setPlaying(false)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing])

  return (
    <div>
      <div className="grid gap-6 sm:grid-cols-2">
        <Panel title={left.title} phases={left.phases} s={s} color="#b94a26" />
        <Panel title={right.title} phases={right.phases} s={s} color="#2e6142" />
      </div>
      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={() => { if (!playing) { setS(0); setPlaying(true) } else setPlaying(false) }}
          className="shrink-0 rounded-lg bg-court-800 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-court-700"
        >
          {playing ? '⏸ Pause' : '▶ Play both'}
        </button>
        <input
          type="range" className="cc-slider w-full"
          min={0} max={1000} value={Math.round(s * 1000)}
          onChange={(e) => { setPlaying(false); setS(Number(e.target.value) / 1000) }}
          aria-label="Scrub both variants together"
        />
      </div>
    </div>
  )
}
