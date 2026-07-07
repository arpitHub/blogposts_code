import { useEffect, useRef, useState } from 'react'
import { StickFigure, FigureStage, evalPhases, racketTrail, trailPath } from './figures'
import { Segmented } from '../ui'

/**
 * The core stroke-technique widget: scrub through a stroke phase by phase.
 * `phases`: [{ name, note, pose }]
 * Optional `variants`: [{ id, label, phases }] renders a toggle (e.g. 1H/2H backhand).
 */
export default function PhaseExplorer({ phases: basePhases, variants }) {
  const [variantId, setVariantId] = useState(variants?.[0]?.id)
  const phases = variants ? variants.find((v) => v.id === variantId).phases : basePhases

  const [s, setS] = useState(0)
  const [playing, setPlaying] = useState(false)
  const rafRef = useRef()

  // Play: sweep s from 0 → 1 over ~2.6s
  useEffect(() => {
    if (!playing) return
    const start = performance.now()
    const dur = 2600
    const tick = (now) => {
      const t = Math.min((now - start) / dur, 1)
      setS(t)
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
      else setPlaying(false)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing, phases])

  const { pose, index } = evalPhases(phases, s)
  const trail = racketTrail(phases)
  const doneCount = Math.round(s * (trail.length - 1))
  const current = phases[index]

  return (
    <div>
      {variants && (
        <div className="mb-4">
          <Segmented
            options={variants.map((v) => ({ value: v.id, label: v.label }))}
            value={variantId}
            onChange={(id) => { setVariantId(id); setS(0); setPlaying(false) }}
          />
        </div>
      )}

      <div className="grid gap-5 md:grid-cols-[minmax(0,340px)_1fr]">
        <div>
          <FigureStage label={`Stroke animation, phase: ${current.name}`}>
            {/* full racket path, faint */}
            <path d={trailPath(trail)} fill="none" stroke="#cf5f38" strokeWidth="1.5" strokeDasharray="3 4" opacity="0.35" />
            {/* covered portion */}
            <path d={trailPath(trail.slice(0, doneCount + 1))} fill="none" stroke="#cf5f38" strokeWidth="2.5" opacity="0.8" />
            <StickFigure pose={pose} />
          </FigureStage>

          <div className="mt-3 flex items-center gap-3">
            <button
              onClick={() => { if (!playing) { setS(0); setPlaying(true) } else setPlaying(false) }}
              className="shrink-0 rounded-lg bg-court-800 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-court-700"
            >
              {playing ? '⏸ Pause' : '▶ Play swing'}
            </button>
            <input
              type="range" className="cc-slider w-full"
              min={0} max={1000} value={Math.round(s * 1000)}
              onChange={(e) => { setPlaying(false); setS(Number(e.target.value) / 1000) }}
              aria-label="Scrub through the stroke"
            />
          </div>
        </div>

        <div>
          {/* phase chips */}
          <div className="flex flex-wrap gap-1.5">
            {phases.map((p, i) => (
              <button
                key={p.name}
                onClick={() => { setPlaying(false); setS(i / (phases.length - 1)) }}
                className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                  i === index
                    ? 'bg-clay-500 text-white shadow-sm'
                    : 'border border-line bg-court-50 text-court-600 hover:border-clay-300'
                }`}
              >
                {i + 1}. {p.name}
              </button>
            ))}
          </div>

          <div className="mt-4 rounded-xl border border-line bg-court-50/50 px-4 py-3">
            <div className="text-xs font-bold uppercase tracking-wide text-clay-600">
              Phase {index + 1} of {phases.length} — {current.name}
            </div>
            <p className="mt-1.5 text-sm leading-relaxed text-court-900/90">{current.note}</p>
            {current.checkpoint && (
              <p className="mt-2 text-xs font-medium text-court-600">
                ✓ Checkpoint: {current.checkpoint}
              </p>
            )}
          </div>

          <p className="mt-3 text-xs text-court-500">
            Drag the slider to scrub the swing frame by frame — the orange trace is the path of the racket head.
          </p>
        </div>
      </div>
    </div>
  )
}
