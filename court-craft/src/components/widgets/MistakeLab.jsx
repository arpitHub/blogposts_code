import { useState } from 'react'
import { StickFigure, FigureStage, evalPhases, racketTrail, trailPath } from './figures'
import { Segmented } from '../ui'

function Panel({ title, phases, s, tone }) {
  const { pose, index } = evalPhases(phases, s)
  const trail = racketTrail(phases)
  const doneCount = Math.round(s * (trail.length - 1))
  const color = tone === 'good' ? '#2e6142' : '#9a3a1f'
  return (
    <div>
      <div className={`mb-2 flex items-center gap-2 text-sm font-bold ${tone === 'good' ? 'text-court-600' : 'text-clay-700'}`}>
        <span>{tone === 'good' ? '✓' : '✗'}</span> {title}
      </div>
      <FigureStage label={title}>
        <path d={trailPath(trail)} fill="none" stroke={color} strokeWidth="1.5" strokeDasharray="3 4" opacity="0.3" />
        <path d={trailPath(trail.slice(0, doneCount + 1))} fill="none" stroke={color} strokeWidth="2.5" opacity="0.8" />
        <StickFigure pose={pose} />
      </FigureStage>
      <div className="mt-1.5 min-h-9 text-xs leading-relaxed text-court-600">
        {phases[index].note}
      </div>
    </div>
  )
}

/**
 * Side-by-side "good form vs. common mistake" comparison with a shared scrubber.
 * `good`: { title, phases }
 * `mistakes`: [{ id, title, why, phases }]
 */
export default function MistakeLab({ good, mistakes }) {
  const [mistakeId, setMistakeId] = useState(mistakes[0].id)
  const [s, setS] = useState(0.55)
  const mistake = mistakes.find((m) => m.id === mistakeId)

  return (
    <div>
      {mistakes.length > 1 && (
        <div className="mb-4">
          <Segmented
            options={mistakes.map((m) => ({ value: m.id, label: m.title }))}
            value={mistakeId}
            onChange={setMistakeId}
            size="sm"
          />
        </div>
      )}

      <div className="grid gap-6 sm:grid-cols-2">
        <Panel title={good.title ?? 'Good form'} phases={good.phases} s={s} tone="good" />
        <Panel title={mistake.title} phases={mistake.phases} s={s} tone="bad" />
      </div>

      <div className="mt-4">
        <input
          type="range" className="cc-slider w-full"
          min={0} max={1000} value={Math.round(s * 1000)}
          onChange={(e) => setS(Number(e.target.value) / 1000)}
          aria-label="Scrub both swings together"
        />
        <div className="mt-0.5 flex justify-between text-[11px] text-court-500">
          <span>start</span><span>one slider drives both swings</span><span>finish</span>
        </div>
      </div>

      <div className="mt-4 rounded-xl border-l-4 border-clay-400 bg-clay-50 px-4 py-3 text-sm leading-relaxed text-court-900/90">
        <b className="text-clay-700">Why it hurts you: </b>{mistake.why}
      </div>
    </div>
  )
}
