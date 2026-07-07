import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { PageIntro, LevelBadge } from '../../components/ui'
import { TopCourtSVG, PlayerDot } from '../../components/widgets/TopCourt'
import { DRILLS, DRILL_STROKES } from '../../data/drills'

function DrillDiagram({ diagram, name }) {
  return (
    <TopCourtSVG label={`Court pattern for ${name}`} className="max-w-[150px] rounded-xl">
      {/* pattern path(s), shown faintly */}
      <path d={diagram.path} fill="none" stroke="#dce65a" strokeWidth="3" strokeDasharray="7 7" opacity="0.5" />
      {diagram.players.map(([x, y], i) => (
        <PlayerDot key={i} x={x} y={y} color={i === 0 ? '#cf5f38' : '#1f3f2e'} r={12} />
      ))}
      {/* animated ball riding the pattern */}
      <circle r="8" fill="#dce65a" stroke="#b3bd2d" strokeWidth="2">
        <animateMotion dur="4s" repeatCount="indefinite" path={diagram.path} />
      </circle>
    </TopCourtSVG>
  )
}

export default function Drills() {
  const [params, setParams] = useSearchParams()
  const [query, setQuery] = useState('')
  const level = params.get('level') ?? 'all'
  const stroke = params.get('stroke') ?? 'all'

  const setFilter = (key, value) => {
    const next = new URLSearchParams(params)
    if (value === 'all') next.delete(key)
    else next.set(key, value)
    setParams(next, { replace: true })
  }

  const filtered = useMemo(() => DRILLS.filter((d) => {
    if (level !== 'all' && d.level !== level) return false
    if (stroke !== 'all' && !d.strokes.includes(stroke)) return false
    if (query.trim()) {
      const q = query.toLowerCase()
      return (d.name + ' ' + d.goal + ' ' + d.how + ' ' + d.strokes.join(' ')).toLowerCase().includes(q)
    }
    return true
  }), [level, stroke, query])

  const chip = (active) =>
    `rounded-full px-3 py-1 text-xs font-medium transition ${
      active ? 'bg-clay-500 text-white' : 'border border-line bg-white text-court-600 hover:border-clay-300'
    }`

  return (
    <div className="pb-16">
      <PageIntro moduleId="drills" kicker="Progression">
        <p>
          Playing sets is fun; drills are how you improve. Every drill here has a specific
          goal, a clear success measure, and an animated court pattern so you can see the
          shape before you step on court. Filter by what you’re working on.
        </p>
      </PageIntro>

      <section className="mx-auto max-w-5xl px-6">
        {/* filters */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-2xl border border-line bg-white px-5 py-4 shadow-sm">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search drills…"
            className="w-full max-w-60 rounded-xl border border-line bg-chalk px-4 py-2 text-sm outline-none placeholder:text-court-400 focus:border-clay-400"
          />
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[11px] font-bold uppercase tracking-wide text-court-500">Level</span>
            {['all', 'beginner', 'intermediate', 'advanced'].map((l) => (
              <button key={l} onClick={() => setFilter('level', l)} className={chip(level === l)}>{l}</button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[11px] font-bold uppercase tracking-wide text-court-500">Focus</span>
            {['all', ...DRILL_STROKES].map((s) => (
              <button key={s} onClick={() => setFilter('stroke', s)} className={chip(stroke === s)}>{s}</button>
            ))}
          </div>
        </div>

        {/* results */}
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {filtered.length === 0 && (
            <p className="col-span-full py-10 text-center text-court-500">
              No drills match those filters — try loosening one.
            </p>
          )}
          {filtered.map((d) => (
            <article key={d.id} className="flex gap-4 rounded-2xl border border-line bg-white p-4 shadow-sm">
              <div className="w-32 shrink-0 sm:w-36">
                <DrillDiagram diagram={d.diagram} name={d.name} />
              </div>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="font-display text-lg font-bold text-court-950">{d.name}</h3>
                  <LevelBadge level={d.level} />
                </div>
                <p className="mt-1 text-sm font-medium text-clay-700">{d.goal}</p>
                <p className="mt-1.5 text-sm leading-relaxed text-court-700">{d.how}</p>
                <div className="mt-2 flex flex-wrap gap-1">
                  {d.strokes.map((s) => (
                    <button
                      key={s}
                      onClick={() => setFilter('stroke', s)}
                      className="rounded bg-court-50 px-1.5 py-0.5 text-[11px] text-court-600 hover:bg-court-100"
                    >
                      #{s}
                    </button>
                  ))}
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
