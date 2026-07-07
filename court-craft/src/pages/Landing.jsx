import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { CATEGORIES, MODULES, BEGINNER_PATH, byId, byCategory } from '../data/modules'
import { LevelBadge } from '../components/ui'

function ModuleCard({ mod }) {
  return (
    <Link
      to={mod.path}
      className="group flex flex-col rounded-2xl border border-line bg-white p-5 shadow-sm transition hover:-translate-y-0.5 hover:border-clay-300 hover:shadow-md"
    >
      <div className="mb-3 flex items-center justify-between">
        <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-court-50 text-lg text-court-700">
          {mod.icon}
        </span>
        <LevelBadge level={mod.level} />
      </div>
      <h3 className="font-display text-lg font-bold text-court-950 group-hover:text-clay-600">
        {mod.title} {mod.flagship && <span className="text-sm text-clay-500" title="Flagship interactive">★</span>}
      </h3>
      <p className="mt-1.5 text-sm leading-relaxed text-court-700/80">{mod.blurb}</p>
    </Link>
  )
}

export default function Landing() {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    if (!query.trim()) return null
    const q = query.toLowerCase()
    return MODULES.filter((m) => (m.title + ' ' + m.blurb + ' ' + m.category).toLowerCase().includes(q))
  }, [query])

  return (
    <div className="pb-20">
      {/* Hero */}
      <div className="relative overflow-hidden bg-court-900 px-6 py-16 lg:py-24">
        {/* Court-line motif */}
        <svg className="pointer-events-none absolute inset-0 h-full w-full opacity-[0.08]" viewBox="0 0 800 400" preserveAspectRatio="xMidYMid slice" aria-hidden>
          <g stroke="white" strokeWidth="3" fill="none">
            <rect x="80" y="40" width="640" height="320" />
            <line x1="80" y1="200" x2="720" y2="200" strokeDasharray="10 10" />
            <rect x="180" y="40" width="440" height="320" />
            <line x1="400" y1="115" x2="400" y2="285" />
            <line x1="180" y1="115" x2="620" y2="115" />
            <line x1="180" y1="285" x2="620" y2="285" />
          </g>
        </svg>
        <div className="relative mx-auto max-w-4xl">
          <h1 className="font-display text-4xl font-bold tracking-tight text-white lg:text-6xl">
            Learn tennis by <span className="text-clay-400">seeing it work.</span>
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-court-100/90">
            Court Craft explains tennis with interactive diagrams you can grab, scrub, and
            experiment with — not walls of text. Drag a slider and watch topspin bend a ball
            into the court. Step through a serve one phase at a time. Play out a scoring
            sequence until deuce finally clicks.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link to="/scoring" className="rounded-xl bg-clay-500 px-6 py-3 font-semibold text-white shadow transition hover:bg-clay-600">
              Start from zero →
            </Link>
            <Link to="/spin" className="rounded-xl border border-court-500 px-6 py-3 font-semibold text-court-50 transition hover:bg-court-800">
              ★ Try the spin explorer
            </Link>
          </div>
        </div>
      </div>

      {/* Start-here path */}
      <section className="mx-auto max-w-5xl px-6 pt-12">
        <h2 className="font-display text-2xl font-bold text-court-950">New to tennis? Start here.</h2>
        <p className="mt-1 text-court-700/80">A linear path from “what does 15–love mean” to your first intentional topspin.</p>
        <div className="mt-5 flex gap-2 overflow-x-auto pb-3">
          {BEGINNER_PATH.map((id, i) => {
            const m = byId(id)
            return (
              <div key={id} className="flex shrink-0 items-center gap-2">
                <Link
                  to={m.path}
                  className="group flex shrink-0 items-center gap-2 rounded-full border border-line bg-white py-2 pl-2 pr-4 shadow-sm transition hover:border-clay-300"
                >
                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-court-800 text-xs font-bold text-white group-hover:bg-clay-500">
                    {i + 1}
                  </span>
                  <span className="text-sm font-medium text-court-900">{m.title}</span>
                </Link>
                {i < BEGINNER_PATH.length - 1 && <span className="shrink-0 text-court-300">→</span>}
              </div>
            )
          })}
        </div>
      </section>

      {/* Search + module grid */}
      <section className="mx-auto max-w-5xl px-6 pt-12">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <h2 className="font-display text-2xl font-bold text-court-950">Browse all modules</h2>
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search modules… (e.g. “serve”, “deuce”, “topspin”)"
            className="w-full max-w-sm rounded-xl border border-line bg-white px-4 py-2.5 text-sm shadow-sm outline-none placeholder:text-court-400 focus:border-clay-400"
          />
        </div>

        {filtered ? (
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {filtered.length === 0 && (
              <p className="col-span-full py-8 text-center text-court-500">No modules match “{query}”.</p>
            )}
            {filtered.map((m) => <ModuleCard key={m.id} mod={m} />)}
          </div>
        ) : (
          CATEGORIES.map((cat) => (
            <div key={cat.id} className="mt-10">
              <div className="mb-4 flex items-baseline gap-3">
                <h3 className="font-display text-xl font-bold text-court-900">{cat.label}</h3>
                <span className="hidden text-sm text-court-500 sm:inline">{cat.blurb}</span>
              </div>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {byCategory(cat.id).map((m) => <ModuleCard key={m.id} mod={m} />)}
              </div>
            </div>
          ))
        )}
      </section>
    </div>
  )
}
