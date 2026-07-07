import { Link } from 'react-router-dom'
import { LEVELS, byId, nextModule } from '../data/modules'

export function LevelBadge({ level }) {
  const l = LEVELS[level] ?? LEVELS.all
  return <span className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium ${l.color}`}>{l.label}</span>
}

/** Consistent page header: category chip, title, "why this matters" intro. */
export function PageIntro({ moduleId, kicker, children }) {
  const mod = byId(moduleId)
  return (
    <header className="mx-auto max-w-3xl px-6 pt-10 pb-6 lg:pt-14">
      <div className="mb-3 flex items-center gap-2">
        <LevelBadge level={mod?.level} />
        {kicker && <span className="text-xs uppercase tracking-widest text-court-500">{kicker}</span>}
      </div>
      <h1 className="font-display text-4xl font-bold tracking-tight text-court-950 lg:text-5xl">
        {mod?.title}
      </h1>
      <div className="mt-4 text-lg leading-relaxed text-court-800/90">{children}</div>
    </header>
  )
}

/** Card wrapper for an interactive widget, with a title bar and hint line. */
export function WidgetFrame({ title, hint, children, wide = false }) {
  return (
    <section className={`mx-auto my-8 px-4 sm:px-6 ${wide ? 'max-w-5xl' : 'max-w-3xl'}`}>
      <div className="overflow-hidden rounded-2xl border border-line bg-white shadow-sm">
        <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-line bg-court-50/60 px-5 py-3">
          <h3 className="font-semibold text-court-900">{title}</h3>
          {hint && <span className="text-xs text-court-500">↳ {hint}</span>}
        </div>
        <div className="p-5">{children}</div>
      </div>
    </section>
  )
}

/** Prose section constrained to reading width. */
export function Prose({ title, children }) {
  return (
    <section className="mx-auto max-w-3xl px-6 py-4">
      {title && <h2 className="font-display mb-3 text-2xl font-bold text-court-950">{title}</h2>}
      <div className="space-y-4 leading-relaxed text-court-900/90">{children}</div>
    </section>
  )
}

/** Aside/callout box for tidbits like "why is it called love?" */
export function Aside({ title, children }) {
  return (
    <div className="my-4 rounded-xl border-l-4 border-clay-400 bg-clay-50 px-5 py-4">
      {title && <div className="mb-1 text-sm font-bold uppercase tracking-wide text-clay-700">{title}</div>}
      <div className="text-[15px] leading-relaxed text-court-900/90">{children}</div>
    </div>
  )
}

/** "Try this" drill callout used at the bottom of stroke pages. */
export function TryThis({ title = 'Try this on court', children }) {
  return (
    <section className="mx-auto max-w-3xl px-6 py-4">
      <div className="rounded-2xl bg-court-800 px-6 py-5 text-court-50">
        <div className="mb-1.5 flex items-center gap-2 text-sm font-bold uppercase tracking-wide text-clay-300">
          <span>◍</span> {title}
        </div>
        <div className="leading-relaxed text-court-100">{children}</div>
      </div>
    </section>
  )
}

/** Footer link to the next logical module. */
export function NextUp({ moduleId, overrideId }) {
  const next = overrideId ? byId(overrideId) : nextModule(moduleId)
  if (!next) return null
  return (
    <section className="mx-auto max-w-3xl px-6 pb-16 pt-6">
      <Link
        to={next.path}
        className="group flex items-center justify-between rounded-2xl border border-line bg-white px-6 py-5 shadow-sm transition hover:border-clay-300 hover:shadow"
      >
        <div>
          <div className="text-xs uppercase tracking-widest text-court-500">Next up</div>
          <div className="font-display text-xl font-bold text-court-950 group-hover:text-clay-600">
            {next.title}
          </div>
        </div>
        <span className="text-2xl text-clay-500 transition group-hover:translate-x-1">→</span>
      </Link>
    </section>
  )
}

/** Segmented control used across widgets. */
export function Segmented({ options, value, onChange, size = 'md' }) {
  return (
    <div className="inline-flex flex-wrap rounded-lg border border-line bg-court-50 p-1">
      {options.map((opt) => {
        const o = typeof opt === 'string' ? { value: opt, label: opt } : opt
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={`rounded-md font-medium transition ${
              size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-1.5 text-sm'
            } ${
              value === o.value ? 'bg-court-800 text-white shadow-sm' : 'text-court-600 hover:text-court-900'
            }`}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

/** Labeled slider row. */
export function SliderRow({ label, value, onChange, min, max, step = 1, format, leftHint, rightHint }) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <label className="text-sm font-medium text-court-800">{label}</label>
        {format && <span className="rounded bg-court-50 px-2 py-0.5 font-mono text-xs text-court-700">{format(value)}</span>}
      </div>
      <input
        type="range" className="cc-slider w-full"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {(leftHint || rightHint) && (
        <div className="mt-0.5 flex justify-between text-[11px] text-court-500">
          <span>{leftHint}</span><span>{rightHint}</span>
        </div>
      )}
    </div>
  )
}
