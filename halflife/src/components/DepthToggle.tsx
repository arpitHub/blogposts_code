import type { Depth } from '../lib/depth'

export function DepthToggle({
  depth,
  onChange,
}: {
  depth: Depth
  onChange: (d: Depth) => void
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Explanation depth"
      className="flex items-center gap-1 rounded-full border border-line bg-panel p-1"
    >
      <span className="hidden pl-2 pr-1 text-xs text-ink-3 sm:inline">
        Explain like I’m…
      </span>
      {(['beginner', 'technical'] as const).map((d) => (
        <button
          key={d}
          role="radio"
          aria-checked={depth === d}
          onClick={() => onChange(d)}
          className={`rounded-full px-3 py-1 text-xs font-semibold capitalize transition-colors ${
            depth === d
              ? 'bg-amber-glow text-void'
              : 'text-ink-2 hover:text-ink'
          }`}
        >
          {d === 'beginner' ? 'Beginner' : 'Technical'}
        </button>
      ))}
    </div>
  )
}
