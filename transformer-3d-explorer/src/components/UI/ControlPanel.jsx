import useAppStore from '../../store/useAppStore'
import { ARCHITECTURES } from '../../data/architectureData'

function ToggleGroup({ label, value, options, onChange }) {
  return (
    <div
      className="flex items-center gap-1 rounded-lg bg-slate-900/80 p-1 ring-1 ring-slate-700/60 backdrop-blur"
      role="group"
      aria-label={label}
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          aria-pressed={value === opt.value}
          className={`rounded-md px-2.5 py-1.5 text-xs font-semibold transition-colors sm:text-sm ${
            value === opt.value
              ? 'bg-indigo-600 text-white shadow'
              : 'text-slate-300 hover:bg-slate-700/60'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

/**
 * Top bar: app title + architecture toggle + explore-mode toggle.
 */
export default function ControlPanel() {
  const architectureMode = useAppStore((s) => s.architectureMode)
  const exploreMode = useAppStore((s) => s.exploreMode)
  const setArchitectureMode = useAppStore((s) => s.setArchitectureMode)
  const setExploreMode = useAppStore((s) => s.setExploreMode)

  return (
    <header className="pointer-events-none absolute inset-x-0 top-0 z-30 flex flex-wrap items-center justify-between gap-2 p-3">
      <div className="pointer-events-auto rounded-lg bg-slate-900/80 px-3 py-1.5 ring-1 ring-slate-700/60 backdrop-blur">
        <h1 className="text-sm font-bold text-white sm:text-base">
          Transformer 3D Explorer
        </h1>
        <p className="hidden text-[11px] text-slate-400 sm:block">
          {ARCHITECTURES[architectureMode].subtitle}
        </p>
      </div>

      <div className="pointer-events-auto flex flex-wrap gap-2">
        <ToggleGroup
          label="Architecture"
          value={architectureMode}
          onChange={setArchitectureMode}
          options={[
            { value: 'encoder-decoder', label: 'Original Transformer' },
            { value: 'decoder-only', label: 'GPT-style' }
          ]}
        />
        <ToggleGroup
          label="Exploration mode"
          value={exploreMode}
          onChange={setExploreMode}
          options={[
            { value: 'tour', label: 'Guided Tour' },
            { value: 'free', label: 'Free Explore' }
          ]}
        />
      </div>
    </header>
  )
}
