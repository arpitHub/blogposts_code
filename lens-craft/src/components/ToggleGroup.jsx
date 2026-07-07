// Segmented control used for scene/mode toggles inside widgets.
export default function ToggleGroup({ options, value, onChange, className = '' }) {
  return (
    <div className={`inline-flex rounded-lg border border-line bg-panel-2 p-0.5 ${className}`}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
            value === opt.value
              ? 'bg-panel-3 text-ink'
              : 'text-ink-3 hover:text-ink-2'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
