import type { ReactNode } from 'react'

export function Btn({
  onClick,
  children,
  primary = false,
  disabled = false,
  ariaLabel,
}: {
  onClick: () => void
  children: ReactNode
  primary?: boolean
  disabled?: boolean
  ariaLabel?: string
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className={`rounded-md px-3.5 py-1.5 text-sm font-medium transition-colors disabled:opacity-40 ${
        primary
          ? 'bg-amber-glow text-void hover:bg-amber-series'
          : 'border border-line bg-panel text-ink-2 hover:bg-panel-2 hover:text-ink'
      }`}
    >
      {children}
    </button>
  )
}

export function SliderRow({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  display,
  blue = false,
}: {
  label: ReactNode
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  display?: string
  blue?: boolean
}) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between gap-3 text-sm">
        <span className="text-ink-2">{label}</span>
        {display && (
          <span className="font-medium tabular-nums text-ink">{display}</span>
        )}
      </div>
      <input
        type="range"
        className={`hl-range ${blue ? 'hl-range-blue' : ''}`}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

export function Stat({
  label,
  value,
  accent,
}: {
  label: string
  value: string
  accent?: 'amber' | 'blue'
}) {
  return (
    <div className="rounded-lg border border-line bg-panel px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-ink-3">
        {label}
      </div>
      <div
        className={`text-lg font-semibold tabular-nums ${
          accent === 'amber'
            ? 'text-amber-glow'
            : accent === 'blue'
              ? 'text-blue-glow'
              : 'text-ink'
        }`}
      >
        {value}
      </div>
    </div>
  )
}

/** Legend swatch + label — identity is never carried by color alone */
export function LegendItem({
  color,
  label,
  dashed = false,
}: {
  color: string
  label: string
  dashed?: boolean
}) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-ink-2">
      {dashed ? (
        <svg width="18" height="4" aria-hidden>
          <line
            x1="0"
            y1="2"
            x2="18"
            y2="2"
            stroke={color}
            strokeWidth="2"
            strokeDasharray="4 3"
          />
        </svg>
      ) : (
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ background: color }}
        />
      )}
      {label}
    </span>
  )
}

/** Monospace-ish equation card shown only in technical mode */
export function EqCard({
  children,
  note,
}: {
  children: ReactNode
  note?: ReactNode
}) {
  return (
    <div className="rounded-lg border border-line bg-panel px-4 py-3">
      <div className="font-mono text-[15px] leading-relaxed text-ink">
        {children}
      </div>
      {note && <div className="mt-1.5 text-xs text-ink-3">{note}</div>}
    </div>
  )
}
