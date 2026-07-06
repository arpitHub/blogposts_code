// Reusable camera-dial slider: label + live value readout + range input.
// Used across modules wherever a "drag this, watch the photo change" control is needed.
export default function Slider({ label, icon, value, display, min, max, step = 1, onChange, color }) {
  const percent = ((value - min) / (max - min)) * 100;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-sm font-medium text-ink-2">
          {icon}
          {label}
        </span>
        <span
          className="rounded-md border px-2 py-0.5 font-mono text-sm tabular-nums"
          style={{ borderColor: color, color }}
        >
          {display}
        </span>
      </div>
      <input
        type="range"
        className="dial"
        style={{ '--slider-color': color, '--slider-fill': `${percent}%` }}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
