import { exposureLabel } from './exposureMath.js';

const RANGE = 3; // -3..+3 stops shown on the meter

export default function ExposureMeter({ totalStops }) {
  const clamped = Math.max(-RANGE, Math.min(RANGE, totalStops));
  const percent = ((clamped + RANGE) / (RANGE * 2)) * 100;
  const label = exposureLabel(totalStops);

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm font-medium text-ink-2">Exposure meter</span>
        <span className="font-mono text-sm" style={{ color: label.color }}>
          {totalStops >= 0 ? '+' : ''}
          {totalStops.toFixed(1)} EV · {label.text}
        </span>
      </div>
      <div className="relative h-2.5 rounded-full bg-panel-3">
        <div className="absolute inset-y-0 left-1/2 w-px bg-ink-3/40" />
        <div
          className="absolute top-1/2 h-4 w-4 -translate-y-1/2 -translate-x-1/2 rounded-full border-2 border-void"
          style={{ left: `${percent}%`, background: label.color }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-3">
        <span>-3</span>
        <span>0</span>
        <span>+3</span>
      </div>
    </div>
  );
}
