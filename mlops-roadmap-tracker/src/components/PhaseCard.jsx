import { useState } from "react";
import ProgressBar from "./ProgressBar.jsx";
import ChecklistItem from "./ChecklistItem.jsx";

// Hex accent + alpha suffix gives the tinted backgrounds/borders
// ("1a" ≈ 10%, "33" ≈ 20%, "4d" ≈ 30%).
export default function PhaseCard({ phase, checkedIds, onToggleItem }) {
  const [open, setOpen] = useState(true);
  const done = phase.items.filter((item) => checkedIds.has(item.id)).length;
  const total = phase.items.length;

  return (
    <section
      className="overflow-hidden rounded-2xl border bg-panel"
      style={{
        borderColor: `${phase.accent}4d`,
        backgroundImage: `linear-gradient(160deg, ${phase.accent}14, transparent 40%)`,
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-start gap-3 px-4 py-4 text-left sm:px-5"
      >
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="rounded-full border px-2.5 py-0.5 text-xs font-semibold tracking-wide"
              style={{
                color: phase.accent,
                borderColor: `${phase.accent}66`,
                backgroundColor: `${phase.accent}1a`,
              }}
            >
              {phase.dayRange}
            </span>
            <h2 className="text-base font-semibold text-slate-100 sm:text-lg">
              {phase.title}
            </h2>
          </div>
          <p className="mt-1 text-sm text-slate-400">{phase.description}</p>
        </div>
        <span
          aria-hidden="true"
          className={`mt-1 shrink-0 text-slate-400 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
        >
          <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none">
            <path
              d="M3 6l5 5 5-5"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>

      <div className="flex items-center gap-3 px-4 pb-3 sm:px-5">
        <ProgressBar value={done} max={total} color={phase.accent} />
        <span
          className="shrink-0 text-xs font-semibold tabular-nums"
          style={{ color: phase.accent }}
        >
          {done}/{total}
        </span>
      </div>

      {open && (
        <ul className="border-t border-white/5 px-2 py-2 sm:px-3">
          {phase.items.map((item) => (
            <ChecklistItem
              key={item.id}
              item={item}
              accent={phase.accent}
              checked={checkedIds.has(item.id)}
              onToggle={() => onToggleItem(phase.id, item.id)}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
