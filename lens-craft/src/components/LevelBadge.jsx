import { LEVELS } from '../data/modules.js';

export default function LevelBadge({ level, className = '' }) {
  const info = LEVELS[level];
  if (!info) return null;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-mono uppercase tracking-wide ${className}`}
      style={{ borderColor: info.color, color: info.color }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: info.color }} />
      {info.label}
    </span>
  );
}
