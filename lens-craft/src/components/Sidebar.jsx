import { NavLink } from 'react-router-dom';
import { modulesByLevel } from '../data/modules.js';

export default function Sidebar({ onNavigate }) {
  const groups = modulesByLevel();

  return (
    <nav className="flex h-full flex-col gap-6 overflow-y-auto px-4 py-6">
      <NavLink
        to="/"
        onClick={onNavigate}
        className="flex items-center gap-2 px-2 font-display text-lg font-semibold tracking-tight text-ink"
      >
        <span className="grid h-7 w-7 place-items-center rounded-full border border-accent text-accent">
          <ApertureIcon />
        </span>
        Lens Craft
      </NavLink>

      {groups.map((group) => (
        <div key={group.level}>
          <div
            className="mb-2 flex items-center gap-2 px-2 font-mono text-[11px] uppercase tracking-wider text-ink-3"
          >
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: group.color }} />
            {group.label}
          </div>
          <ul className="flex flex-col gap-0.5">
            {group.modules.map((m) => (
              <li key={m.slug}>
                <NavLink
                  to={`/learn/${m.slug}`}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    `flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors ${
                      isActive
                        ? 'bg-panel-2 text-ink'
                        : 'text-ink-2 hover:bg-panel hover:text-ink'
                    }`
                  }
                >
                  <span className="truncate">{m.title}</span>
                  {!m.ready && (
                    <span className="shrink-0 rounded border border-line px-1 text-[10px] font-mono text-ink-3">
                      soon
                    </span>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

function ApertureIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M12 12L16 5.5M12 12L20 12M12 12L16 18.5M12 12L8 18.5M12 12L4 12M12 12L8 5.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
