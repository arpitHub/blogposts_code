// One checklist row: custom checkbox + label, linked out when the
// item has docs. Whole row is a ≥44px tap target.
export default function ChecklistItem({ item, checked, accent, onToggle }) {
  return (
    <li>
      <label className="flex min-h-[44px] cursor-pointer items-center gap-3 rounded-lg px-2 py-1.5 transition-colors hover:bg-white/5">
        <input
          type="checkbox"
          checked={checked}
          onChange={onToggle}
          className="peer sr-only"
        />
        <span
          aria-hidden="true"
          className="flex h-5 w-5 shrink-0 items-center justify-center rounded border-2 transition-colors"
          style={{
            borderColor: checked ? accent : "#3a4150",
            backgroundColor: checked ? accent : "transparent",
          }}
        >
          {checked && (
            <svg viewBox="0 0 12 12" className="h-3 w-3 text-night" fill="none">
              <path
                d="M2 6.5L4.5 9L10 3.5"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </span>
        {item.link ? (
          <a
            href={item.link}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className={`text-sm underline decoration-dotted underline-offset-4 transition-colors hover:decoration-solid ${
              checked ? "text-slate-500 line-through" : "text-slate-200"
            }`}
            style={checked ? undefined : { textDecorationColor: accent }}
          >
            {item.label}
          </a>
        ) : (
          <span
            className={`text-sm ${
              checked ? "text-slate-500 line-through" : "text-slate-200"
            }`}
          >
            {item.label}
          </span>
        )}
      </label>
    </li>
  );
}
