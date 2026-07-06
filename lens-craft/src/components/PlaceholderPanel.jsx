export default function PlaceholderPanel({ title }) {
  return (
    <div className="flex aspect-video w-full flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-line bg-panel text-center">
      <div className="grid h-12 w-12 place-items-center rounded-full border border-line-soft text-ink-3">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path
            d="M4 7a2 2 0 012-2h2l1.5-2h5L16 5h2a2 2 0 012 2v10a2 2 0 01-2 2H6a2 2 0 01-2-2V7z"
            stroke="currentColor"
            strokeWidth="1.5"
          />
          <circle cx="12" cy="13" r="3.5" stroke="currentColor" strokeWidth="1.5" />
        </svg>
      </div>
      <p className="font-mono text-xs uppercase tracking-wider text-ink-3">
        Interactive widget in development
      </p>
      <p className="max-w-xs text-sm text-ink-3">
        The {title} widget hasn't been built yet — check back soon.
      </p>
    </div>
  );
}
