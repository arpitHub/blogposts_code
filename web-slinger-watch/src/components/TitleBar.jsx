// Small original web-corner glyph used as the title mark.
function WebMark() {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0 text-teal">
      <g
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      >
        <path d="M2 2 L22 22" />
        <path d="M2 2 L2 22" />
        <path d="M2 2 L22 2" />
        <path d="M2 2 L12 22" />
        <path d="M2 2 L22 12" />
        <path d="M2 8 Q8 8 8 2" />
        <path d="M2 14 Q14 14 14 2" />
        <path d="M2 20 Q20 20 20 2" />
      </g>
    </svg>
  );
}

export default function TitleBar({ tagline }) {
  return (
    <header className="flex items-center gap-3 border-b border-grid bg-navy/90 px-4 py-3">
      <WebMark />
      <div className="min-w-0">
        <h1 className="font-display text-xs uppercase leading-none tracking-widest text-teal sm:text-sm">
          Web-Slinger Watch
        </h1>
        <p className="mt-1.5 truncate font-body text-[11px] text-text-muted">
          {tagline}
        </p>
      </div>
    </header>
  );
}
