"use client";

interface SearchBarProps {
  query: string;
  onQueryChange: (query: string) => void;
  /** Number of matching tools, or null when no search is active. */
  resultCount: number | null;
}

export default function SearchBar({
  query,
  onQueryChange,
  resultCount,
}: SearchBarProps) {
  return (
    <div className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur">
      <div className="mx-auto flex w-full max-w-7xl items-center gap-3 px-4 py-3 sm:px-6">
        <div className="relative flex-1">
          <svg
            aria-hidden="true"
            viewBox="0 0 20 20"
            fill="currentColor"
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500"
          >
            <path
              fillRule="evenodd"
              d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
              clipRule="evenodd"
            />
          </svg>
          <input
            type="search"
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search tools by name or description…"
            aria-label="Search tools"
            className="w-full rounded-lg border border-zinc-800 bg-zinc-900 py-2 pl-9 pr-3 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-600"
          />
        </div>
        {resultCount !== null && (
          <span className="shrink-0 text-xs text-zinc-500">
            {resultCount} {resultCount === 1 ? "tool" : "tools"}
          </span>
        )}
      </div>
    </div>
  );
}
