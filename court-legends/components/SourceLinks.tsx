import type { Source } from "@/content/types";

/** A compact "Sources:" line. Each link opens the source in a new tab. */
export function SourceLinks({ sources, label = "Sources", className = "" }: { sources: Source[]; label?: string; className?: string }) {
  if (!sources.length) return null;
  return (
    <p className={`text-xs leading-relaxed text-muted ${className}`}>
      <span className="board-text uppercase">{label}:</span>{" "}
      {sources.map((s, i) => (
        <span key={s.url}>
          {i > 0 ? <span aria-hidden="true"> · </span> : null}
          <a href={s.url} className="link" target="_blank" rel="noopener noreferrer">
            {s.label}
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        </span>
      ))}
    </p>
  );
}
