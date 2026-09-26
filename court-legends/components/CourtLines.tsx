import type { ReactNode } from "react";

/**
 * Court markings used as section dividers and frames. All decorative (aria-hidden) and drawn in the
 * theme's --line / --court colours, so every player theme gets its own version for free.
 */

/** The net: posts, mesh and white tape. Used between the page's major sections. */
export function NetDivider({ className = "" }: { className?: string }) {
  return (
    <div aria-hidden="true" className={`mx-auto w-full max-w-6xl px-4 sm:px-8 ${className}`}>
      <div className="flex h-10 items-stretch">
        <span className="w-2 rounded-sm bg-ink" />
        <div className="relative flex-1">
          <span className="absolute inset-x-0 top-1 h-[6px] border-y border-ink/30 bg-line" />
          <span
            className="absolute inset-x-0 bottom-0 top-[10px]"
            style={{
              backgroundImage:
                "repeating-linear-gradient(45deg, color-mix(in srgb, var(--ink) 26%, transparent) 0 1px, transparent 1px 7px), repeating-linear-gradient(-45deg, color-mix(in srgb, var(--ink) 26%, transparent) 0 1px, transparent 1px 7px)",
            }}
          />
        </div>
        <span className="w-2 rounded-sm bg-ink" />
      </div>
    </div>
  );
}

/** A baseline with its centre mark. Used between chapters. */
export function BaselineDivider({ className = "" }: { className?: string }) {
  return (
    <div aria-hidden="true" className={`flex items-center ${className}`}>
      <span className="h-[3px] flex-1 bg-ink/70" />
      <span className="h-3 w-[3px] bg-ink/70" />
      <span className="h-[3px] flex-1 bg-ink/70" />
    </div>
  );
}

/** The service line with the centre service line dropping from it: a "T". */
export function ServiceLine({ className = "" }: { className?: string }) {
  return (
    <div aria-hidden="true" className={`relative h-6 ${className}`}>
      <span className="absolute inset-x-0 top-0 h-[2px] bg-ink/45" />
      <span className="absolute left-1/2 top-0 h-full w-[2px] -translate-x-1/2 bg-ink/45" />
    </div>
  );
}

/**
 * A court seen from above, used to frame the portrait: coloured surface, outer doubles lines,
 * inner singles sidelines and a service line.
 */
export function CourtFrame({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`relative rounded-[var(--radius)] bg-court p-3 shadow-[0_1px_0_var(--line)_inset] sm:p-4 ${className}`}>
      <div className="relative border-[3px] border-line">
        {/* singles sidelines inside the doubles alleys */}
        <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 left-[7%] z-10 w-[2px] bg-line/80" />
        <span aria-hidden="true" className="pointer-events-none absolute inset-y-0 right-[7%] z-10 w-[2px] bg-line/80" />
        {children}
      </div>
    </div>
  );
}

/** A section heading styled like a painted court label. */
export function SectionHeading({ id, kicker, title, intro }: { id: string; kicker: string; title: string; intro?: string }) {
  return (
    <header className="mb-10 max-w-3xl sm:mb-14">
      <p className="board-text text-sm uppercase text-accent-ink">{kicker}</p>
      <h2 id={id} className="display mt-2 text-4xl leading-[1.05] sm:text-5xl">
        {title}
      </h2>
      {intro ? <p className="mt-4 text-lg text-muted">{intro}</p> : null}
    </header>
  );
}
