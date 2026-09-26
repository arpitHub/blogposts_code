import Link from "next/link";
import type { Player } from "@/content/types";

const SECTIONS = [
  { id: "journey", label: "Journey" },
  { id: "records", label: "Records" },
  { id: "era", label: "Era" },
  { id: "beyond", label: "Beyond the court" },
  { id: "legacy", label: "Legacy" },
  { id: "sources", label: "Sources" },
];

/** "On this page" links under the hero: these pages are long. */
export function PageNav() {
  return (
    <nav aria-label="On this page" className="mx-auto w-full max-w-6xl px-4 pb-8 sm:px-8">
      <ul className="board-text flex flex-wrap gap-2 text-sm uppercase">
        {SECTIONS.map((s) => (
          <li key={s.id}>
            <a href={`#${s.id}`} className="inline-block rounded-full border-2 border-ink/20 px-3 py-1 text-ink hover:border-ink hover:bg-paper-2">
              {s.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

/** Link to the next player in the draw, so a reader can keep going. */
export function NextPlayer({ next }: { next: Player }) {
  return (
    <nav aria-label="Next profile" className="mx-auto w-full max-w-6xl px-4 pb-16 sm:px-8">
      <Link
        href={`/players/${next.slug}`}
        className="group flex flex-wrap items-end justify-between gap-x-6 gap-y-2 rounded-[var(--radius)] border-2 border-board bg-board px-6 py-5 text-board-ink"
      >
        <span>
          <span className="board-text block text-xs uppercase text-board-ink/80">Next legend</span>
          <span className="board-text mt-1 block text-3xl uppercase text-board-accent">{next.name}</span>
        </span>
        <span className="board-text text-sm uppercase">
          {next.eraLabel} · {next.lifespan}{" "}
          <span aria-hidden="true" className="inline-block transition-transform group-hover:translate-x-1">
            →
          </span>
        </span>
      </Link>
    </nav>
  );
}
