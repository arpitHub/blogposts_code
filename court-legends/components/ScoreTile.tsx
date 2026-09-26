import type { Tile } from "@/content/types";
import { CountUp } from "./Scoreboard";
import { T } from "./Text";

/**
 * A scoreboard tile: dark board, painted label, big numerals. Used for each chapter's key moment.
 * The numeric part animates if `countTo` is given.
 */
export function ScoreTile({ tile, countTo, className = "" }: { tile: Tile; countTo?: number; className?: string }) {
  return (
    <figure
      className={`relative overflow-hidden rounded-[var(--radius)] border-2 border-board bg-board px-5 py-4 text-board-ink shadow-[inset_0_0_0_3px_rgb(255_255_255/0.06)] ${className}`}
    >
      <span aria-hidden="true" className="absolute inset-x-0 top-0 h-1 bg-board-accent" />
      <figcaption className="board-text text-xs uppercase text-board-ink/80">{tile.label}</figcaption>
      <p className="board-text mt-2 text-4xl leading-none text-board-accent sm:text-5xl">
        {countTo !== undefined ? <CountUp value={tile.value} countTo={countTo} /> : tile.value}
      </p>
      {tile.caption ? <p className="mt-2 text-sm text-board-ink/85">
          <T>{tile.caption}</T>
        </p> : null}
    </figure>
  );
}
