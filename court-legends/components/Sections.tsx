import type { ImageCredit, NoteItem, Player, RecordStatus } from "@/content/types";
import { allSources } from "@/lib/format";
import { SectionHeading } from "./CourtLines";
import { Reveal } from "./Motion";
import { PortraitCredit } from "./Portrait";
import { CountUp } from "./Scoreboard";
import { SourceLinks } from "./SourceLinks";

const STATUS: Record<RecordStatus, { label: string; tone: string } | null> = {
  "still-stands": { label: "Still stands", tone: "bg-board-accent text-board" },
  "since-broken": { label: "Since broken", tone: "border border-board-ink/60 text-board-ink" },
  first: { label: "A first: can't be broken", tone: "bg-board-ink text-board" },
  "sources-differ": { label: "Sources differ", tone: "border border-dashed border-board-accent text-board-accent" },
  stat: null,
};

export function Records({ player }: { player: Player }) {
  return (
    <section aria-labelledby="records" className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-8 sm:py-24">
      <SectionHeading
        id="records"
        kicker="Records & stats"
        title="On the scoreboard"
        intro="Each record is labelled honestly: still standing, since broken, a first, or disputed by the sources."
      />
      <ul className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {player.records.map((r, i) => {
          const status = STATUS[r.status];
          return (
            <li key={r.label}>
              <Reveal delay={(i % 3) * 0.06} className="h-full">
                <div className="flex h-full flex-col rounded-[var(--radius)] border-2 border-board bg-board p-5 text-board-ink">
                  <div className="flex min-h-7 items-start justify-between gap-3">
                    <p className="board-text text-xs uppercase text-board-ink/80">{r.label}</p>
                  </div>
                  <p className="board-text mt-3 text-5xl leading-none text-board-accent">
                    <CountUp value={r.value} countTo={r.countTo} />
                  </p>
                  {status ? (
                    <p className={`board-text mt-4 self-start rounded-full px-3 py-0.5 text-[0.7rem] uppercase ${status.tone}`}>
                      {status.label}
                    </p>
                  ) : (
                    <p className="board-text mt-4 text-[0.7rem] uppercase text-board-ink/70">Career stat</p>
                  )}
                  <p className="mt-3 flex-1 text-sm leading-relaxed text-board-ink/90">{r.detail}</p>
                  <p className="mt-4 text-xs text-board-ink/80">
                    <span className="board-text uppercase">Sources:</span>{" "}
                    {r.sources.map((s, j) => (
                      <span key={s.url}>
                        {j > 0 ? " · " : null}
                        <a href={s.url} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-board-accent">
                          {s.label}
                        </a>
                      </span>
                    ))}
                  </p>
                </div>
              </Reveal>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function NoteCard({ note, index }: { note: NoteItem; index: number }) {
  return (
    <li>
      <Reveal delay={(index % 2) * 0.06} className="h-full">
        <div className="h-full rounded-[var(--radius)] border-2 border-ink/15 bg-paper/70 p-5">
          <h4 className="display text-xl leading-snug">{note.title}</h4>
          <p className="mt-2 leading-relaxed">{note.body}</p>
          <SourceLinks sources={note.sources} className="mt-3" />
        </div>
      </Reveal>
    </li>
  );
}

export function EraContext({ player }: { player: Player }) {
  return (
    <section aria-labelledby="era" className="bg-paper-2/70">
      <div className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-8 sm:py-24">
        <SectionHeading
          id="era"
          kicker="Era context"
          title={`Tennis in ${player.shortName}'s time`}
          intro="Amateurs and professionals, prize money, rankings, and the world outside the court."
        />
        <div className="grid gap-12 lg:grid-cols-[3fr_2fr]">
          <div>
            <h3 className="board-text mb-4 text-sm uppercase text-accent-ink">On court</h3>
            <ul className="grid gap-4 sm:grid-cols-2">
              {player.era.onCourt.map((n, i) => (
                <NoteCard key={n.title} note={n} index={i} />
              ))}
            </ul>
          </div>
          <div>
            <h3 className="board-text mb-4 text-sm uppercase text-accent-ink">In the world</h3>
            <ol className="relative space-y-6 border-l-[3px] border-ink/25 pl-6">
              {player.era.inTheWorld.map((n) => (
                <li key={n.title} className="relative">
                  <span aria-hidden="true" className="absolute -left-[33px] top-1.5 h-4 w-4 rounded-full border-[3px] border-paper bg-accent" />
                  <h4 className="board-text text-lg">{n.title}</h4>
                  <p className="mt-1 leading-relaxed">{n.body}</p>
                  <SourceLinks sources={n.sources} className="mt-2" />
                </li>
              ))}
            </ol>
          </div>
        </div>
      </div>
    </section>
  );
}

export function BeyondTheCourt({ player }: { player: Player }) {
  return (
    <section aria-labelledby="beyond" className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-8 sm:py-24">
      <SectionHeading id="beyond" kicker="Beyond the court" title="More than titles" />
      {/* Service boxes: a court's grid of rectangles, with the net across the top */}
      <ul className="grid overflow-hidden rounded-[var(--radius)] border-[3px] border-line bg-court sm:grid-cols-2">
        {player.beyond.map((b, i) => (
          <li key={b.title} className="border-line/70 p-6 text-line sm:p-8 [&:not(:last-child)]:border-b-2 sm:[&:nth-child(odd):not(:last-child)]:border-r-2 sm:[&:last-child:nth-child(odd)]:col-span-2">
            <p className="board-text text-xs uppercase text-line">{String(i + 1).padStart(2, "0")}</p>
            <h3 className="display mt-2 text-2xl leading-snug">{b.title}</h3>
            <p className="mt-3 leading-relaxed">{b.body}</p>
            <p className="mt-4 text-xs text-line">
              <span className="board-text uppercase">Sources:</span>{" "}
              {b.sources.map((s, j) => (
                <span key={s.url}>
                  {j > 0 ? " · " : null}
                  <a href={s.url} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2">
                    {s.label}
                  </a>
                </span>
              ))}
            </p>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function LegacyAndSources({ player, credit }: { player: Player; credit?: ImageCredit }) {
  const sources = allSources(player);
  return (
    <section aria-labelledby="legacy" className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-8 sm:py-24">
      <SectionHeading id="legacy" kicker="Legacy" title="Match point" />
      <div className="max-w-3xl space-y-5 text-xl leading-relaxed sm:text-2xl">
        {player.legacy.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>

      <div className="mt-16 grid gap-12 lg:grid-cols-2">
        <div>
          <h3 id="sources-differ" className="display text-2xl">
            Where sources differ
          </h3>
          <ul className="mt-5 space-y-5">
            {player.sourcesDiffer.map((n) => (
              <li key={n.title} className="border-l-[3px] border-dashed border-accent pl-4">
                <h4 className="font-semibold">{n.title}</h4>
                <p className="mt-1 leading-relaxed">{n.body}</p>
                <SourceLinks sources={n.sources} className="mt-2" />
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 id="sources" className="display text-2xl">
            Sources
          </h3>
          <p className="mt-2 text-sm text-muted">Every source cited on this page ({sources.length}).</p>
          <ol className="mt-4 list-decimal space-y-1.5 pl-6 text-sm marker:text-muted">
            {sources.map((s) => (
              <li key={s.url}>
                <a href={s.url} className="link break-words" target="_blank" rel="noopener noreferrer">
                  {s.label}
                  <span className="sr-only"> (opens in a new tab)</span>
                </a>
              </li>
            ))}
          </ol>
          <h3 className="display mt-10 text-2xl">Image credit</h3>
          <div className="mt-2">
            <PortraitCredit player={player} credit={credit} />
          </div>
        </div>
      </div>
    </section>
  );
}
