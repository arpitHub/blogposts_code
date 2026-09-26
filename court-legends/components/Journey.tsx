import type { Chapter, Match, Player } from "@/content/types";
import { yearRange } from "@/lib/format";
import { BaselineDivider, SectionHeading, ServiceLine } from "./CourtLines";
import { JourneyShell, type RailItem } from "./JourneyShell";
import { Reveal } from "./Motion";
import { ScoreTile } from "./ScoreTile";
import { SourceLinks } from "./SourceLinks";
import { T } from "./Text";

const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"];

export function Journey({ player }: { player: Player }) {
  const items: RailItem[] = player.journey.map((c) => ({
    set: c.set,
    title: c.title,
    years: yearRange(c.years),
    startYear: c.years[0],
  }));
  const count = player.journey.length;

  return (
    <section aria-labelledby="journey" className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-8 sm:py-24">
      <SectionHeading
        id="journey"
        kicker="The Journey"
        title={`A career in ${WORDS[count] ?? count} sets`}
        intro={`From ${player.born.place} to today, told like a match. Each set closes with an era note, and every claim links to its source.`}
      />
      <JourneyShell items={items}>
        <ol className="space-y-20 sm:space-y-28">
          {player.journey.map((chapter, i) => (
            <li key={chapter.set}>
              <ChapterView chapter={chapter} shortName={player.shortName} />
              {i < count - 1 ? <BaselineDivider className="mt-20 sm:mt-28" /> : null}
            </li>
          ))}
        </ol>
      </JourneyShell>
    </section>
  );
}

function ChapterView({ chapter, shortName }: { chapter: Chapter; shortName: string }) {
  const headingId = `set-${chapter.set}-title`;
  return (
    <article id={`set-${chapter.set}`} data-set={chapter.set} aria-labelledby={headingId} className="scroll-mt-16 lg:scroll-mt-8">
      <header>
        <p className="board-text inline-flex items-stretch overflow-hidden rounded-[var(--radius)] border-2 border-board text-sm uppercase">
          <span className="bg-board px-3 py-1 text-board-accent">Set {chapter.set}</span>
          <span className="px-3 py-1 text-ink">{yearRange(chapter.years)}</span>
        </p>
        <p className="board-text mt-5 text-sm uppercase text-accent-ink">{chapter.title}</p>
        <h3 id={headingId} className="display mt-1 text-3xl leading-tight sm:text-[2.6rem]">
          {chapter.headline}
        </h3>
      </header>

      <div className="mt-8 grid gap-10 xl:grid-cols-[minmax(0,1fr)_260px]">
        <div className="prose-body max-w-prose text-[1.075rem] leading-relaxed sm:text-lg">
          {chapter.paragraphs.map((p, i) => (
            <p key={i}>
              <T>{p}</T>
            </p>
          ))}
        </div>
        <Reveal className="xl:pt-1">
          <ScoreTile tile={chapter.tile} />
        </Reveal>
      </div>

      {chapter.matches?.length ? <MatchList matches={chapter.matches} shortName={shortName} /> : null}

      <Reveal>
        <aside aria-label={`Era note, set ${chapter.set}`} className="mt-10 max-w-prose border-l-[3px] border-accent pl-5">
          <p className="board-text text-xs uppercase text-accent-ink">Era note</p>
          <p className="mt-1 text-base leading-relaxed">
            <T>{chapter.eraNote.text}</T>
          </p>
          <SourceLinks sources={chapter.eraNote.sources} className="mt-2" />
        </aside>
      </Reveal>

      <details className="group mt-8 max-w-prose">
        <summary className="board-text cursor-pointer list-none text-xs uppercase text-muted hover:text-ink [&::-webkit-details-marker]:hidden">
          <span aria-hidden="true" className="mr-1 inline-block transition-transform group-open:rotate-90">
            ▸
          </span>
          Sources for set {chapter.set} ({chapter.sources.length})
        </summary>
        <ul className="mt-3 space-y-1 text-sm">
          {chapter.sources.map((s) => (
            <li key={s.url}>
              <a href={s.url} className="link" target="_blank" rel="noopener noreferrer">
                {s.label}
                <span className="sr-only"> (opens in a new tab)</span>
              </a>
            </li>
          ))}
        </ul>
      </details>
    </article>
  );
}

function MatchList({ matches, shortName }: { matches: Match[]; shortName: string }) {
  return (
    <div className="mt-12">
      <ServiceLine className="mb-4 max-w-xs" />
      <h4 className="board-text text-sm uppercase text-muted">Key matches · scores from {shortName}&rsquo;s side</h4>
      <ul className="mt-4 grid gap-4 md:grid-cols-2">
        {matches.map((m, i) => (
          <li key={i}>
            <Reveal delay={Math.min(i, 3) * 0.05} className="h-full">
              <div className="flex h-full flex-col rounded-[var(--radius)] border-2 border-ink/15 bg-paper-2/60 p-4">
                <div className="flex items-start gap-3">
                  <span
                    className={`board-text grid h-9 w-9 shrink-0 place-items-center rounded-[calc(var(--radius)/2)] text-base ${
                      m.result === "W" ? "bg-accent-ink text-paper" : "border-2 border-ink/40 text-ink"
                    }`}
                  >
                    <span aria-hidden="true">{m.result}</span>
                    <span className="sr-only">{m.result === "W" ? "Won" : "Lost"}</span>
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm text-muted">
                      {m.year} · {m.event} · {m.round}
                    </p>
                    <p className="font-semibold leading-snug">vs {m.opponent}</p>
                  </div>
                </div>
                <p className="board-text mt-3 text-lg">
                  {m.score ?? <span className="text-sm normal-case text-muted">Score not given in our sources</span>}
                </p>
                <p className="mt-2 flex-1 text-sm leading-relaxed">
                  <T>{m.why}</T>
                </p>
                <SourceLinks sources={m.sources} className="mt-3" />
              </div>
            </Reveal>
          </li>
        ))}
      </ul>
    </div>
  );
}
