import Link from "next/link";
import { CourtFrame, NetDivider } from "@/components/CourtLines";
import { Reveal } from "@/components/Motion";
import { Portrait } from "@/components/Portrait";
import { SiteFooter, SiteHeader, SkipLink } from "@/components/SiteChrome";
import { getPortrait, players } from "@/content/players";

export default function Home() {
  return (
    <div data-theme="home" className="theme-root min-h-screen">
      <SkipLink />
      <SiteHeader />
      <main id="main" tabIndex={-1} className="focus:outline-none">
        <section aria-labelledby="home-title" className="mx-auto w-full max-w-6xl px-4 pb-12 pt-8 sm:px-8 sm:pb-16 sm:pt-14">
          <p className="board-text text-sm uppercase text-accent-ink">Profiles of the game's greats</p>
          <h1 id="home-title" className="display mt-3 text-[clamp(3.4rem,13vw,8.5rem)] leading-[0.9]">
            Court Legends
          </h1>
          <p className="mt-6 max-w-2xl text-xl leading-snug sm:text-2xl">
            One full-page profile per tennis great, each told like a match, set by set, in the look of the player's era. Every
            fact links to its source, and where the sources disagree, the page says so.
          </p>
        </section>

        <NetDivider />

        <section aria-labelledby="players-title" className="mx-auto w-full max-w-6xl px-4 py-14 sm:px-8 sm:py-20">
          <h2 id="players-title" className="board-text text-sm uppercase text-muted">
            The draw · {players.length} players
          </h2>
          <ul className="mt-6 grid gap-8 md:grid-cols-2">
            {players.map((p, i) => (
              <li key={p.slug}>
                <Reveal delay={i * 0.08} className="h-full">
                  <article
                    data-theme={p.theme}
                    className="theme-root group relative flex h-full flex-col overflow-hidden rounded-[var(--radius)] border-2 border-ink/20 transition-shadow focus-within:shadow-[0_0_0_3px_var(--accent-2)] hover:shadow-[0_10px_30px_-12px_rgb(0_0_0/0.45)]"
                  >
                    <div className="grid grid-cols-[1fr_1.2fr] items-stretch">
                      <div className="p-3">
                        <CourtFrame className="!p-2">
                          <Portrait player={p} credit={getPortrait(p.slug)} sizes="(min-width: 768px) 220px, 40vw" />
                        </CourtFrame>
                      </div>
                      <div className="flex flex-col justify-center p-5 pl-2">
                        <p className="board-text text-xs uppercase text-accent-ink">
                          {p.eraLabel} · {p.country}
                        </p>
                        <h3 className="display mt-2 text-3xl leading-none sm:text-4xl">
                          <Link href={`/players/${p.slug}`} className="after:absolute after:inset-0 focus:outline-none">
                            {p.name}
                          </Link>
                        </h3>
                        <p className="board-text mt-2 text-sm uppercase text-muted">{p.lifespan}</p>
                      </div>
                    </div>
                    <div className="flex flex-1 flex-col border-t-2 border-ink/15 p-5">
                      <p className="flex-1 leading-relaxed">{p.identity}</p>
                      <p className="board-text mt-4 text-sm uppercase text-accent-ink">
                        Read the profile <span aria-hidden="true" className="inline-block transition-transform group-hover:translate-x-1">→</span>
                      </p>
                    </div>
                  </article>
                </Reveal>
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby="method-title" className="mx-auto w-full max-w-6xl px-4 pb-20 sm:px-8">
          <div className="rounded-[var(--radius)] border-2 border-board bg-board p-6 text-board-ink sm:p-10">
            <h2 id="method-title" className="board-text text-sm uppercase text-board-accent">
              How these profiles are made
            </h2>
            <ul className="mt-5 grid gap-6 text-base leading-relaxed sm:grid-cols-3">
              <li>
                <p className="board-text text-2xl text-board-accent">01</p>
                <p className="mt-1">Each player starts as a research sheet: dates, opponents and scores, every line linked to a source.</p>
              </li>
              <li>
                <p className="board-text text-2xl text-board-accent">02</p>
                <p className="mt-1">Records are labelled honestly: still standing, since broken, a first, or disputed.</p>
              </li>
              <li>
                <p className="board-text text-2xl text-board-accent">03</p>
                <p className="mt-1">Photos come only from Wikimedia Commons under free licences, credited on the page.</p>
              </li>
            </ul>
          </div>
        </section>
      </main>
      <SiteFooter />
    </div>
  );
}
