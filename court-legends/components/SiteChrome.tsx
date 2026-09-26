import Link from "next/link";
import { players } from "@/content/players";

export function SkipLink() {
  return (
    <a
      href="#main"
      className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-board focus:px-4 focus:py-2 focus:text-board-ink"
    >
      Skip to content
    </a>
  );
}

export function SiteHeader({ current }: { current?: string }) {
  return (
    <header className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-5 sm:px-8">
      <Link href="/" className="board-text text-base uppercase tracking-[0.12em] text-ink hover:text-accent-ink">
        Court Legends
      </Link>
      <nav aria-label="Players">
        <ul className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
          {players.map((p) => (
            <li key={p.slug}>
              <Link
                href={`/players/${p.slug}`}
                aria-current={current === p.slug ? "page" : undefined}
                className="board-text uppercase text-muted hover:text-ink aria-[current=page]:text-accent-ink aria-[current=page]:underline aria-[current=page]:underline-offset-4"
              >
                {p.name}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}

export function SiteFooter({ children }: { children?: React.ReactNode }) {
  return (
    <footer className="mx-auto w-full max-w-6xl px-4 pb-10 text-sm text-muted sm:px-8">
      <div className="space-y-3 border-t-[3px] border-ink/70 pt-10">
      {children}
      <p>
        Court Legends profiles are built only from sourced facts. Where sources disagree, the page says so. Each profile's
        research sheet lives in the project repository under <code>content/research/</code>.
      </p>
      <p>
        <Link href="/" className="link">
          All players
        </Link>
      </p>
      </div>
    </footer>
  );
}
