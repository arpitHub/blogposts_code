import { Link } from 'react-router-dom';
import { MODULES, modulesByLevel } from '../data/modules.js';
import LevelBadge from '../components/LevelBadge.jsx';

export default function Landing() {
  const firstModule = MODULES.find((m) => m.order === 1);
  const groups = modulesByLevel();

  return (
    <div className="px-6 py-12 md:px-10 md:py-16">
      <section className="mx-auto mb-16 max-w-3xl text-center">
        <p className="mb-3 font-mono text-xs uppercase tracking-[0.2em] text-accent">
          Learn photography by playing with it
        </p>
        <h1 className="font-display text-4xl font-semibold tracking-tight text-ink md:text-5xl">
          Lens Craft
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-base leading-relaxed text-ink-2">
          Every concept here has a dial you can drag. No walls of text to memorize —
          move a slider, watch the photo change, and the concept explains itself.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <Link
            to={`/learn/${firstModule.slug}`}
            className="rounded-full bg-accent px-6 py-3 text-sm font-medium text-[#1a1200] transition-transform hover:scale-[1.03]"
          >
            Start here — Exposure Triangle
          </Link>
          <a
            href="#browse"
            className="rounded-full border border-line px-6 py-3 text-sm text-ink-2 hover:border-accent hover:text-accent"
          >
            Browse all modules
          </a>
        </div>
      </section>

      <section id="browse" className="mx-auto max-w-6xl scroll-mt-10">
        <h2 className="mb-6 font-display text-xl font-semibold text-ink">
          Browse by skill level
        </h2>
        <div className="flex flex-col gap-10">
          {groups.map((group) => (
            <div key={group.level}>
              <div className="mb-3">
                <LevelBadge level={group.level} />
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {group.modules.map((m) => (
                  <Link
                    key={m.slug}
                    to={`/learn/${m.slug}`}
                    className="group flex flex-col justify-between rounded-2xl border border-line bg-panel p-5 transition-colors hover:border-ink-3"
                  >
                    <div>
                      <div className="mb-2 flex items-center justify-between">
                        <span className="font-mono text-[11px] text-ink-3">
                          {String(m.order).padStart(2, '0')}
                        </span>
                        {!m.ready && (
                          <span className="rounded border border-line px-1.5 py-0.5 text-[10px] font-mono text-ink-3">
                            coming soon
                          </span>
                        )}
                      </div>
                      <h3 className="font-display text-base font-medium text-ink group-hover:text-accent">
                        {m.title}
                      </h3>
                      <p className="mt-1.5 text-sm leading-snug text-ink-2">{m.tagline}</p>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
