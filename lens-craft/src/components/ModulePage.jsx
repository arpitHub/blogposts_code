import { Link } from 'react-router-dom';
import LevelBadge from './LevelBadge.jsx';
import { getNextModule } from '../data/modules.js';

// Shared template for every module page:
// why-this-matters intro -> interactive widget -> short explanation -> "try this" challenge -> next link
export default function ModulePage({ module, intro, explanation, challenge, children }) {
  const next = getNextModule(module.slug);

  return (
    <div className="mx-auto max-w-5xl px-6 py-10 md:px-10 md:py-14">
      <div className="mb-8">
        <LevelBadge level={module.level} className="mb-4" />
        <h1 className="font-display text-3xl font-semibold tracking-tight text-ink md:text-4xl">
          {module.title}
        </h1>
        <p className="mt-3 max-w-2xl text-base text-ink-2">{intro}</p>
      </div>

      <div className="mb-10">{children}</div>

      {explanation && (
        <div className="mb-8 max-w-2xl rounded-xl border border-line bg-panel px-5 py-4 text-sm leading-relaxed text-ink-2">
          {explanation}
        </div>
      )}

      {challenge && (
        <div className="mb-10 max-w-2xl rounded-xl border border-accent/30 bg-accent/5 px-5 py-4">
          <div className="mb-1 font-mono text-[11px] uppercase tracking-wider text-accent">
            Try this
          </div>
          <p className="text-sm leading-relaxed text-ink">{challenge}</p>
        </div>
      )}

      <div className="flex items-center justify-between border-t border-line pt-6">
        <Link to="/" className="text-sm text-ink-3 hover:text-ink-2">
          ← All modules
        </Link>
        {next && (
          <Link
            to={`/learn/${next.slug}`}
            className="flex items-center gap-1.5 rounded-full border border-line px-4 py-2 text-sm text-ink hover:border-accent hover:text-accent"
          >
            Next: {next.title} →
          </Link>
        )}
      </div>
    </div>
  );
}
