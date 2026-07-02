import { useEffect, useMemo, useState } from 'react';
import { themes } from '../data/themes.js';

function EpicColumn({ epic, label, content, characters, moment, className }) {
  return (
    <article className={`epic-column ${className}`}>
      <header className="epic-header">
        <span className="epic-eyebrow">{label}</span>
        <h3 className="epic-name">{epic}</h3>
      </header>

      <div className="epic-body">
        {content.map((paragraph, i) => (
          <p key={i} className="epic-paragraph">
            {paragraph}
          </p>
        ))}
      </div>

      <div className="epic-meta">
        <div className="epic-meta-block">
          <span className="epic-meta-label">Key figures</span>
          <ul className="epic-characters">
            {characters.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
        <div className="epic-meta-block">
          <span className="epic-meta-label">A moment</span>
          <p className="epic-moment">{moment}</p>
        </div>
      </div>
    </article>
  );
}

export default function ThemeDetail({ themeId, onSelectTheme, onBack }) {
  const themeIndex = useMemo(
    () => themes.findIndex((t) => t.id === themeId),
    [themeId],
  );
  const theme = themes[themeIndex];

  const [phase, setPhase] = useState(0);

  useEffect(() => {
    setPhase(0);
    const t1 = window.setTimeout(() => setPhase(1), 60);
    const t2 = window.setTimeout(() => setPhase(2), 480);
    const t3 = window.setTimeout(() => setPhase(3), 960);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.clearTimeout(t3);
    };
  }, [themeId]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
  }, [themeId]);

  if (!theme) {
    return (
      <section className="detail detail--missing">
        <p>That thread cannot be found.</p>
        <button type="button" className="text-button" onClick={onBack}>
          Return to all threads
        </button>
      </section>
    );
  }

  const prevIndex = (themeIndex - 1 + themes.length) % themes.length;
  const nextIndex = (themeIndex + 1) % themes.length;
  const prevTheme = themes[prevIndex];
  const nextTheme = themes[nextIndex];
  const hasSiblings = themes.length > 1;

  return (
    <section className={`detail phase-${phase}`}>
      <header className="detail-topbar">
        <button
          type="button"
          className="icon-button"
          onClick={onBack}
          aria-label="Back to all themes"
        >
          <span aria-hidden="true">←</span>
        </button>
        <span className="detail-crumb">
          <span className="eyebrow">Thread</span>
          <span className="detail-crumb-count">
            {String(themeIndex + 1).padStart(2, '0')}
            <span className="detail-crumb-sep"> / </span>
            {String(themes.length).padStart(2, '0')}
          </span>
        </span>
        <span className="detail-symbol" aria-hidden="true">
          {theme.symbol}
        </span>
      </header>

      <section className="detail-hero">
        <h1 className="detail-name">{theme.name}</h1>
        <p className="detail-provocation">{theme.provocation}</p>
      </section>

      <div className="detail-rule" aria-hidden="true" />

      <div className="detail-columns">
        <EpicColumn
          epic="Ramayana"
          label="The Solar Dynasty"
          className="epic-column--ramayana"
          content={theme.ramayana.content}
          characters={theme.ramayana.characters}
          moment={theme.ramayana.moment}
        />

        <div className="detail-divider" aria-hidden="true">
          <span className="detail-divider-lotus">✦</span>
        </div>

        <EpicColumn
          epic="Mahabharata"
          label="The Lunar Dynasty"
          className="epic-column--mahabharata"
          content={theme.mahabharata.content}
          characters={theme.mahabharata.characters}
          moment={theme.mahabharata.moment}
        />
      </div>

      <section className="synthesis">
        <div className="synthesis-rule" aria-hidden="true" />
        <span className="eyebrow synthesis-eyebrow">The Common Thread</span>
        <p className="synthesis-body">{theme.synthesis}</p>

        <blockquote className="shloka">
          <p className="shloka-sanskrit">{theme.shloka.sanskrit}</p>
          <p className="shloka-translation">“{theme.shloka.translation}”</p>
          <footer className="shloka-source">— {theme.shloka.source}</footer>
        </blockquote>
      </section>

      <nav className="detail-nav" aria-label="Theme navigation">
        <button
          type="button"
          className="nav-button nav-button--prev"
          onClick={() => hasSiblings && onSelectTheme(prevTheme.id)}
          disabled={!hasSiblings}
        >
          <span className="nav-arrow" aria-hidden="true">←</span>
          <span className="nav-labels">
            <span className="eyebrow">Previous</span>
            <span className="nav-title">{prevTheme.name}</span>
          </span>
        </button>

        <button
          type="button"
          className="text-button detail-back"
          onClick={onBack}
        >
          All threads
        </button>

        <button
          type="button"
          className="nav-button nav-button--next"
          onClick={() => hasSiblings && onSelectTheme(nextTheme.id)}
          disabled={!hasSiblings}
        >
          <span className="nav-labels nav-labels--right">
            <span className="eyebrow">Next</span>
            <span className="nav-title">{nextTheme.name}</span>
          </span>
          <span className="nav-arrow" aria-hidden="true">→</span>
        </button>
      </nav>
    </section>
  );
}
