import { themes } from '../data/themes.js';

function shortDescription(theme) {
  return theme.provocation;
}

export default function ThemeNavigator({ onOpenTheme, onBack }) {
  return (
    <section className="navigator">
      <header className="navigator-header">
        <button
          type="button"
          className="icon-button"
          onClick={onBack}
          aria-label="Back to landing"
        >
          <span aria-hidden="true">←</span>
        </button>
        <div className="navigator-title">
          <span className="eyebrow">Ekasutra</span>
          <span className="navigator-title-main">The Threads</span>
        </div>
        <span className="navigator-count">
          {themes.length} {themes.length === 1 ? 'theme' : 'themes'}
        </span>
      </header>

      <ol className="navigator-list">
        {themes.map((theme, index) => (
          <li key={theme.id} className="navigator-item">
            <button
              type="button"
              className="navigator-card"
              onClick={() => onOpenTheme(theme.id)}
            >
              <span className="navigator-index">
                {String(index + 1).padStart(2, '0')}
              </span>
              <div className="navigator-body">
                <span className="navigator-symbol" aria-hidden="true">
                  {theme.symbol}
                </span>
                <h2 className="navigator-name">{theme.name}</h2>
                <p className="navigator-desc">{shortDescription(theme)}</p>
              </div>
              <span className="navigator-arrow" aria-hidden="true">
                →
              </span>
            </button>
          </li>
        ))}
      </ol>

      <footer className="navigator-footer">
        <span className="navigator-footer-line" aria-hidden="true" />
        <span className="navigator-footer-text">
          Two epics, woven from one thread.
        </span>
        <span className="navigator-footer-line" aria-hidden="true" />
      </footer>
    </section>
  );
}
