import { useEffect, useState } from 'react'

// Real components arrive in later steps. For now we render inline placeholders
// so the app shell, navigation, and view-switching logic can be exercised.

const VIEWS = {
  GRID: 'grid',
  DETAIL: 'detail',
  SAGE: 'sage'
}

function GridPlaceholder({ factionFilter, onFactionChange, onSelectCharacter }) {
  const factions = ['all', 'pandava', 'ally', 'kaurava']
  return (
    <section className="placeholder">
      <div className="placeholder__filter-bar">
        {factions.map((f) => (
          <button
            key={f}
            className={`pill ${factionFilter === f ? 'pill--active' : ''}`}
            onClick={() => onFactionChange(f)}
          >
            {f === 'all' ? 'All' : f[0].toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>
      <p className="placeholder__note">
        Character grid will render here. Click below to preview the detail view.
      </p>
      <button className="link-button" onClick={() => onSelectCharacter('arjuna')}>
        Open sample character →
      </button>
    </section>
  )
}

function DetailPlaceholder({ characterId, onBack }) {
  return (
    <section className="placeholder">
      <button className="link-button" onClick={onBack}>
        ← Back to atlas
      </button>
      <p className="placeholder__note">
        Detail view for <em>{characterId ?? 'unknown'}</em> will render here.
      </p>
    </section>
  )
}

function SagePlaceholder() {
  return (
    <section className="placeholder">
      <p className="placeholder__note">
        Ask a Sage chat interface will render here.
      </p>
    </section>
  )
}

export default function App() {
  const [view, setView] = useState(VIEWS.GRID)
  const [selectedCharacterId, setSelectedCharacterId] = useState(null)
  const [factionFilter, setFactionFilter] = useState('all')

  const goHome = () => {
    setSelectedCharacterId(null)
    setView(VIEWS.GRID)
  }

  const openCharacter = (id) => {
    setSelectedCharacterId(id)
    setView(VIEWS.DETAIL)
  }

  const openSage = () => setView(VIEWS.SAGE)

  // Scroll to top whenever the view changes — feels less jarring than retaining
  // the previous scroll position when switching between atlas and detail.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }, [view, selectedCharacterId])

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__header-inner">
          <button
            type="button"
            className="brand"
            onClick={goHome}
            aria-label="Return to atlas home"
          >
            <span className="brand__eyebrow">Mahābhārata</span>
            <span className="brand__title">Character Atlas</span>
          </button>

          <nav className="app__nav" aria-label="Primary">
            <button
              type="button"
              className={`nav-link ${view !== VIEWS.SAGE ? 'nav-link--active' : ''}`}
              onClick={goHome}
            >
              Atlas
            </button>
            <button
              type="button"
              className={`nav-link ${view === VIEWS.SAGE ? 'nav-link--active' : ''}`}
              onClick={openSage}
            >
              Ask a Sage
            </button>
          </nav>
        </div>
      </header>

      <main className="app__main">
        <div className="app__view fade-in" key={`${view}-${selectedCharacterId ?? 'none'}`}>
          {view === VIEWS.GRID && (
            <GridPlaceholder
              factionFilter={factionFilter}
              onFactionChange={setFactionFilter}
              onSelectCharacter={openCharacter}
            />
          )}
          {view === VIEWS.DETAIL && (
            <DetailPlaceholder
              characterId={selectedCharacterId}
              onBack={goHome}
            />
          )}
          {view === VIEWS.SAGE && <SagePlaceholder />}
        </div>
      </main>

      <footer className="app__footer">
        <span>An editorial atlas of dharma, doubt, and devotion.</span>
      </footer>
    </div>
  )
}
