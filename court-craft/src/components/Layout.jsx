import { useState } from 'react'
import { NavLink, Link, Outlet, useLocation } from 'react-router-dom'
import { CATEGORIES, byCategory, LEVELS } from '../data/modules'

function NavItem({ mod, onNavigate }) {
  return (
    <NavLink
      to={mod.path}
      onClick={onNavigate}
      className={({ isActive }) =>
        `flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition-colors ${
          isActive
            ? 'bg-clay-500 text-white font-medium'
            : 'text-court-100/80 hover:bg-court-800 hover:text-white'
        }`
      }
    >
      <span className="truncate">{mod.title}</span>
      {mod.flagship && <span className="text-[10px] uppercase tracking-wider text-clay-300">★</span>}
    </NavLink>
  )
}

function Sidebar({ onNavigate }) {
  return (
    <div className="flex h-full flex-col overflow-y-auto bg-court-900 px-4 py-6">
      <Link to="/" onClick={onNavigate} className="mb-6 flex items-baseline gap-2 px-2">
        <span className="font-display text-2xl font-bold tracking-tight text-white">
          Court<span className="text-clay-400">Craft</span>
        </span>
      </Link>

      <nav className="flex flex-col gap-5">
        {CATEGORIES.map((cat) => (
          <div key={cat.id}>
            <div className="mb-1.5 px-3 text-[11px] font-semibold uppercase tracking-widest text-court-300">
              {cat.label}
            </div>
            <div className="flex flex-col gap-0.5">
              {byCategory(cat.id).map((m) => (
                <NavItem key={m.id} mod={m} onNavigate={onNavigate} />
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="mt-8 border-t border-court-800 pt-4 px-3">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-court-300">Levels</div>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(LEVELS).filter(([k]) => k !== 'all').map(([k, v]) => (
            <span key={k} className={`rounded-full px-2 py-0.5 text-[11px] ${v.color}`}>{v.label}</span>
          ))}
        </div>
        <p className="mt-3 text-xs leading-relaxed text-court-300/80">
          Every module marks its level. Start anywhere — beginner pages never assume prior knowledge.
        </p>
      </div>
    </div>
  )
}

export default function Layout() {
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()
  const close = () => setMenuOpen(false)

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
      {/* Desktop sidebar */}
      <aside className="hidden lg:block sticky top-0 h-screen">
        <Sidebar />
      </aside>

      {/* Mobile header */}
      <header className="sticky top-0 z-40 flex items-center justify-between bg-court-900 px-4 py-3 lg:hidden">
        <Link to="/" className="font-display text-xl font-bold text-white">
          Court<span className="text-clay-400">Craft</span>
        </Link>
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="rounded-lg border border-court-700 px-3 py-1.5 text-sm text-court-100"
          aria-label="Toggle navigation"
        >
          {menuOpen ? 'Close' : 'Menu'}
        </button>
      </header>
      {menuOpen && (
        <div className="fixed inset-0 top-[52px] z-30 lg:hidden">
          <Sidebar onNavigate={close} />
        </div>
      )}

      <main key={location.pathname} className="cc-fade-up min-w-0">
        <Outlet />
      </main>
    </div>
  )
}
