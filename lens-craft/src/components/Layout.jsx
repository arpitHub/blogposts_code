import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar.jsx';

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="min-h-screen bg-void text-ink">
      {/* mobile top bar */}
      <div className="flex items-center justify-between border-b border-line px-4 py-3 md:hidden">
        <span className="font-display text-base font-semibold">Lens Craft</span>
        <button
          onClick={() => setMobileOpen((v) => !v)}
          className="rounded-md border border-line px-3 py-1.5 text-sm text-ink-2"
          aria-label="Toggle navigation"
        >
          {mobileOpen ? 'Close' : 'Menu'}
        </button>
      </div>

      <div className="mx-auto flex max-w-[1400px]">
        <aside className="hidden w-64 shrink-0 border-r border-line md:block">
          <div className="sticky top-0 h-screen">
            <Sidebar />
          </div>
        </aside>

        {mobileOpen && (
          <div className="fixed inset-0 z-40 bg-void/95 backdrop-blur md:hidden">
            <Sidebar onNavigate={() => setMobileOpen(false)} />
          </div>
        )}

        <main className="min-w-0 flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
