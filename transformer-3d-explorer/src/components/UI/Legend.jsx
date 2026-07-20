import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import useAppStore from '../../store/useAppStore'
import { legendFor } from '../../data/architectureData'

/**
 * Color legend, always visible bottom-left; collapsible on small screens.
 * Entries adapt to the active architecture (no cross-attention row in
 * decoder-only mode).
 */
export default function Legend() {
  const architectureMode = useAppStore((s) => s.architectureMode)
  const [open, setOpen] = useState(true)
  const entries = legendFor(architectureMode)

  return (
    <div className="pointer-events-auto absolute bottom-24 left-3 z-20 sm:bottom-4">
      <button
        onClick={() => setOpen((o) => !o)}
        className="mb-1 flex items-center gap-1.5 rounded-md bg-slate-800/80 px-2.5 py-1 text-xs font-semibold text-slate-200 backdrop-blur hover:bg-slate-700/80"
        aria-expanded={open}
      >
        <span
          className={`inline-block transition-transform ${open ? 'rotate-90' : ''}`}
        >
          ▸
        </span>
        Legend
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.ul
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden rounded-lg bg-slate-900/80 p-2.5 text-xs text-slate-200 shadow-lg ring-1 ring-slate-700/60 backdrop-blur"
          >
            {entries.map((e) => (
              <li key={e.type} className="flex items-center gap-2 py-0.5">
                <span
                  className="h-3 w-3 shrink-0 rounded-sm ring-1 ring-white/20"
                  style={{ backgroundColor: e.color }}
                />
                {e.label}
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
