import { motion, AnimatePresence } from 'framer-motion'
import useAppStore from '../../store/useAppStore'
import {
  getBlockById,
  pipelinePositionOf
} from '../../data/architectureData'

/**
 * Slide-in panel (right side) shown when a block is selected in Free Explore.
 */
export default function InfoPanel() {
  const architectureMode = useAppStore((s) => s.architectureMode)
  const selectedBlockId = useAppStore((s) => s.selectedBlockId)
  const clearSelection = useAppStore((s) => s.clearSelection)

  const block = selectedBlockId
    ? getBlockById(architectureMode, selectedBlockId)
    : null

  return (
    <AnimatePresence>
      {block && (
        <motion.aside
          key={block.id}
          initial={{ x: '110%' }}
          animate={{ x: 0 }}
          exit={{ x: '110%' }}
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
          className="pointer-events-auto absolute right-0 top-16 z-30 m-3 w-[min(20rem,calc(100vw-1.5rem))] rounded-xl bg-slate-900/90 shadow-2xl ring-1 ring-slate-700/60 backdrop-blur"
          role="dialog"
          aria-label={`About ${block.label}`}
        >
          <div
            className="rounded-t-xl px-4 py-3"
            style={{ backgroundColor: `${block.color}33` }}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <span
                    className="h-3 w-3 rounded-full ring-1 ring-white/30"
                    style={{ backgroundColor: block.color }}
                  />
                  <h2 className="text-sm font-bold text-white">
                    {block.label}
                  </h2>
                </div>
                <p className="mt-1 text-[11px] uppercase tracking-wide text-slate-400">
                  {pipelinePositionOf(architectureMode, block.id)}
                </p>
              </div>
              <button
                onClick={clearSelection}
                aria-label="Close info panel"
                className="rounded-md px-2 py-0.5 text-slate-400 hover:bg-slate-700/60 hover:text-white"
              >
                ✕
              </button>
            </div>
          </div>

          <div className="panel-scroll max-h-[40vh] overflow-y-auto px-4 py-3">
            <p className="text-sm leading-relaxed text-slate-200">
              {block.description}
            </p>
            <p className="mt-3 text-xs italic text-slate-500">
              Visuals are illustrative — attention strengths and vectors are
              stylized for teaching, not computed from real data.
            </p>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}
