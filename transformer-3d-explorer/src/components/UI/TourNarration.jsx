import { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import useAppStore from '../../store/useAppStore'
import { getTourScript, TOUR_STEP_SECONDS } from '../../data/tourScripts'

/**
 * Bottom narration bar for the Guided Tour: current step's plain-language
 * explanation plus Play/Pause, Previous and Next controls. While playing,
 * steps auto-advance every TOUR_STEP_SECONDS.
 */
export default function TourNarration() {
  const architectureMode = useAppStore((s) => s.architectureMode)
  const exploreMode = useAppStore((s) => s.exploreMode)
  const currentTourStep = useAppStore((s) => s.currentTourStep)
  const tourPlaying = useAppStore((s) => s.tourPlaying)
  const setTourPlaying = useAppStore((s) => s.setTourPlaying)
  const nextTourStep = useAppStore((s) => s.nextTourStep)
  const prevTourStep = useAppStore((s) => s.prevTourStep)

  const script = getTourScript(architectureMode)
  const total = script.length
  const step = script[Math.min(currentTourStep, total - 1)]

  useEffect(() => {
    if (!tourPlaying || exploreMode !== 'tour') return undefined
    const id = setInterval(
      () => nextTourStep(total),
      TOUR_STEP_SECONDS * 1000
    )
    return () => clearInterval(id)
  }, [tourPlaying, exploreMode, total, nextTourStep])

  if (exploreMode !== 'tour' || !step) return null

  const atStart = currentTourStep === 0
  const atEnd = currentTourStep >= total - 1

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 flex justify-center p-3">
      <div className="pointer-events-auto w-full max-w-2xl rounded-xl bg-slate-900/90 p-3 shadow-2xl ring-1 ring-slate-700/60 backdrop-blur">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Guided Tour · Step {currentTourStep + 1} / {total}
          </span>
          <div className="flex items-center gap-1.5">
            <button
              onClick={prevTourStep}
              disabled={atStart}
              aria-label="Previous step"
              className="rounded-md bg-slate-800 px-2.5 py-1 text-sm text-slate-200 hover:bg-slate-700 disabled:opacity-40"
            >
              ‹ Prev
            </button>
            <button
              onClick={() => setTourPlaying(!tourPlaying)}
              aria-label={tourPlaying ? 'Pause tour' : 'Play tour'}
              className="rounded-md bg-indigo-600 px-3 py-1 text-sm font-semibold text-white hover:bg-indigo-500"
            >
              {tourPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
            <button
              onClick={() => nextTourStep(total)}
              disabled={atEnd}
              aria-label="Next step"
              className="rounded-md bg-slate-800 px-2.5 py-1 text-sm text-slate-200 hover:bg-slate-700 disabled:opacity-40"
            >
              Next ›
            </button>
          </div>
        </div>

        <AnimatePresence mode="wait">
          <motion.p
            key={`${architectureMode}-${currentTourStep}`}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.25 }}
            className="mt-2 text-sm leading-relaxed text-slate-100"
          >
            {step.narration}
          </motion.p>
        </AnimatePresence>

        {/* progress dots */}
        <div className="mt-2 flex flex-wrap gap-1">
          {script.map((s, i) => (
            <button
              key={s.blockId + i}
              onClick={() => useAppStore.getState().setTourStep(i)}
              aria-label={`Go to step ${i + 1}`}
              className={`h-1.5 flex-1 rounded-full transition-colors ${
                i <= currentTourStep ? 'bg-indigo-500' : 'bg-slate-700'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
