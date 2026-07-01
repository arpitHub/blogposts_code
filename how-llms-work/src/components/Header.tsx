import { motion } from 'framer-motion'
import { useDepth, type Depth } from '../context/DepthContext'

const OPTIONS: Array<{ value: Depth; label: string }> = [
  { value: 'beginner', label: 'Beginner' },
  { value: 'technical', label: 'Technical' },
]

export default function Header() {
  const { depth, setDepth } = useDepth()
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-hairline bg-page/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <a href="#intro" className="text-sm font-semibold tracking-tight text-ink">
          How LLMs <span className="text-tok-blue">Actually</span> Work
        </a>
        <div className="flex items-center gap-3">
          <span className="hidden text-xs text-ink-3 sm:block">Explain like I’m…</span>
          <div
            role="radiogroup"
            aria-label="Explanation depth"
            className="relative flex rounded-full border border-hairline bg-surface p-1"
          >
            {OPTIONS.map((opt) => (
              <button
                key={opt.value}
                role="radio"
                aria-checked={depth === opt.value}
                onClick={() => setDepth(opt.value)}
                className={`relative z-10 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                  depth === opt.value ? 'text-white' : 'text-ink-3 hover:text-ink-2'
                }`}
              >
                {depth === opt.value && (
                  <motion.span
                    layoutId="depth-pill"
                    className="absolute inset-0 -z-10 rounded-full bg-tok-blue"
                    transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                  />
                )}
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </header>
  )
}
