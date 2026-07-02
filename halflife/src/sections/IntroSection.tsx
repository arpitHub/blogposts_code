import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { mulberry32 } from '../lib/decay'
import { D } from '../lib/depth'

export function IntroSection() {
  const stars = useMemo(() => {
    const rand = mulberry32(42)
    return Array.from({ length: 90 }, (_, i) => ({
      id: i,
      x: rand() * 100,
      y: rand() * 100,
      r: 0.5 + rand() * 1.4,
      amber: rand() < 0.3,
      dur: 2.5 + rand() * 5,
      delay: rand() * 5,
    }))
  }, [])

  return (
    <section
      id="intro"
      className="relative flex min-h-[100svh] snap-start flex-col items-center justify-center overflow-hidden px-6 text-center"
    >
      <svg
        className="pointer-events-none absolute inset-0 h-full w-full"
        preserveAspectRatio="none"
        aria-hidden
      >
        {stars.map((s) => (
          <circle
            key={s.id}
            cx={`${s.x}%`}
            cy={`${s.y}%`}
            r={s.r}
            fill={s.amber ? '#f5b846' : '#7db5f0'}
            style={{
              animation: `hl-twinkle ${s.dur}s ease-in-out ${s.delay}s infinite`,
            }}
          />
        ))}
      </svg>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        className="relative"
      >
        <h1 className="text-5xl font-bold tracking-tight sm:text-7xl">
          <span className="text-amber-glow">Half</span>
          <span className="text-blue-glow">Life</span>
        </h1>
        <p className="mx-auto mt-5 max-w-xl text-lg leading-relaxed text-ink-2">
          <D
            b="How did we figure out the Earth is 4.55 billion years old? Not from a book — from atoms that keep time. Scroll to watch the clock tick."
            t="An interactive tour of radiometric dating: exponential decay, decay-chain systematics, the Pb–Pb isochron that dated the Earth, and the helioseismic cross-check. Scroll to begin."
          />
        </p>
        <div className="mx-auto mt-8 flex max-w-md items-center justify-center gap-2 text-xs text-ink-3">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-glow" />
          ancient / parent isotope
          <span className="ml-4 inline-block h-2.5 w-2.5 rounded-full bg-blue-series" />
          recent / daughter isotope
        </div>
      </motion.div>

      <motion.button
        onClick={() =>
          document.getElementById('decay')?.scrollIntoView({ behavior: 'smooth' })
        }
        className="absolute bottom-10 text-ink-3 transition-colors hover:text-ink"
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
        aria-label="Scroll to first section"
      >
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M6 9l6 6 6-6"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </motion.button>
    </section>
  )
}
