import { motion } from 'framer-motion'
import { Depth } from '../context/DepthContext'
import { TOKEN_COLORS } from '../lib/palette'

const FLOATERS = [
  { x: '12%', y: '22%', size: 10, delay: 0 },
  { x: '82%', y: '18%', size: 8, delay: 1.2 },
  { x: '70%', y: '70%', size: 12, delay: 0.6 },
  { x: '20%', y: '74%', size: 7, delay: 1.8 },
  { x: '48%', y: '12%', size: 6, delay: 2.4 },
  { x: '90%', y: '48%', size: 9, delay: 0.3 },
]

export default function Intro() {
  return (
    <section
      id="intro"
      data-section
      className="relative flex min-h-screen snap-start flex-col items-center justify-center overflow-hidden px-4 text-center"
    >
      {FLOATERS.map((f, i) => (
        <motion.span
          key={i}
          className="pointer-events-none absolute rounded-full"
          style={{
            left: f.x,
            top: f.y,
            width: f.size,
            height: f.size,
            background: TOKEN_COLORS[i % TOKEN_COLORS.length],
            opacity: 0.35,
          }}
          animate={{ y: [0, -18, 0], opacity: [0.2, 0.5, 0.2] }}
          transition={{ duration: 6 + i, repeat: Infinity, delay: f.delay, ease: 'easeInOut' }}
        />
      ))}

      <motion.h1
        className="max-w-3xl text-4xl font-bold tracking-tight sm:text-6xl"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7 }}
      >
        How LLMs <span className="text-tok-blue">Actually</span> Work
      </motion.h1>

      <motion.p
        className="mt-6 max-w-xl text-base leading-relaxed text-ink-2 sm:text-lg"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.15 }}
      >
        <Depth
          b="A guided, hands-on tour of what happens between your prompt and the model's reply — no math required. Five ideas, each one something you can poke at."
          t="An interactive walkthrough of the transformer pipeline: tokenization, embeddings, self-attention, stacked blocks, and sampling from the output distribution. Toy numbers, real mechanics."
        />
      </motion.p>

      <motion.p
        className="mt-4 text-sm text-ink-3"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        Use the <span className="font-medium text-ink-2">Beginner / Technical</span> toggle in the
        header — every section speaks both languages.
      </motion.p>

      <motion.a
        href="#tokens"
        aria-label="Scroll to the first section"
        className="absolute bottom-10 text-ink-3 transition-colors hover:text-ink"
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </motion.a>
    </section>
  )
}
