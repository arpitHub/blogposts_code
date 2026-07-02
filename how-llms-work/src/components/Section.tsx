import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

export default function Section({
  id,
  kicker,
  title,
  lede,
  children,
}: {
  id: string
  kicker: string
  title: string
  lede: ReactNode
  children: ReactNode
}) {
  return (
    <section
      id={id}
      data-section
      className="flex min-h-screen snap-start flex-col justify-center px-4 py-24 sm:px-6"
    >
      <motion.div
        className="mx-auto w-full max-w-4xl"
        initial={{ opacity: 0, y: 32 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-tok-blue">
          {kicker}
        </p>
        <h2 className="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">{title}</h2>
        <div className="mb-10 max-w-2xl text-base leading-relaxed text-ink-2">{lede}</div>
        {children}
      </motion.div>
    </section>
  )
}
