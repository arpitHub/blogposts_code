import type { ReactNode } from 'react'
import { motion } from 'framer-motion'

export interface SectionMeta {
  id: string
  navLabel: string
}

export const SECTIONS: SectionMeta[] = [
  { id: 'intro', navLabel: 'HalfLife' },
  { id: 'decay', navLabel: 'Radioactive decay' },
  { id: 'discovery', navLabel: 'The discovery story' },
  { id: 'chains', navLabel: 'Decay chains' },
  { id: 'clocks', navLabel: 'Different clocks' },
  { id: 'isochron', navLabel: 'Dating the Earth' },
  { id: 'sun', navLabel: 'The Sun agrees' },
  { id: 'deeptime', navLabel: 'Deep time' },
]

export function Section({
  id,
  kicker,
  title,
  children,
}: {
  id: string
  kicker: string
  title: string
  children: ReactNode
}) {
  return (
    <section
      id={id}
      className="relative flex min-h-[100svh] snap-start flex-col justify-center px-4 py-20 sm:px-8 lg:px-16"
    >
      <div className="mx-auto w-full max-w-6xl">
        <motion.header
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.55, ease: 'easeOut' }}
          className="mb-8"
        >
          <div className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-amber-glow">
            {kicker}
          </div>
          <h2 className="text-3xl font-bold tracking-tight text-ink sm:text-4xl">
            {title}
          </h2>
        </motion.header>
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.55, delay: 0.1, ease: 'easeOut' }}
        >
          {children}
        </motion.div>
      </div>
    </section>
  )
}
