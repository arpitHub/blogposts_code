import { useEffect, useState } from 'react'
import { SECTIONS } from './Section'

/**
 * Right-edge progress dots. Tracks the section nearest the viewport center
 * via IntersectionObserver on the scroll container.
 */
export function SectionNav({
  scrollRef,
}: {
  scrollRef: React.RefObject<HTMLElement | null>
}) {
  const [active, setActive] = useState('intro')

  useEffect(() => {
    const root = scrollRef.current
    if (!root) return
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setActive(e.target.id)
        }
      },
      { root, rootMargin: '-45% 0px -45% 0px' },
    )
    for (const s of SECTIONS) {
      const el = document.getElementById(s.id)
      if (el) observer.observe(el)
    }
    return () => observer.disconnect()
  }, [scrollRef])

  return (
    <nav
      aria-label="Sections"
      className="fixed right-3 top-1/2 z-40 hidden -translate-y-1/2 flex-col gap-3 md:flex"
    >
      {SECTIONS.map((s) => (
        <a
          key={s.id}
          href={`#${s.id}`}
          onClick={(e) => {
            e.preventDefault()
            document
              .getElementById(s.id)
              ?.scrollIntoView({ behavior: 'smooth' })
          }}
          className="group relative flex items-center justify-end"
          aria-current={active === s.id ? 'true' : undefined}
        >
          <span className="pointer-events-none absolute right-6 whitespace-nowrap rounded-md border border-line bg-panel px-2 py-1 text-xs text-ink-2 opacity-0 transition-opacity group-hover:opacity-100">
            {s.navLabel}
          </span>
          <span
            className={`block rounded-full transition-all duration-300 ${
              active === s.id
                ? 'h-2.5 w-2.5 bg-amber-glow shadow-[0_0_8px_rgba(245,184,70,0.6)]'
                : 'h-2 w-2 bg-axis group-hover:bg-ink-3'
            }`}
          />
        </a>
      ))}
    </nav>
  )
}
