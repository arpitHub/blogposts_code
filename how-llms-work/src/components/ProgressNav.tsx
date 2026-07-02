export interface SectionMeta {
  id: string
  label: string
}

export default function ProgressNav({
  sections,
  active,
}: {
  sections: SectionMeta[]
  active: string
}) {
  return (
    <nav
      aria-label="Sections"
      className="fixed right-4 top-1/2 z-40 hidden -translate-y-1/2 flex-col gap-3 md:flex"
    >
      {sections.map((s) => {
        const isActive = s.id === active
        return (
          <a
            key={s.id}
            href={`#${s.id}`}
            aria-current={isActive ? 'true' : undefined}
            className="group flex items-center justify-end gap-2"
          >
            <span
              className={`text-[11px] transition-opacity ${
                isActive ? 'text-ink-2 opacity-100' : 'text-ink-3 opacity-0 group-hover:opacity-100'
              }`}
            >
              {s.label}
            </span>
            <span
              className={`block rounded-full transition-all duration-300 ${
                isActive ? 'h-5 w-2 bg-tok-blue' : 'h-2 w-2 bg-axis group-hover:bg-ink-3'
              }`}
            />
          </a>
        )
      })}
    </nav>
  )
}
