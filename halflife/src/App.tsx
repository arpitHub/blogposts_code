import { useRef, useState } from 'react'
import { DepthContext } from './lib/depth'
import type { Depth } from './lib/depth'
import { DepthToggle } from './components/DepthToggle'
import { SectionNav } from './components/SectionNav'
import { IntroSection } from './sections/IntroSection'
import { DecaySection } from './sections/DecaySection'
import { DiscoverySection } from './sections/DiscoverySection'
import { ChainsSection } from './sections/ChainsSection'
import { ClocksSection } from './sections/ClocksSection'
import { IsochronSection } from './sections/IsochronSection'
import { HelioSection } from './sections/HelioSection'
import { DeepTimeSection } from './sections/DeepTimeSection'

export default function App() {
  const [depth, setDepth] = useState<Depth>('beginner')
  const scrollRef = useRef<HTMLElement>(null)

  return (
    <DepthContext.Provider value={depth}>
      <header className="fixed inset-x-0 top-0 z-50 border-b border-line/70 bg-void/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-8">
          <a
            href="#intro"
            onClick={(e) => {
              e.preventDefault()
              document
                .getElementById('intro')
                ?.scrollIntoView({ behavior: 'smooth' })
            }}
            className="text-lg font-bold tracking-tight"
          >
            <span className="text-amber-glow">Half</span>
            <span className="text-blue-glow">Life</span>
          </a>
          <DepthToggle depth={depth} onChange={setDepth} />
        </div>
      </header>

      <SectionNav scrollRef={scrollRef} />

      <main ref={scrollRef} className="snap-main">
        <IntroSection />
        <DecaySection />
        <DiscoverySection />
        <ChainsSection />
        <ClocksSection />
        <IsochronSection />
        <HelioSection />
        <DeepTimeSection />
        <footer className="border-t border-line px-6 py-10 text-center text-xs leading-relaxed text-ink-3">
          Built as an interactive explainer. Half-lives, isochron data and ages
          are real published values; the particle simulations are pedagogical
          (a few hundred atoms, not 10²³).
        </footer>
      </main>
    </DepthContext.Provider>
  )
}
