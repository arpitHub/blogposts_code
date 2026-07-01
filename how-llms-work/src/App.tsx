import { useEffect, useRef, useState } from 'react'
import { DepthProvider } from './context/DepthContext'
import Header from './components/Header'
import ProgressNav, { type SectionMeta } from './components/ProgressNav'
import Intro from './sections/Intro'
import Tokens from './sections/Tokens'
import Embeddings from './sections/Embeddings'
import Attention from './sections/Attention'
import TransformerStack from './sections/TransformerStack'
import Generation from './sections/Generation'

const SECTIONS: SectionMeta[] = [
  { id: 'intro', label: 'Intro' },
  { id: 'tokens', label: 'Tokens' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'attention', label: 'Attention' },
  { id: 'stack', label: 'Transformer' },
  { id: 'generation', label: 'Generation' },
]

export default function App() {
  const [active, setActive] = useState('intro')
  const mainRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const root = mainRef.current
    if (!root) return
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActive(entry.target.id)
        }
      },
      { root, threshold: 0.5 },
    )
    root.querySelectorAll('[data-section]').forEach((el) => observer.observe(el))
    return () => observer.disconnect()
  }, [])

  return (
    <DepthProvider>
      <Header />
      <ProgressNav sections={SECTIONS} active={active} />
      <main
        ref={mainRef}
        className="snap-container h-screen snap-y snap-proximity scroll-pt-14 overflow-y-auto scroll-smooth"
      >
        <Intro />
        <Tokens />
        <Embeddings />
        <Attention />
        <TransformerStack />
        <Generation />
        <footer className="snap-start px-4 py-10 text-center text-xs text-ink-3">
          Built as an interactive explainer · all data is precomputed or mocked — no model calls, no
          backend.
        </footer>
      </main>
    </DepthProvider>
  )
}
