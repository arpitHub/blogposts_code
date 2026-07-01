import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

export type Depth = 'beginner' | 'technical'

const DepthContext = createContext<{ depth: Depth; setDepth: (d: Depth) => void }>({
  depth: 'beginner',
  setDepth: () => {},
})

export function DepthProvider({ children }: { children: ReactNode }) {
  const [depth, setDepth] = useState<Depth>(() => {
    const saved = localStorage.getItem('explain-depth')
    return saved === 'technical' ? 'technical' : 'beginner'
  })
  useEffect(() => {
    localStorage.setItem('explain-depth', depth)
  }, [depth])
  return <DepthContext.Provider value={{ depth, setDepth }}>{children}</DepthContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useDepth() {
  return useContext(DepthContext)
}

/** Renders one of two texts depending on the global depth toggle. */
export function Depth({ b, t }: { b: ReactNode; t: ReactNode }) {
  const { depth } = useDepth()
  return <>{depth === 'beginner' ? b : t}</>
}
