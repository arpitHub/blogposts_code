import { createContext, useContext } from 'react'
import type { ReactNode } from 'react'

export type Depth = 'beginner' | 'technical'

export const DepthContext = createContext<Depth>('beginner')

export function useDepth(): Depth {
  return useContext(DepthContext)
}

/** Renders beginner or technical copy depending on the global toggle. */
export function D({ b, t }: { b: ReactNode; t: ReactNode }) {
  const depth = useDepth()
  return <>{depth === 'beginner' ? b : t}</>
}
