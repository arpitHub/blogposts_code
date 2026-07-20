import { create } from 'zustand'

/**
 * Global app state.
 *
 * architectureMode: 'encoder-decoder' (original Transformer) | 'decoder-only' (GPT-style)
 * exploreMode:     'tour' (guided walkthrough) | 'free' (orbit + click to inspect)
 * currentTourStep: index into the active architecture's tour script
 * tourPlaying:     whether the guided tour auto-advances
 * selectedBlockId: block whose info panel is open in Free Explore (null = closed)
 * hoveredBlockId:  block currently hovered in the 3D scene (tooltip)
 */
const useAppStore = create((set) => ({
  architectureMode: 'encoder-decoder',
  exploreMode: 'tour',
  currentTourStep: 0,
  tourPlaying: false,
  selectedBlockId: null,
  hoveredBlockId: null,

  setArchitectureMode: (mode) =>
    set({
      architectureMode: mode,
      // A new architecture has its own tour script and blocks — reset progress
      currentTourStep: 0,
      tourPlaying: false,
      selectedBlockId: null,
      hoveredBlockId: null
    }),

  setExploreMode: (mode) =>
    set({
      exploreMode: mode,
      tourPlaying: false,
      selectedBlockId: null,
      hoveredBlockId: null
    }),

  setTourStep: (step) => set({ currentTourStep: step }),

  nextTourStep: (totalSteps) =>
    set((state) => ({
      currentTourStep: Math.min(state.currentTourStep + 1, totalSteps - 1),
      tourPlaying:
        state.currentTourStep + 1 >= totalSteps ? false : state.tourPlaying
    })),

  prevTourStep: () =>
    set((state) => ({
      currentTourStep: Math.max(state.currentTourStep - 1, 0)
    })),

  setTourPlaying: (playing) => set({ tourPlaying: playing }),

  selectBlock: (blockId) => set({ selectedBlockId: blockId }),
  clearSelection: () => set({ selectedBlockId: null }),

  setHoveredBlock: (blockId) => set({ hoveredBlockId: blockId })
}))

export default useAppStore
