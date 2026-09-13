import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Lean on purpose: just which myths have been guessed, and what the guess was.
// No badges, streaks, or scoring — those can come later.
export const useMythStore = create(
  persist(
    (set, get) => ({
      guesses: {},
      setGuess: (slug, guess) =>
        set((state) => ({ guesses: { ...state.guesses, [slug]: guess } })),
      getGuess: (slug) => get().guesses[slug],
    }),
    { name: 'mythcraft-progress' }
  )
);
