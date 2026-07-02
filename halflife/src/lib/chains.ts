/** Real decay-chain data (main branches; rare branchings omitted). */

const MIN = 60
const HR = 3600
const DAY = 86400
const YR = 3.156e7

export type DecayMode = 'α' | 'β'

export interface ChainStep {
  isotope: string
  /** half-life in seconds; undefined = stable end of chain */
  halfLifeS?: number
  mode?: DecayMode
}

export interface Chain {
  id: string
  name: string
  parent: string
  end: string
  note: string
  steps: ChainStep[]
}

export const CHAINS: Chain[] = [
  {
    id: 'u238',
    name: 'U-238 → Pb-206',
    parent: '²³⁸U',
    end: '²⁰⁶Pb',
    note: 'the uranium series — the chain behind uranium–lead dating',
    steps: [
      { isotope: 'U-238', mode: 'α', halfLifeS: 4.468e9 * YR },
      { isotope: 'Th-234', mode: 'β', halfLifeS: 24.1 * DAY },
      { isotope: 'Pa-234m', mode: 'β', halfLifeS: 1.17 * MIN },
      { isotope: 'U-234', mode: 'α', halfLifeS: 245500 * YR },
      { isotope: 'Th-230', mode: 'α', halfLifeS: 75380 * YR },
      { isotope: 'Ra-226', mode: 'α', halfLifeS: 1600 * YR },
      { isotope: 'Rn-222', mode: 'α', halfLifeS: 3.8235 * DAY },
      { isotope: 'Po-218', mode: 'α', halfLifeS: 3.098 * MIN },
      { isotope: 'Pb-214', mode: 'β', halfLifeS: 26.8 * MIN },
      { isotope: 'Bi-214', mode: 'β', halfLifeS: 19.9 * MIN },
      { isotope: 'Po-214', mode: 'α', halfLifeS: 164.3e-6 },
      { isotope: 'Pb-210', mode: 'β', halfLifeS: 22.2 * YR },
      { isotope: 'Bi-210', mode: 'β', halfLifeS: 5.012 * DAY },
      { isotope: 'Po-210', mode: 'α', halfLifeS: 138.4 * DAY },
      { isotope: 'Pb-206' },
    ],
  },
  {
    id: 'u235',
    name: 'U-235 → Pb-207',
    parent: '²³⁵U',
    end: '²⁰⁷Pb',
    note: 'the actinium series — the second clock inside every zircon',
    steps: [
      { isotope: 'U-235', mode: 'α', halfLifeS: 7.04e8 * YR },
      { isotope: 'Th-231', mode: 'β', halfLifeS: 25.52 * HR },
      { isotope: 'Pa-231', mode: 'α', halfLifeS: 32760 * YR },
      { isotope: 'Ac-227', mode: 'β', halfLifeS: 21.77 * YR },
      { isotope: 'Th-227', mode: 'α', halfLifeS: 18.68 * DAY },
      { isotope: 'Ra-223', mode: 'α', halfLifeS: 11.43 * DAY },
      { isotope: 'Rn-219', mode: 'α', halfLifeS: 3.96 },
      { isotope: 'Po-215', mode: 'α', halfLifeS: 1.781e-3 },
      { isotope: 'Pb-211', mode: 'β', halfLifeS: 36.1 * MIN },
      { isotope: 'Bi-211', mode: 'α', halfLifeS: 2.14 * MIN },
      { isotope: 'Tl-207', mode: 'β', halfLifeS: 4.77 * MIN },
      { isotope: 'Pb-207' },
    ],
  },
  {
    id: 'th232',
    name: 'Th-232 → Pb-208',
    parent: '²³²Th',
    end: '²⁰⁸Pb',
    note: 'the thorium series — its parent outlives the universe so far',
    steps: [
      { isotope: 'Th-232', mode: 'α', halfLifeS: 1.405e10 * YR },
      { isotope: 'Ra-228', mode: 'β', halfLifeS: 5.75 * YR },
      { isotope: 'Ac-228', mode: 'β', halfLifeS: 6.15 * HR },
      { isotope: 'Th-228', mode: 'α', halfLifeS: 1.9116 * YR },
      { isotope: 'Ra-224', mode: 'α', halfLifeS: 3.66 * DAY },
      { isotope: 'Rn-220', mode: 'α', halfLifeS: 55.6 },
      { isotope: 'Po-216', mode: 'α', halfLifeS: 0.145 },
      { isotope: 'Pb-212', mode: 'β', halfLifeS: 10.64 * HR },
      { isotope: 'Bi-212', mode: 'β', halfLifeS: 60.55 * MIN },
      { isotope: 'Po-212', mode: 'α', halfLifeS: 299e-9 },
      { isotope: 'Pb-208' },
    ],
  },
]

/**
 * Animation dwell time for a step: log-mapped so µs and Gyr both stay
 * watchable (0.35 s – 2.6 s of wall time).
 */
export function dwellSeconds(halfLifeS: number): number {
  const lg = Math.log10(halfLifeS) // ~ -6.8 … 17.6
  const norm = Math.min(1, Math.max(0, (lg + 7) / 25))
  return 0.35 + norm * 2.25
}
