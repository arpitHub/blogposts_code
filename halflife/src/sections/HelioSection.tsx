import { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Section } from '../components/Section'
import { EqCard, SliderRow } from '../components/ui'
import { D, useDepth } from '../lib/depth'

const CX = 200
const CY = 168
const R = 122

type View = 'modes' | 'rays'

export function HelioSection() {
  const depth = useDepth()
  const [view, setView] = useState<View>('modes')
  const [ell, setEll] = useState(4)

  return (
    <Section id="sun" kicker="06 · The cross-check" title="The Sun rings like a bell — and agrees">
      <div className="grid gap-8 lg:grid-cols-[minmax(0,6fr)_minmax(0,6fr)]">
        <div>
          <div className="mb-3 flex gap-2">
            {(
              [
                ['modes', 'Surface oscillation'],
                ['rays', 'Sound-wave paths'],
              ] as const
            ).map(([v, label]) => (
              <button
                key={v}
                role="tab"
                aria-selected={view === v}
                onClick={() => setView(v)}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  view === v
                    ? 'bg-amber-glow text-void'
                    : 'border border-line bg-panel text-ink-2 hover:text-ink'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="rounded-xl border border-line bg-panel p-2">
            <SunViz view={view} ell={ell} />
          </div>

          <div className="mt-4">
            <SliderRow
              label={
                <D
                  b="Wave pattern complexity"
                  t={<>Spherical-harmonic degree ℓ</>}
                />
              }
              min={2}
              max={8}
              step={1}
              value={ell}
              onChange={setEll}
              display={depth === 'technical' ? `ℓ = ${ell}` : String(ell)}
              blue
            />
            <p className="mt-2 text-xs leading-relaxed text-ink-3">
              <D
                b="Regions pulsing toward you are amber, away from you blue. Simpler patterns dive deep into the core; busier ones stay near the surface."
                t="Rising patches amber, sinking blue (as Doppler imaging sees them). Low-ℓ modes have deep turning points and sample the core; high-ℓ modes are trapped near the surface — inverting millions of mode frequencies maps ρ and c(r) through the interior."
              />
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-5">
          <p className="max-w-prose text-[15px] leading-relaxed text-ink-2">
            <D
              b={
                <>
                  Here’s a completely different clock — no radioactivity
                  involved. The Sun hums with millions of trapped sound waves,
                  like a struck bell that never stops. The exact notes depend on
                  what’s inside — and what’s inside changes as the Sun slowly
                  burns hydrogen into helium. Reading the helium built up in the
                  core is reading the Sun’s age off a fuel gauge:{' '}
                  <strong>about 4.6 billion years</strong>. Rocks and sunquakes
                  know nothing about each other. They agree anyway.
                </>
              }
              t={
                <>
                  Doppler imaging (SOHO/MDI, GONG) resolves ~10⁷ acoustic
                  p-modes near 3 mHz. Their frequencies — especially the small
                  separations δν<sub>ℓ,ℓ+2</sub> — are sensitive to the sound
                  speed in the core, hence to the mean molecular weight raised
                  by 4.6 Gyr of H→He fusion. Calibrated solar models fit the
                  seismic data at <strong>t☉ ≈ 4.57 ± 0.11 Gyr</strong>{' '}
                  (Bonanno et al. 2002) — statistically indistinguishable from
                  the meteoritic Pb–Pb age, with entirely independent physics.
                </>
              }
            />
          </p>

          <AgreementChart />

          {depth === 'technical' && (
            <EqCard
              note="Two chronometers, zero shared assumptions: nuclear decay counting vs. stellar-structure seismology."
            >
              Pb–Pb meteorites: 4.567 ± 0.007 Gyr&ensp;·&ensp;helioseismology:
              4.57 ± 0.11 Gyr
            </EqCard>
          )}
        </div>
      </div>
    </Section>
  )
}

/** The oscillating sun: harmonic patches or refracting ray paths */
function SunViz({ view, ell }: { view: View; ell: number }) {
  const wedges = useMemo(() => {
    const n = 2 * ell
    const step = (Math.PI * 2) / n
    return Array.from({ length: n }, (_, i) => {
      const a0 = -Math.PI / 2 + i * step
      const a1 = a0 + step
      const large = step > Math.PI ? 1 : 0
      const p0 = [CX + R * Math.cos(a0), CY + R * Math.sin(a0)]
      const p1 = [CX + R * Math.cos(a1), CY + R * Math.sin(a1)]
      return {
        d: `M ${CX} ${CY} L ${p0[0]} ${p0[1]} A ${R} ${R} 0 ${large} 1 ${p1[0]} ${p1[1]} Z`,
        even: i % 2 === 0,
      }
    })
  }, [ell])

  // acoustic ray: chords refracting around a turning radius; more bounces
  // (shallower turning point) at higher ℓ
  const rayPath = useMemo(() => {
    const bounces = ell + 3
    const rTurn = R * (0.15 + ell * 0.07)
    const step = (Math.PI * 2) / bounces
    let d = ''
    for (let i = 0; i <= bounces; i++) {
      const a = -Math.PI / 2 + i * step
      const x = CX + R * Math.cos(a)
      const y = CY + R * Math.sin(a)
      if (i === 0) {
        d = `M ${x} ${y}`
      } else {
        const am = a - step / 2
        const cx = CX + rTurn * Math.cos(am)
        const cy = CY + rTurn * Math.sin(am)
        d += ` Q ${cx} ${cy} ${x} ${y}`
      }
    }
    return d
  }, [ell])

  return (
    <svg
      viewBox="0 0 400 336"
      className="h-auto w-full"
      role="img"
      aria-label="Cross-section of the Sun showing acoustic oscillation patterns"
    >
      <defs>
        <radialGradient id="sunBody" cx="42%" cy="38%" r="75%">
          <stop offset="0%" stopColor="#3d2f12" />
          <stop offset="55%" stopColor="#241d0e" />
          <stop offset="100%" stopColor="#12100a" />
        </radialGradient>
        <radialGradient id="sunCore" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#f5b846" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#f5b846" stopOpacity="0" />
        </radialGradient>
        <clipPath id="sunClip">
          <circle cx={CX} cy={CY} r={R} />
        </clipPath>
      </defs>

      {/* corona glow */}
      <circle cx={CX} cy={CY} r={R + 9} fill="none" stroke="#f5b846" strokeOpacity="0.18" strokeWidth="10" />
      <circle cx={CX} cy={CY} r={R} fill="url(#sunBody)" stroke="#c98500" strokeOpacity="0.6" strokeWidth="1.5" />

      {view === 'modes' ? (
        <g clipPath="url(#sunClip)">
          {/* breathing whole-disk pulse to feel alive */}
          <motion.circle
            cx={CX} cy={CY} r={R}
            fill="url(#sunCore)"
            animate={{ opacity: [0.5, 0.9, 0.5] }}
            transition={{ duration: 3.4, repeat: Infinity, ease: 'easeInOut' }}
          />
          {wedges.map((w, i) => (
            <motion.path
              key={`${ell}-${i}`}
              d={w.d}
              fill={w.even ? '#f5b846' : '#3987e5'}
              animate={{ opacity: w.even ? [0.28, 0.04, 0.28] : [0.04, 0.28, 0.04] }}
              transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }}
            />
          ))}
          {/* radial node circle for a hint of depth structure */}
          <motion.circle
            cx={CX} cy={CY}
            fill="none" stroke="#0a0a0f" strokeOpacity="0.55" strokeWidth="6"
            initial={{ r: R * 0.55 }}
            animate={{ r: [R * 0.52, R * 0.58, R * 0.52] }}
            transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }}
          />
        </g>
      ) : (
        <g clipPath="url(#sunClip)">
          <circle cx={CX} cy={CY} r={R * 0.28} fill="url(#sunCore)" opacity="0.8" />
          <text x={CX} y={CY + 4} textAnchor="middle" fontSize="10" fill="#f5b846" opacity="0.9">
            core
          </text>
          {/* the ray cage, static & faint */}
          <path d={rayPath} fill="none" stroke="#3987e5" strokeOpacity="0.25" strokeWidth="1.2" />
          {/* travelling wave along the same path */}
          <motion.path
            d={rayPath}
            fill="none"
            stroke="#7db5f0"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeDasharray="26 240"
            animate={{ strokeDashoffset: [0, -1064] }}
            transition={{ duration: 6.5, repeat: Infinity, ease: 'linear' }}
          />
        </g>
      )}

      <text x={CX} y={322} textAnchor="middle" fontSize="10" fill="#8a8894">
        {view === 'modes'
          ? 'standing sound waves seen at the surface'
          : 'sound waves refract off the deep interior and return to the surface'}
      </text>
    </svg>
  )
}

/** Two independent methods, one answer — interval comparison on a shared axis */
function AgreementChart() {
  const X0 = 4.4
  const X1 = 4.75
  const W2 = 460
  const ML = 158
  const MR = 16
  const px = (v: number) => ML + ((v - X0) / (X1 - X0)) * (W2 - ML - MR)
  const rows = [
    { label: 'Meteorite Pb–Pb', v: 4.567, err: 0.007, color: '#c98500' },
    { label: 'Helioseismology', v: 4.57, err: 0.11, color: '#3987e5' },
  ]
  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <div className="mb-2 text-sm font-medium text-ink">
        Two unrelated clocks, one answer
      </div>
      <svg viewBox={`0 0 ${W2} 108`} className="h-auto w-full" role="img" aria-label="Meteorite and helioseismic ages agree within uncertainties">
        {[4.4, 4.5, 4.6, 4.7].map((v) => (
          <g key={v}>
            <line x1={px(v)} y1={16} x2={px(v)} y2={78} stroke="#1c1c26" />
            <text x={px(v)} y={94} textAnchor="middle" fontSize="10" fill="#8a8894">
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        <text x={(ML + W2 - MR) / 2} y={107} textAnchor="middle" fontSize="9" fill="#8a8894">
          age (billion years)
        </text>
        {rows.map((r, i) => {
          const y = 32 + i * 32
          return (
            <g key={r.label}>
              <text x={ML - 10} y={y + 4} textAnchor="end" fontSize="11" fill="#b9b7c0">
                {r.label}
              </text>
              <line
                x1={px(r.v - r.err)} y1={y} x2={px(r.v + r.err)} y2={y}
                stroke={r.color} strokeWidth="3" strokeLinecap="round" opacity="0.5"
              />
              <circle cx={px(r.v)} cy={y} r="5" fill={r.color} stroke="#0a0a0f" strokeWidth="1.5" />
            </g>
          )
        })}
      </svg>
    </div>
  )
}
