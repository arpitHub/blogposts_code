import { useMemo, useState } from 'react'
import { Section } from '../components/Section'
import { ParticleJar } from '../components/ParticleJar'
import { DecayChart } from '../components/DecayChart'
import { Btn, EqCard, SliderRow, Stat } from '../components/ui'
import { D, useDepth } from '../lib/depth'
import { makeUnitLifetimes, survivingCount } from '../lib/decay'
import { useClock } from '../lib/useClock'

const N = 200
const MAX_TIME = 60 // seconds of sim time shown on the chart

export function DecaySection() {
  const depth = useDepth()
  const [seed, setSeed] = useState(7)
  const [halfLife, setHalfLife] = useState(10)
  const lifetimes = useMemo(() => makeUnitLifetimes(N, seed), [seed])
  const { time, running, setRunning, reset } = useClock(1, MAX_TIME)

  const alive = survivingCount(lifetimes, halfLife, time)
  const lambda = Math.LN2 / halfLife

  const handleReset = () => {
    reset()
    setSeed((s) => s + 1) // fresh random draws each run
  }

  return (
    <Section id="decay" kicker="01 · The mechanism" title="Atoms are unstable — and that's a clock">
      <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div>
          <ParticleJar
            lifetimes={lifetimes}
            halfLife={halfLife}
            time={time}
            seed={seed}
          />
          <div className="mt-3 grid grid-cols-3 gap-2">
            <Stat label="Parent left" value={String(alive)} accent="amber" />
            <Stat label="Decayed" value={String(N - alive)} accent="blue" />
            <Stat label="Elapsed" value={`${time.toFixed(1)} s`} />
          </div>
        </div>

        <div className="flex flex-col gap-5">
          <p className="max-w-prose text-[15px] leading-relaxed text-ink-2">
            <D
              b={
                <>
                  Each glowing dot is an unstable atom. At some random moment it
                  flips — once, forever — into a different kind of atom (the dim
                  blue dots). It’s like a room full of coins that each flip
                  themselves exactly once: you can never say <em>which</em> coin
                  flips next, but you can say with eerie precision how long until{' '}
                  <em>half</em> of them have flipped. That reliable “half-time”
                  is the tick of a natural clock.
                </>
              }
              t={
                <>
                  Each parent nucleus has a fixed decay probability per unit
                  time, λ. Individual decays are stochastic and memoryless, but
                  the ensemble follows an exact exponential: the survival curve
                  below is the measured fraction of {N} simulated nuclei against
                  the analytic prediction. The half-life t<sub>½</sub> = ln 2 / λ
                  is when the parent fraction crosses 50%. Watch the measured
                  amber curve wobble around the dashed prediction — that’s
                  counting statistics (√N noise), exactly what a lab detector
                  sees.
                </>
              }
            />
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <Btn primary onClick={() => setRunning(!running)}>
              {running ? '❚❚ Pause' : time > 0 && time < MAX_TIME ? '▶ Resume' : '▶ Play'}
            </Btn>
            <Btn onClick={handleReset}>↺ Reset</Btn>
            <div className="min-w-[200px] flex-1">
              <SliderRow
                label={<D b="Half-life (how fast atoms flip)" t={<>Half-life t<sub>½</sub></>} />}
                min={2}
                max={20}
                step={0.5}
                value={halfLife}
                onChange={setHalfLife}
                display={`${halfLife} s`}
              />
            </div>
          </div>

          <DecayChart
            lifetimes={lifetimes}
            halfLife={halfLife}
            time={time}
            maxTime={MAX_TIME}
            showTheory={depth === 'technical'}
            timeUnit="s"
          />

          {depth === 'technical' && (
            <EqCard
              note={
                <>
                  λ is set by the slider: doubling the half-life halves the
                  decay constant. Real half-lives span from nanoseconds to
                  quintillions of years — same law, different λ.
                </>
              }
            >
              N(t) = N₀·e<sup>−λt</sup>&ensp;·&ensp;λ = ln 2 / t½ ={' '}
              {lambda.toFixed(3)} s⁻¹&ensp;·&ensp;t½ = {halfLife} s
            </EqCard>
          )}
          {depth === 'beginner' && (
            <p className="max-w-prose text-sm leading-relaxed text-ink-3">
              Drag the half-life slider — even mid-run. A short half-life burns
              through the jar quickly; a long one barely dents it. Nature stocks
              atoms with half-lives from split-seconds to billions of years,
              which is exactly what makes them useful as clocks for wildly
              different stretches of time.
            </p>
          )}
        </div>
      </div>
    </Section>
  )
}
