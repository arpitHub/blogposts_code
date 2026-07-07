import { useState } from 'react'
import { SliderRow } from '../ui'

const clamp01 = (v) => Math.min(1, Math.max(0, v))

const PRESETS = [
  { name: 'Total beginner', weight: 275, head: 105, tension: 21.5 },
  { name: 'Improving club player', weight: 295, head: 100, tension: 23 },
  { name: 'Advanced baseliner', weight: 305, head: 98, tension: 24.5 },
  { name: 'Classic control frame', weight: 320, head: 95, tension: 25.5 },
]

function Meter({ label, value, hint }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-court-800">{label}</span>
        <span className="font-mono text-xs text-court-500">{Math.round(value * 100)}</span>
      </div>
      <div className="mt-1 h-3 overflow-hidden rounded-full bg-court-100">
        <div
          className="h-full rounded-full bg-gradient-to-r from-court-400 to-clay-500 transition-all duration-300"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <p className="mt-1 text-[11px] leading-snug text-court-500">{hint}</p>
    </div>
  )
}

export default function RacketTuner() {
  const [weight, setWeight] = useState(295) // grams, strung
  const [head, setHead] = useState(100) // sq inches
  const [tension, setTension] = useState(23) // kg

  const wN = clamp01((weight - 260) / (340 - 260))
  const hN = clamp01((head - 85) / (110 - 85))
  const tN = clamp01((tension - 20) / (27 - 20))

  const power = clamp01(0.42 * hN + 0.33 * wN + 0.25 * (1 - tN))
  const control = clamp01(0.45 * (1 - hN) + 0.2 * wN + 0.35 * tN)
  const comfort = clamp01(0.4 * (1 - tN) + 0.32 * wN + 0.28 * hN)
  const maneuver = clamp01(0.75 * (1 - wN) + 0.25 * (1 - hN))

  let archetype
  if (head >= 103 && weight <= 285) archetype = { name: 'Game-improvement frame', text: 'Big, light, forgiving — the racket does the work while you learn the swing. Exactly right for new players; you may crave more control in a year or two.' }
  else if (weight >= 315 && head <= 97) archetype = { name: 'Player’s control frame', text: 'Heavy and precise. Rewards full, fast, well-timed swings — and punishes everything else. Earn it before you buy it.' }
  else if (power > 0.62 && control < 0.45) archetype = { name: 'Power-leaning setup', text: 'Free depth and easy pace, but the ball may fly on you under pressure. Adding tension or head weight would rein it in.' }
  else if (control > 0.6 && power < 0.45) archetype = { name: 'Control-leaning setup', text: 'The ball goes where you aim — if you supply the power. Best with fast, confident swings and good conditioning.' }
  else archetype = { name: 'Modern all-rounder ("tweener")', text: 'The balanced middle where most club rackets live: enough power to hold the baseline, enough control to trust on big points.' }

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-court-500">Try a setup:</span>
        {PRESETS.map((p) => (
          <button
            key={p.name}
            onClick={() => { setWeight(p.weight); setHead(p.head); setTension(p.tension) }}
            className="rounded-full border border-line bg-court-50 px-3 py-1 text-xs font-medium text-court-700 transition hover:border-clay-400 hover:text-clay-600"
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        <div className="space-y-5">
          <SliderRow
            label="Racket weight (strung)" value={weight} onChange={setWeight}
            min={260} max={340} step={5}
            format={(v) => `${v} g`}
            leftHint="whippy, easy on the arm to swing" rightHint="stable, plows through the ball"
          />
          <SliderRow
            label="Head size" value={head} onChange={setHead}
            min={85} max={110} step={1}
            format={(v) => `${v} in²`}
            leftHint="surgical, small sweet spot" rightHint="forgiving, trampoline effect"
          />
          <SliderRow
            label="String tension" value={tension} onChange={setTension}
            min={20} max={27} step={0.5}
            format={(v) => `${v} kg`}
            leftHint="looser = power + comfort" rightHint="tighter = control + feel"
          />
        </div>

        <div className="space-y-3">
          <Meter label="Power" value={power} hint="How much free depth and pace the racket adds to a medium swing." />
          <Meter label="Control" value={control} hint="How predictably the ball goes where you aimed on fast swings." />
          <Meter label="Comfort" value={comfort} hint="Shock and vibration reaching your arm — low comfort + lots of play = elbow trouble." />
          <Meter label="Maneuverability" value={maneuver} hint="How fast you can whip it into position — matters at the net and on returns." />
        </div>
      </div>

      <div className="mt-5 rounded-xl border border-line bg-court-50/60 px-4 py-3">
        <span className="font-display text-base font-bold text-court-950">{archetype.name}.</span>{' '}
        <span className="text-sm leading-relaxed text-court-800">{archetype.text}</span>
      </div>
    </div>
  )
}
