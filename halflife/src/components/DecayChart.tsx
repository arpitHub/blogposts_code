import { useMemo } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { survivingFraction, theoreticalFraction } from '../lib/decay'
import { LegendItem } from './ui'

const POINTS = 90

interface Datum {
  t: number
  parent?: number
  daughter?: number
  theory?: number
}

/**
 * Live parent-fraction vs time curve that accompanies a ParticleJar.
 * "Measured" series only exists up to the current sim time, so the curve
 * draws itself as the jar runs.
 */
export function DecayChart({
  lifetimes,
  halfLife,
  time,
  maxTime,
  showTheory = false,
  timeUnit = 's',
  height = 260,
}: {
  lifetimes: Float64Array
  halfLife: number
  time: number
  maxTime: number
  showTheory?: boolean
  timeUnit?: string
  height?: number
}) {
  const data = useMemo<Datum[]>(() => {
    const out: Datum[] = []
    for (let i = 0; i <= POINTS; i++) {
      const t = (i / POINTS) * maxTime
      const d: Datum = { t }
      if (t <= time) {
        const f = survivingFraction(lifetimes, halfLife, t)
        d.parent = f
        d.daughter = 1 - f
      }
      if (showTheory) d.theory = theoreticalFraction(halfLife, t)
      out.push(d)
    }
    return out
  }, [lifetimes, halfLife, time, maxTime, showTheory])

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1">
        <LegendItem color="#c98500" label="Parent atoms remaining" />
        <LegendItem color="#3987e5" label="Daughter atoms" />
        {showTheory && (
          <LegendItem color="#8a8894" label="e^(−λt) prediction" dashed />
        )}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={data}
          margin={{ top: 6, right: 12, bottom: 4, left: -14 }}
        >
          <CartesianGrid stroke="#1c1c26" vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={[0, maxTime]}
            tickFormatter={(v: number) => `${Math.round(v)}`}
            tick={{ fill: '#8a8894', fontSize: 11 }}
            stroke="#34343f"
            tickLine={false}
            label={{
              value: `time (${timeUnit})`,
              position: 'insideBottomRight',
              offset: -2,
              fill: '#8a8894',
              fontSize: 11,
            }}
          />
          <YAxis
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fill: '#8a8894', fontSize: 11 }}
            stroke="#34343f"
            tickLine={false}
          />
          <Tooltip
            cursor={{ stroke: '#8a8894', strokeDasharray: '3 3' }}
            content={<DecayTooltip timeUnit={timeUnit} />}
          />
          {/* half-life gridline: where the parent curve crosses 50% */}
          <ReferenceLine
            y={0.5}
            stroke="#34343f"
            strokeDasharray="4 4"
            label={{
              value: '½',
              position: 'insideLeft',
              fill: '#8a8894',
              fontSize: 11,
            }}
          />
          {showTheory && (
            <Line
              dataKey="theory"
              stroke="#8a8894"
              strokeWidth={1.5}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={false}
            />
          )}
          <Line
            dataKey="parent"
            stroke="#c98500"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            dataKey="daughter"
            stroke="#3987e5"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <ReferenceLine x={Math.min(time, maxTime)} stroke="#f4f2ec" strokeOpacity={0.35} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function DecayTooltip({
  active,
  payload,
  label,
  timeUnit,
}: {
  active?: boolean
  payload?: { dataKey?: string | number; value?: number | string }[]
  label?: number
  timeUnit: string
}) {
  if (!active || !payload?.length) return null
  const get = (k: string) => {
    const p = payload.find((x) => x.dataKey === k)
    return typeof p?.value === 'number' ? p.value : undefined
  }
  const parent = get('parent')
  const daughter = get('daughter')
  const theory = get('theory')
  return (
    <div className="rounded-md border border-line bg-panel px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-medium text-ink">
        t = {typeof label === 'number' ? label.toFixed(1) : label} {timeUnit}
      </div>
      {parent !== undefined && (
        <div className="text-ink-2">
          <span className="mr-1 inline-block h-2 w-2 rounded-full bg-amber-series" />
          parent {Math.round(parent * 100)}%
        </div>
      )}
      {daughter !== undefined && (
        <div className="text-ink-2">
          <span className="mr-1 inline-block h-2 w-2 rounded-full bg-blue-series" />
          daughter {Math.round(daughter * 100)}%
        </div>
      )}
      {theory !== undefined && (
        <div className="text-ink-3">prediction {Math.round(theory * 100)}%</div>
      )}
    </div>
  )
}
