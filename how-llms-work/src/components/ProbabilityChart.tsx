import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { useDepth } from '../context/DepthContext'
import { FOCUS, INK } from '../lib/palette'

const HIGHLIGHT = '#cde2fb' // lightest step of the same blue ramp — sampled bar

export interface ChartDatum {
  token: string
  logit: number
  p: number
  pct: number
}

// Loaded via React.lazy so recharts stays out of the main bundle; keep this
// module's imports limited to what the chart itself needs.
export default function ProbabilityChart({ data, sampled }: { data: ChartDatum[]; sampled: string | null }) {
  const { depth } = useDepth()
  return (
    <ResponsiveContainer>
      <BarChart data={data} margin={{ top: 24, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke={INK.grid} />
        <XAxis
          dataKey="token"
          tickLine={false}
          axisLine={{ stroke: INK.axis }}
          tick={{ fill: INK.secondary, fontSize: 12, fontFamily: 'ui-monospace, monospace' }}
          interval={0}
        />
        <YAxis
          tickLine={false}
          axisLine={false}
          tick={{ fill: INK.muted, fontSize: 11 }}
          tickFormatter={(v: number) => `${v.toFixed(0)}%`}
          width={40}
        />
        <Tooltip
          cursor={{ fill: 'rgba(255,255,255,0.04)' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[0].payload as ChartDatum
            return (
              <div className="rounded-md border border-hairline bg-page px-3 py-2 font-mono text-xs text-ink shadow-lg">
                <p className="font-semibold">“{d.token}”</p>
                <p className="text-ink-2">p = {(d.p * 100).toFixed(1)}%</p>
                {depth === 'technical' && <p className="text-ink-3">logit = {d.logit.toFixed(1)}</p>}
              </div>
            )
          }}
        />
        <Bar dataKey="pct" radius={[4, 4, 0, 0]} isAnimationActive animationDuration={350}>
          {data.map((d) => (
            <Cell
              key={d.token}
              fill={d.token === sampled ? HIGHLIGHT : FOCUS}
              stroke={d.token === sampled ? '#fff' : 'none'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
