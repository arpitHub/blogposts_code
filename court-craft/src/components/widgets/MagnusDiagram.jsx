import { useState } from 'react'
import { Segmented } from '../ui'

const MODES = {
  topspin: {
    label: 'Topspin',
    spinDir: 1,
    force: { dy: 1, label: 'Extra downward force', color: '#b94a26' },
    caption: 'The top of the ball rotates in the same direction as the airflow while the bottom fights it. Air gets deflected upward off the back of the ball — and the ball gets pushed DOWN in return. That’s why topspin dives back into the court even when you hit it high and hard.',
  },
  flat: {
    label: 'Flat (no spin)',
    spinDir: 0,
    force: null,
    caption: 'With little spin, air flows evenly around the ball — no extra push in either direction. The ball follows a plain gravity arc: fast and direct, but with the smallest margin over the net.',
  },
  slice: {
    label: 'Slice (backspin)',
    spinDir: -1,
    force: { dy: -1, label: 'Extra lift', color: '#3b6ea5' },
    caption: 'Backspin does the opposite: air is deflected downward, so the ball gets pushed UP. Sliced balls float, carry deeper than you expect, and stay low after the bounce.',
  },
}

export default function MagnusDiagram() {
  const [mode, setMode] = useState('topspin')
  const m = MODES[mode]

  return (
    <div>
      <div className="mb-4">
        <Segmented
          options={Object.entries(MODES).map(([value, v]) => ({ value, label: v.label }))}
          value={mode}
          onChange={setMode}
        />
      </div>

      <div className="grid items-center gap-6 sm:grid-cols-[280px_1fr]">
        <svg viewBox="0 0 280 232" className="mx-auto w-full max-w-[280px]" role="img" aria-label="Magnus effect diagram">
          {/* airflow streamlines (ball moving right = air moving left relative to ball) */}
          {[45, 75, 145, 175].map((y) => (
            <g key={y}>
              <path
                d={`M265 ${y} C 200 ${y}, 190 ${y + (y < 110 ? -10 : 10) * (m.spinDir)}, 130 ${y + (y < 110 ? -12 : 12) * m.spinDir} S 40 ${y + (y < 110 ? -18 : 18) * m.spinDir}, 15 ${y + (y < 110 ? -18 : 18) * m.spinDir}`}
                fill="none" stroke="#8bb899" strokeWidth="1.5" strokeDasharray="6 5"
              >
                <animate attributeName="stroke-dashoffset" from="22" to="0" dur="0.7s" repeatCount="indefinite" />
              </path>
            </g>
          ))}

          {/* ball */}
          <g transform="translate(140,110)">
            <circle r="34" fill="#dce65a" stroke="#b3bd2d" strokeWidth="2" />
            <path d="M-34,0 A40,40 0 0 1 34,0" fill="none" stroke="#fff" strokeWidth="3" />
            {/* spin arrows */}
            {m.spinDir !== 0 && (
              <g stroke="#7d311e" strokeWidth="2.5" fill="#7d311e">
                {m.spinDir === 1 ? (
                  <>
                    <path d="M -8 -44 A 45 45 0 0 1 30 -33" fill="none" />
                    <path d="M 30 -33 l -2 -9 l 9 3 z" stroke="none" />
                    <path d="M 8 44 A 45 45 0 0 1 -30 33" fill="none" />
                    <path d="M -30 33 l 2 9 l -9 -3 z" stroke="none" />
                  </>
                ) : (
                  <>
                    <path d="M 30 -33 A 45 45 0 0 0 -8 -44" fill="none" />
                    <path d="M -8 -44 l 2 -8 l -9 4 z" stroke="none" />
                    <path d="M -30 33 A 45 45 0 0 0 8 44" fill="none" />
                    <path d="M 8 44 l -2 8 l 9 -4 z" stroke="none" />
                  </>
                )}
              </g>
            )}
          </g>

          {/* velocity arrow */}
          <g stroke="#1f3f2e" strokeWidth="3" fill="#1f3f2e">
            <line x1="185" y1="110" x2="235" y2="110" />
            <path d="M235 110 l -8 -5 v 10 z" stroke="none" />
            <text x="238" y="114" fontSize="12" stroke="none" fill="#1f3f2e">ball</text>
          </g>

          {/* magnus force arrow */}
          {m.force && (
            <g stroke={m.force.color} strokeWidth="4" fill={m.force.color}>
              <line x1="140" y1={m.force.dy > 0 ? 155 : 65} x2="140" y2={m.force.dy > 0 ? 195 : 25} />
              <path
                d={m.force.dy > 0 ? 'M140 200 l -7 -10 h 14 z' : 'M140 20 l -7 10 h 14 z'}
                stroke="none"
              />
              <text
                x="140" y={m.force.dy > 0 ? 220 : 12}
                textAnchor="middle" fontSize="12" stroke="none" fontWeight="bold"
              >
                {m.force.label}
              </text>
            </g>
          )}
        </svg>

        <p className="leading-relaxed text-court-900/90">{m.caption}</p>
      </div>
    </div>
  )
}
