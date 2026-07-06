import { useState } from 'react'
import { Segmented } from '../ui'

// Top-down court. Scale: 1 m = 24 px. Court: 23.77 m × 10.97 m (doubles).
const S = 24
const CW = 10.97 * S // 263
const CL = 23.77 * S // 570
const X0 = 58
const Y0 = 35
const ALLEY = 1.37 * S // 33
const SRV = 6.4 * S // 154
const NET_Y = Y0 + CL / 2
const MID_X = X0 + CW / 2

const REGIONS = {
  serviceBox: {
    name: 'Service boxes',
    dims: '6.40 m deep × 4.11 m wide',
    text: 'Every point starts here: the serve must bounce inside the diagonally-opposite box. Serving from the right of the center mark, you aim at the receiver’s left box (the “deuce court”), and vice versa (“ad court”). Land anywhere else and it’s a fault — two faults lose the point.',
  },
  baseline: {
    name: 'Baseline',
    dims: '10.97 m wide, 11.885 m from the net',
    text: 'The back boundary for every rally ball, and your launch pad for serving. FOOT FAULT rule: while serving, neither foot may touch the baseline or the court before you strike the ball — toeing the line is the same as missing the serve.',
  },
  alley: {
    name: 'Doubles alleys ("tramlines")',
    dims: '1.37 m wide, each side',
    text: 'The strips between the singles and doubles sidelines. They only count in doubles — in singles a ball landing here is out. This is why the court has two sets of sidelines, and why singles players ignore the outer ones.',
  },
  net: {
    name: 'The net',
    dims: '0.914 m high at center, 1.07 m at the posts',
    text: 'Lower in the middle — which is why smart players aim cross-court: more net clearance AND more court to hit into. LET rule: if a serve clips the net cord and still lands in the correct box, it’s a “let” — replay that serve, no penalty, unlimited times. During a rally, a net-cord ball that dribbles over is simply in play (apologizing is traditional).',
  },
  noMans: {
    name: '“No man’s land”',
    dims: 'Between service line and baseline',
    text: 'The tactical dead zone: too far back to volley, too far forward to let balls bounce comfortably — most balls land at your feet here. Fine to pass THROUGH on your way to the net; costly to stand in. Rally from behind the baseline, volley from inside the service line.',
  },
  centerMark: {
    name: 'Center mark & center line',
    dims: 'Mark: 10 cm; line splits the service area',
    text: 'The small tick on the baseline splits serving territory: on deuce points you serve from the right of it, on ad points from the left. The long center line divides the two service boxes — a serve landing on any line it’s aimed at, including this one, is IN. Line = in, always.',
  },
}

export default function CourtExplorer() {
  const [selected, setSelected] = useState('serviceBox')
  const [mode, setMode] = useState('doubles')
  const [showDims, setShowDims] = useState(true)
  const r = REGIONS[selected]

  const singles = mode === 'singles'
  const inX0 = X0 + ALLEY
  const inX1 = X0 + CW - ALLEY

  const hl = (id) => (selected === id ? { fill: '#cf5f38', opacity: 0.45 } : { fill: 'transparent', opacity: 1 })
  const hotspot = (id) => ({
    onClick: () => setSelected(id),
    className: 'cursor-pointer transition-opacity hover:opacity-80',
    style: { pointerEvents: 'all' },
  })

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,330px)_1fr]">
      <svg viewBox="0 0 380 660" className="mx-auto w-full max-w-[330px]" role="img" aria-label="Interactive tennis court diagram">
        {/* surround */}
        <rect x="0" y="0" width="380" height="660" rx="12" fill="#b45838" />
        {/* court */}
        <rect x={X0} y={Y0} width={CW} height={CL} fill="#3f7a54" />
        {/* alleys dimmed in singles */}
        {singles && (
          <>
            <rect x={X0} y={Y0} width={ALLEY} height={CL} fill="#1a3427" opacity="0.35" />
            <rect x={inX1} y={Y0} width={ALLEY} height={CL} fill="#1a3427" opacity="0.35" />
          </>
        )}

        {/* lines */}
        <g stroke="#faf7f2" strokeWidth="3" fill="none">
          <rect x={X0} y={Y0} width={CW} height={CL} />
          <line x1={inX0} y1={Y0} x2={inX0} y2={Y0 + CL} />
          <line x1={inX1} y1={Y0} x2={inX1} y2={Y0 + CL} />
          <line x1={inX0} y1={NET_Y - SRV} x2={inX1} y2={NET_Y - SRV} />
          <line x1={inX0} y1={NET_Y + SRV} x2={inX1} y2={NET_Y + SRV} />
          <line x1={MID_X} y1={NET_Y - SRV} x2={MID_X} y2={NET_Y + SRV} />
          {/* center marks */}
          <line x1={MID_X} y1={Y0} x2={MID_X} y2={Y0 + 10} />
          <line x1={MID_X} y1={Y0 + CL} x2={MID_X} y2={Y0 + CL - 10} />
        </g>

        {/* net */}
        <line x1={X0 - 18} y1={NET_Y} x2={X0 + CW + 18} y2={NET_Y} stroke="#0d1d15" strokeWidth="7" strokeLinecap="round" />
        <circle cx={X0 - 18} cy={NET_Y} r="5" fill="#0d1d15" />
        <circle cx={X0 + CW + 18} cy={NET_Y} r="5" fill="#0d1d15" />

        {/* dimension labels */}
        {showDims && (
          <g fontSize="11" fill="#faf7f2">
            <text x={MID_X} y={Y0 - 12} textAnchor="middle">10.97 m (doubles) / 8.23 m (singles)</text>
            <g transform={`translate(${X0 + CW + 34} ${Y0 + CL / 2}) rotate(90)`}>
              <text textAnchor="middle">23.77 m baseline to baseline</text>
            </g>
            <g transform={`translate(${X0 - 28} ${NET_Y + SRV / 2}) rotate(-90)`}>
              <text textAnchor="middle" fontSize="10">6.40 m</text>
            </g>
            <text x={MID_X} y={NET_Y - 8} textAnchor="middle" fontSize="10" opacity="0.9">net: 0.914 m center</text>
          </g>
        )}

        {/* highlight overlays */}
        <rect x={inX0} y={NET_Y - SRV} width={(inX1 - inX0)} height={SRV * 2} {...hl('serviceBox')} />
        <rect x={X0 - 4} y={Y0 - 5} width={CW + 8} height={10} {...hl('baseline')} />
        <rect x={X0 - 4} y={Y0 + CL - 5} width={CW + 8} height={10} {...hl('baseline')} />
        <rect x={X0} y={Y0} width={ALLEY} height={CL} {...hl('alley')} />
        <rect x={inX1} y={Y0} width={ALLEY} height={CL} {...hl('alley')} />
        <rect x={X0 - 20} y={NET_Y - 7} width={CW + 40} height={14} {...hl('net')} />
        <rect x={inX0} y={Y0} width={inX1 - inX0} height={CL / 2 - SRV} {...hl('noMans')} />
        <rect x={inX0} y={NET_Y + SRV} width={inX1 - inX0} height={CL / 2 - SRV} {...hl('noMans')} />
        <rect x={MID_X - 6} y={NET_Y - SRV} width={12} height={SRV * 2} {...hl('centerMark')} />
        <rect x={MID_X - 8} y={Y0 - 2} width={16} height={14} {...hl('centerMark')} />
        <rect x={MID_X - 8} y={Y0 + CL - 12} width={16} height={14} {...hl('centerMark')} />

        {/* invisible click targets (drawn last so they always catch clicks) */}
        <g fill="transparent">
          <rect x={X0 - 6} y={Y0 - 8} width={CW + 12} height={14} {...hotspot('baseline')} />
          <rect x={X0 - 6} y={Y0 + CL - 6} width={CW + 12} height={14} {...hotspot('baseline')} />
          <rect x={X0} y={Y0 + 8} width={ALLEY} height={CL - 16} {...hotspot('alley')} />
          <rect x={inX1} y={Y0 + 8} width={ALLEY} height={CL - 16} {...hotspot('alley')} />
          <rect x={X0 - 24} y={NET_Y - 9} width={CW + 48} height={18} {...hotspot('net')} />
          <rect x={inX0} y={Y0 + 6} width={inX1 - inX0} height={CL / 2 - SRV - 14} {...hotspot('noMans')} />
          <rect x={inX0} y={NET_Y + SRV + 8} width={inX1 - inX0} height={CL / 2 - SRV - 14} {...hotspot('noMans')} />
          <rect x={MID_X - 7} y={NET_Y - SRV + 4} width={14} height={SRV * 2 - 8} {...hotspot('centerMark')} />
          <rect x={inX0 + 14} y={NET_Y - SRV + 4} width={MID_X - inX0 - 21} height={SRV * 2 - 8} {...hotspot('serviceBox')} />
          <rect x={MID_X + 7} y={NET_Y - SRV + 4} width={inX1 - MID_X - 21} height={SRV * 2 - 8} {...hotspot('serviceBox')} />
        </g>
      </svg>

      <div>
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <Segmented
            options={[{ value: 'doubles', label: 'Doubles court' }, { value: 'singles', label: 'Singles court' }]}
            value={mode} onChange={setMode} size="sm"
          />
          <label className="flex items-center gap-1.5 text-xs text-court-600">
            <input type="checkbox" checked={showDims} onChange={(e) => setShowDims(e.target.checked)} className="accent-clay-500" />
            show dimensions
          </label>
        </div>

        <div className="mb-3 flex flex-wrap gap-1.5">
          {Object.entries(REGIONS).map(([id, reg]) => (
            <button
              key={id}
              onClick={() => setSelected(id)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                selected === id ? 'bg-clay-500 text-white' : 'border border-line bg-court-50 text-court-600 hover:border-clay-300'
              }`}
            >
              {reg.name}
            </button>
          ))}
        </div>

        <div className="rounded-xl border border-line bg-court-50/50 px-4 py-3">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h4 className="font-display text-lg font-bold text-court-950">{r.name}</h4>
            <span className="font-mono text-xs text-court-500">{r.dims}</span>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-court-900/90">{r.text}</p>
        </div>

        <p className="mt-3 text-xs text-court-500">
          Tap regions on the court, or use the chips. {singles ? 'Darkened alleys are out of play in singles.' : ''}
        </p>
      </div>
    </div>
  )
}
