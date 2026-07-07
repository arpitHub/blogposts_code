// Shared top-down court geometry + base SVG for the movement/tactics widgets.
// Scale: 1 m = 24 px. Vertical court: "you" at the bottom, opponent at the top.

export const S = 24
export const CW = 10.97 * S
export const CL = 23.77 * S
export const X0 = 58
export const Y0 = 35
export const ALLEY = 1.37 * S
export const SRV = 6.4 * S
export const NET_Y = Y0 + CL / 2
export const MID_X = X0 + CW / 2
export const IN_X0 = X0 + ALLEY
export const IN_X1 = X0 + CW - ALLEY
export const BASE_TOP = Y0
export const BASE_BOT = Y0 + CL
export const VIEW_W = 380
export const VIEW_H = 660

export function TopCourtSVG({ children, label, dimAlleys = true, className = '', viewBox }) {
  return (
    <svg
      viewBox={viewBox ?? `0 0 ${VIEW_W} ${VIEW_H}`}
      className={`w-full select-none ${className}`}
      role="img"
      aria-label={label}
    >
      <rect x="0" y="0" width={VIEW_W} height={VIEW_H} rx="12" fill="#b45838" />
      <rect x={X0} y={Y0} width={CW} height={CL} fill="#3f7a54" />
      {dimAlleys && (
        <>
          <rect x={X0} y={Y0} width={ALLEY} height={CL} fill="#1a3427" opacity="0.25" />
          <rect x={IN_X1} y={Y0} width={ALLEY} height={CL} fill="#1a3427" opacity="0.25" />
        </>
      )}
      <g stroke="#faf7f2" strokeWidth="2.5" fill="none" opacity="0.95">
        <rect x={X0} y={Y0} width={CW} height={CL} />
        <line x1={IN_X0} y1={Y0} x2={IN_X0} y2={BASE_BOT} />
        <line x1={IN_X1} y1={Y0} x2={IN_X1} y2={BASE_BOT} />
        <line x1={IN_X0} y1={NET_Y - SRV} x2={IN_X1} y2={NET_Y - SRV} />
        <line x1={IN_X0} y1={NET_Y + SRV} x2={IN_X1} y2={NET_Y + SRV} />
        <line x1={MID_X} y1={NET_Y - SRV} x2={MID_X} y2={NET_Y + SRV} />
        <line x1={MID_X} y1={BASE_TOP} x2={MID_X} y2={BASE_TOP + 9} />
        <line x1={MID_X} y1={BASE_BOT} x2={MID_X} y2={BASE_BOT - 9} />
      </g>
      <line x1={X0 - 16} y1={NET_Y} x2={X0 + CW + 16} y2={NET_Y} stroke="#0d1d15" strokeWidth="6" strokeLinecap="round" />
      {children}
    </svg>
  )
}

export function PlayerDot({ x, y, color = '#cf5f38', label, r = 10 }) {
  return (
    <g>
      <circle cx={x} cy={y} r={r} fill={color} stroke="white" strokeWidth="2.5" />
      {label && (
        <text x={x} y={y + r + 13} textAnchor="middle" fontSize="11" fontWeight="bold" fill="white">
          {label}
        </text>
      )}
    </g>
  )
}

export function Ball({ x, y, r = 6 }) {
  return <circle cx={x} cy={y} r={r} fill="#dce65a" stroke="#b3bd2d" strokeWidth="1.5" />
}
