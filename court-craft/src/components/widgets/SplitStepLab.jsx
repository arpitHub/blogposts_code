import { useEffect, useRef, useState } from 'react'
import { TopCourtSVG, PlayerDot, Ball, MID_X, NET_Y, BASE_TOP, BASE_BOT, IN_X0, IN_X1, SRV } from './TopCourt'

// The split-step timing game: hit "Split!" just before the opponent strikes.
// Timing feedback mirrors coaching reality: initiate the hop as the opponent
// swings so you LAND right as the ball leaves — landed feet = instant push-off.

const OPP = { x: MID_X, y: BASE_TOP + 24 }
const YOU_HOME = { x: MID_X, y: BASE_BOT - 30 }
const TARGETS = [
  { x: IN_X0 + 22, y: BASE_BOT - 60 },
  { x: IN_X1 - 22, y: BASE_BOT - 60 },
]

const FLIGHT_MS = 1000

export default function SplitStepLab() {
  const [running, setRunning] = useState(false)
  const [now, setNow] = useState(0) // ms within current round
  const [round, setRound] = useState(null)
  const [result, setResult] = useState(null)
  const [score, setScore] = useState({ perfect: 0, tries: 0 })
  const splitAtRef = useRef(null)
  const rafRef = useRef()

  const newRound = () => ({
    windup: 1300 + Math.random() * 1100, // opponent's swing time — varies!
    target: TARGETS[Math.random() < 0.5 ? 0 : 1],
    start: performance.now(),
  })

  useEffect(() => {
    if (!running) return
    setRound(newRound())
    splitAtRef.current = null
    setResult(null)
  }, [running])

  useEffect(() => {
    if (!running || !round) return
    const tick = () => {
      const t = performance.now() - round.start
      setNow(t)
      const total = round.windup + FLIGHT_MS
      if (t >= total + 1600) {
        // evaluate + next round
        setRound(newRound())
        splitAtRef.current = null
        setResult(null)
        return
      }
      if (t >= total && !result) {
        // ball has arrived — grade the round
        const split = splitAtRef.current
        let r
        if (split == null) {
          r = { grade: 'none', msg: 'No split step — you were flat-footed when the ball left. Reaction started a whole beat late.' }
        } else {
          const delta = split - round.windup // + = after contact
          if (delta < -420) r = { grade: 'early', msg: `Too early (${Math.round(-delta)} ms before contact): you landed, settled, and lost the bounce. The hop's energy was wasted.` }
          else if (delta <= 80) r = { grade: 'perfect', msg: `Perfect (${delta >= 0 ? '+' : ''}${Math.round(delta)} ms): you were airborne at contact and landed reading the ball — push-off for free.` }
          else if (delta <= 400) r = { grade: 'late', msg: `Late (+${Math.round(delta)} ms): the ball was already flying while you were still going up. You'll reach fast balls a step short.` }
          else r = { grade: 'late', msg: `Way late (+${Math.round(delta)} ms): that's a reaction to the ball, not to the swing. Watch the opponent's racket, not the ball.` }
        }
        setResult(r)
        setScore((s) => ({ perfect: s.perfect + (r.grade === 'perfect' ? 1 : 0), tries: s.tries + 1 }))
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [running, round, result])

  const doSplit = () => {
    if (!running || !round || splitAtRef.current != null) return
    const t = performance.now() - round.start
    if (t <= round.windup + FLIGHT_MS) splitAtRef.current = t
  }

  // keyboard: spacebar splits
  useEffect(() => {
    const onKey = (e) => {
      if (e.code === 'Space' && running) { e.preventDefault(); doSplit() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, round])

  // ---- render state ----
  let ball = null
  let ringR = 0
  let you = { ...YOU_HOME }
  let hopScale = 1

  if (round) {
    const t = now
    if (t < round.windup) {
      ringR = 30 * (1 - t / round.windup) + 8
      ball = { x: OPP.x + 12, y: OPP.y + 6 }
    } else {
      const ft = Math.min((t - round.windup) / FLIGHT_MS, 1)
      ball = {
        x: OPP.x + (round.target.x - OPP.x) * ft,
        y: OPP.y + (round.target.y - OPP.y) * ft,
      }
      // you move toward intercept if you split
      const split = splitAtRef.current
      if (split != null) {
        const delay = Math.max(0, (split - round.windup + 200) / 1000) // late split = late start
        const moveT = Math.max(0, Math.min((ft - delay), 1))
        const quality = split - round.windup <= 80 && split - round.windup >= -420 ? 1 : 0.55
        you = {
          x: YOU_HOME.x + (round.target.x - YOU_HOME.x) * Math.min(moveT * quality * 1.4, 1),
          y: YOU_HOME.y + (round.target.y - YOU_HOME.y) * Math.min(moveT * quality * 1.4, 1),
        }
      }
    }
    // hop animation right after split
    const split = splitAtRef.current
    if (split != null && now - split < 260) {
      hopScale = 1 + 0.35 * Math.sin(((now - split) / 260) * Math.PI)
    }
  }

  const gradeColors = { perfect: 'bg-court-500 text-white', early: 'bg-clay-400 text-white', late: 'bg-clay-600 text-white', none: 'bg-court-900 text-white' }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,300px)_1fr]">
      <TopCourtSVG label="Split-step timing game" viewBox="0 0 380 660">
        {/* opponent + windup ring */}
        <PlayerDot x={OPP.x} y={OPP.y} color="#1f3f2e" label="opponent" />
        {running && ringR > 0 && (
          <circle cx={OPP.x} cy={OPP.y} r={ringR + 12} fill="none" stroke="#dce65a" strokeWidth="3" opacity="0.85" />
        )}
        {ball && <Ball x={ball.x} y={ball.y} />}
        <g transform={`translate(${you.x} ${you.y}) scale(${hopScale})`}>
          <circle r="10" fill="#cf5f38" stroke="white" strokeWidth="2.5" />
        </g>
        <text x={you.x} y={you.y + 24} textAnchor="middle" fontSize="11" fontWeight="bold" fill="white">you</text>
      </TopCourtSVG>

      <div>
        <p className="text-sm leading-relaxed text-court-800/90">
          The <strong>split step</strong> is a small hop timed to your opponent’s swing: leave
          the ground as they swing, <em>land just as they hit</em>. Landed, loaded feet react
          two steps faster than flat ones. The shrinking yellow ring is their wind-up —{' '}
          <strong>hit Split! (or spacebar) right as it closes.</strong>
        </p>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          {!running ? (
            <button
              onClick={() => setRunning(true)}
              className="rounded-xl bg-clay-500 px-6 py-3 font-semibold text-white shadow transition hover:bg-clay-600"
            >
              ▶ Start the drill
            </button>
          ) : (
            <>
              <button
                onClick={doSplit}
                className="rounded-xl bg-court-800 px-8 py-3 text-lg font-bold text-white shadow transition active:scale-95 hover:bg-court-700"
              >
                SPLIT!
              </button>
              <button
                onClick={() => setRunning(false)}
                className="rounded-xl border border-line bg-white px-4 py-3 text-sm font-medium text-court-600"
              >
                Stop
              </button>
            </>
          )}
          {score.tries > 0 && (
            <span className="text-sm text-court-600">
              Perfect: <b>{score.perfect}</b> / {score.tries}
            </span>
          )}
        </div>

        <div className="mt-4 min-h-24">
          {result && (
            <div className="cc-fade-up">
              <span className={`inline-block rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wide ${gradeColors[result.grade]}`}>
                {result.grade === 'none' ? 'no split' : result.grade}
              </span>
              <p className="mt-2 text-sm leading-relaxed text-court-800">{result.msg}</p>
            </div>
          )}
          {running && !result && (
            <p className="text-sm text-court-500">Watch the ring… the wind-up varies, just like a real opponent.</p>
          )}
        </div>
      </div>
    </div>
  )
}
