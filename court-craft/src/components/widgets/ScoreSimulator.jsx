import { useReducer, useRef, useEffect } from 'react'

const POINT_NAMES = ['0', '15', '30', '40']

const initial = {
  p: [0, 0], // points in current game (or tiebreak points)
  g: [0, 0], // games in current set
  s: [0, 0], // sets won
  prevSets: [], // finished set scores, e.g. ["6–4"]
  tiebreak: false,
  finished: false,
  log: [{ text: 'Fresh match. Every point starts a story — award points below and watch the scoring system do its thing.', tone: 'info' }],
}

function pointName(state, who) {
  if (state.tiebreak) return String(state.p[who])
  const [a, b] = state.p
  if (a >= 3 && b >= 3) {
    if (a === b) return 'Deuce'
    return (who === 0 ? a > b : b > a) ? 'Ad' : '40'
  }
  return POINT_NAMES[state.p[who]]
}

function reducer(state, action) {
  if (action.type === 'reset') return { ...initial, log: [{ text: 'New match — love all. (Yes, “love” means zero. See the aside below for why.)', tone: 'info' }] }
  if (state.finished || action.type !== 'point') return state

  const w = action.who // 0 = you, 1 = opponent
  const o = 1 - w
  const who = w === 0 ? 'You' : 'Opponent'
  let { p, g, s, prevSets, tiebreak } = state
  p = [...p]; g = [...g]; s = [...s]; prevSets = [...prevSets]
  const log = []
  let finished = false

  p[w] += 1

  const winGame = (viaTiebreak = false) => {
    g[w] += 1
    p = [0, 0]
    const setWon = viaTiebreak || (g[w] >= 6 && g[w] - g[o] >= 2)
    if (viaTiebreak) {
      log.push({ text: `${who} win${w ? 's' : ''} the tiebreak — that takes the set ${g[w]}–${g[o]}.`, tone: 'big' })
    } else {
      log.push({ text: `Game, ${who.toLowerCase()}. Games are now ${g[0]}–${g[1]}.`, tone: 'win' })
    }
    tiebreak = false
    if (setWon) {
      if (!viaTiebreak) log.push({ text: `That's ${g[w]} games with a two-game cushion — ${who.toLowerCase()} take${w ? 's' : ''} the set ${g[w]}–${g[o]}.`, tone: 'big' })
      s[w] += 1
      prevSets.push(`${g[0]}–${g[1]}`)
      g = [0, 0]
      if (s[w] === 2) {
        log.push({ text: `🏆 ${who} win${w ? 's' : ''} the match, two sets to ${s[o]}. Best-of-three: first to two sets.`, tone: 'big' })
        finished = true
      } else {
        log.push({ text: `Sets: ${s[0]}–${s[1]}. New set starts at 0–0 — earlier games don't carry over.`, tone: 'info' })
      }
    } else if (g[0] === 6 && g[1] === 6) {
      tiebreak = true
      log.push({ text: '6–6! Tiebreak time: scoring switches to plain numbers. First to 7, win by 2. Serve changes every two points.', tone: 'big' })
    }
  }

  if (tiebreak) {
    log.push({ text: `Tiebreak point to ${who.toLowerCase()}: ${p[w]}–${p[o]}.`, tone: 'point' })
    if (p[w] >= 7 && p[w] - p[o] >= 2) {
      winGame(true)
    } else if (p[w] === 6 && p[o] === 6) {
      log.push({ text: '6–6 in the tiebreak — now it\'s sudden pressure: keep playing until someone leads by two.', tone: 'info' })
    }
  } else {
    const a = p[w]; const b = p[o]
    if (a >= 3 && b >= 3) {
      if (a === b) {
        log.push({ text: 'Deuce. Both players have at least 40 — from here you must win TWO points in a row to take the game.', tone: 'point' })
      } else if (a - b === 1) {
        log.push({ text: `Advantage ${who.toLowerCase()}. Win the next point and the game is ${w ? 'theirs' : 'yours'}; lose it and we're back to deuce.`, tone: 'point' })
      } else {
        winGame()
      }
    } else if (a === 4) {
      winGame()
    } else {
      const names = { 1: '15', 2: '30', 3: '40' }
      log.push({
        text: `Point ${who.toLowerCase() === 'you' ? 'to you' : 'to your opponent'}: ${pointNameRaw(p, 0)}–${pointNameRaw(p, 1)}.` +
          (names[a] === '40' ? ` One more point wins ${w ? 'them' : 'you'} the game — unless it reaches deuce.` : ''),
        tone: 'point',
      })
    }
  }

  return { p, g, s, prevSets, tiebreak, finished, log: [...state.log, ...log].slice(-8) }
}

function pointNameRaw(p, who) {
  const a = p[who]; const b = p[1 - who]
  if (a >= 3 && b >= 3) return a === b ? '40' : a > b ? 'Ad' : '40'
  return POINT_NAMES[a] ?? '40'
}

function Row({ label, sets, games, points, tiebreak, serving }) {
  return (
    <div className="grid grid-cols-[1fr_repeat(3,52px)] items-center gap-1 sm:grid-cols-[1fr_repeat(3,64px)]">
      <div className="flex items-center gap-2 truncate px-2 font-semibold text-court-50">
        {label}
      </div>
      <div className="rounded-md bg-court-800 py-2 text-center font-mono text-lg text-court-200">{sets}</div>
      <div className="rounded-md bg-court-800 py-2 text-center font-mono text-lg text-court-200">{games}</div>
      <div className={`rounded-md py-2 text-center font-mono text-lg font-bold ${tiebreak ? 'bg-clay-500 text-white' : 'bg-chalk text-court-900'}`}>
        {points}
      </div>
    </div>
  )
}

export default function ScoreSimulator() {
  const [state, dispatch] = useReducer(reducer, initial)
  const logRef = useRef(null)

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' })
  }, [state.log])

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,380px)_1fr]">
      <div>
        <div className="rounded-2xl bg-court-900 p-4 shadow-inner">
          <div className="mb-2 grid grid-cols-[1fr_repeat(3,52px)] gap-1 px-2 text-[10px] font-bold uppercase tracking-widest text-court-400 sm:grid-cols-[1fr_repeat(3,64px)]">
            <span></span><span className="text-center">Sets</span><span className="text-center">Games</span><span className="text-center">{state.tiebreak ? 'TB' : 'Points'}</span>
          </div>
          <div className="space-y-1.5">
            <Row label="You" sets={state.s[0]} games={state.g[0]} points={pointName(state, 0)} tiebreak={state.tiebreak} />
            <Row label="Opponent" sets={state.s[1]} games={state.g[1]} points={pointName(state, 1)} tiebreak={state.tiebreak} />
          </div>
          {state.prevSets.length > 0 && (
            <div className="mt-2 px-2 text-xs text-court-300">
              Finished sets: {state.prevSets.join(', ')}
            </div>
          )}
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2">
          <button
            onClick={() => dispatch({ type: 'point', who: 0 })}
            disabled={state.finished}
            className="rounded-xl bg-clay-500 px-4 py-3 font-semibold text-white shadow transition hover:bg-clay-600 disabled:opacity-40"
          >
            Point to you
          </button>
          <button
            onClick={() => dispatch({ type: 'point', who: 1 })}
            disabled={state.finished}
            className="rounded-xl bg-court-700 px-4 py-3 font-semibold text-white shadow transition hover:bg-court-600 disabled:opacity-40"
          >
            Point to opponent
          </button>
        </div>
        <button
          onClick={() => dispatch({ type: 'reset' })}
          className="mt-2 w-full rounded-xl border border-line bg-white px-4 py-2 text-sm font-medium text-court-600 transition hover:border-clay-300"
        >
          ↺ Reset match
        </button>
      </div>

      <div>
        <div className="mb-1.5 text-xs font-bold uppercase tracking-wide text-court-500">Umpire's commentary</div>
        <div ref={logRef} className="h-72 space-y-2 overflow-y-auto rounded-xl border border-line bg-court-50/50 p-3">
          {state.log.map((l, i) => (
            <div
              key={i}
              className={`rounded-lg px-3 py-2 text-sm leading-relaxed ${
                l.tone === 'big'
                  ? 'bg-clay-100 font-medium text-clay-900'
                  : l.tone === 'win'
                    ? 'bg-court-100 text-court-900'
                    : l.tone === 'info'
                      ? 'bg-white text-court-700'
                      : 'bg-white text-court-800'
              }`}
            >
              {l.text}
            </div>
          ))}
        </div>
        <p className="mt-2 text-xs text-court-500">
          Tip: to see a tiebreak, trade games back and forth until 6–6. To see deuce, get both players to 40.
        </p>
      </div>
    </div>
  )
}
