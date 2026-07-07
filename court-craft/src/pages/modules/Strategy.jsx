import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import PatternPlayer from '../../components/widgets/PatternPlayer'

const DECISION_ROWS = [
  ['Their ball lands deep, you’re behind the baseline', 'RALLY', 'High, heavy, cross-court. Reset the point. Attacking deep balls is how you donate errors.'],
  ['Their ball lands short (inside the service line-ish)', 'ATTACK', 'Step in, take it early, hit to a corner or approach. Short balls are earned — cash them in.'],
  ['You’re stretched wide, off balance', 'DEFEND', 'Slice high and deep down the middle, or lob. Buy time; the point restarts when you’re back in position.'],
  ['They’re at the net, you have time', 'PASS OR LOB', 'Dip a pass cross-court at their feet, or lob the backhand shoulder. Make them volley UP.'],
  ['Second serve, big point', 'SPIN, NOT SOFT', 'A slower flat serve is a sitting duck; a kicking spin serve at 80% is still hard to attack.'],
]

export default function Strategy() {
  return (
    <div>
      <PageIntro moduleId="strategy" kicker="Movement & tactics">
        <p>
          Rallies feel chaotic until you learn that most points follow a handful of scripts.
          Pros don’t invent tennis point by point — they run patterns and wait for triggers.
          Step through the four foundational ones below, shot by shot.
        </p>
      </PageIntro>

      <WidgetFrame wide title="Patterns of play" hint="step through each shot — yellow trails number the sequence">
        <PatternPlayer />
      </WidgetFrame>

      <Prose title="Attack, rally, or defend? Read the incoming ball">
        <p>
          Shot selection isn’t about creativity — it’s a lookup table keyed on{' '}
          <strong>where their ball lands and how much balance you have</strong>. The classic
          traffic-light version:
        </p>
      </Prose>

      <section className="mx-auto max-w-3xl px-6 py-2">
        <div className="overflow-x-auto rounded-2xl border border-line bg-white shadow-sm">
          <table className="w-full min-w-130 text-sm">
            <thead>
              <tr className="border-b border-line bg-court-50/60 text-left">
                <th className="px-4 py-3 font-semibold text-court-900">Situation</th>
                <th className="px-4 py-3 font-semibold text-court-900">Mode</th>
                <th className="px-4 py-3 font-semibold text-court-900">Play</th>
              </tr>
            </thead>
            <tbody>
              {DECISION_ROWS.map(([sit, mode, play]) => (
                <tr key={sit} className="border-b border-line align-top last:border-0">
                  <td className="px-4 py-3 leading-relaxed text-court-700">{sit}</td>
                  <td className="px-4 py-3">
                    <span className={`whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-bold ${
                      mode === 'ATTACK' ? 'bg-clay-100 text-clay-700'
                        : mode === 'RALLY' ? 'bg-court-100 text-court-700'
                          : 'bg-line text-court-600'
                    }`}>{mode}</span>
                  </td>
                  <td className="px-4 py-3 leading-relaxed text-court-700">{play}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Prose title="Reading opponents (the honest version)">
        <p>
          Scouting at club level isn’t film study — it’s three questions in the warm-up:{' '}
          <strong>Which side is weaker?</strong> (Usually the backhand; hit there on big
          points.) <strong>Do they move well?</strong> (If not, drop shots and angles beat
          power.) <strong>What do they do under pressure?</strong> (Push? Then come to the
          net. Blast? Then give them nothing to blast — slow, deep, spinny.)
        </p>
        <Aside title="The 70% rule">
          Most club matches are lost, not won: the player who makes fewer unforced errors
          wins about 70% of the time. Every pattern on this page is really an error-farming
          machine — you’re not hitting winners, you’re making THEIR next shot harder than
          yours.
        </Aside>
      </Prose>

      <TryThis>
        <p>
          <strong>One-pattern match:</strong> play a practice set where every service point
          you win must start with the Serve + 1 pattern (wide serve, next ball to the open
          court). You’ll lose some points executing it badly. Doesn’t matter. By the third
          game you’ll feel the difference between “hitting shots” and “running a play” —
          and that feeling is match strategy.
        </p>
      </TryThis>

      <NextUp moduleId="strategy" />
    </div>
  )
}
