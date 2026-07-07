import { Link } from 'react-router-dom'
import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import PhaseExplorer from '../../components/widgets/PhaseExplorer'
import { OVERHEAD_PHASES, OVERHEAD_VS_SERVE } from '../../data/strokes/overhead'

export default function Overhead() {
  return (
    <div>
      <PageIntro moduleId="overhead" kicker="Stroke technique">
        <p>
          Good news: if you’ve worked through <Link to="/serve" className="font-medium text-clay-600 underline">the
          serve</Link>, you already own 80% of the overhead. The smash is a serve where your
          opponent controls the toss — which means the remaining 20% is entirely about
          footwork and timing, not a new swing.
        </p>
      </PageIntro>

      <WidgetFrame wide title="The overhead, phase by phase" hint="phases 3–5 are literally the serve">
        <PhaseExplorer phases={OVERHEAD_PHASES} />
      </WidgetFrame>

      <Prose title="Serve vs. smash — what actually changes">
        <p>
          Scrub the explorer to the trophy position and compare it with the serve page:
          identical. Here’s the honest list of what’s different:
        </p>
      </Prose>

      <section className="mx-auto max-w-3xl px-6 py-2">
        <div className="overflow-x-auto rounded-2xl border border-line bg-white shadow-sm">
          <table className="w-full min-w-130 text-sm">
            <thead>
              <tr className="border-b border-line bg-court-50/60 text-left">
                <th className="px-4 py-3 font-semibold text-court-900"></th>
                <th className="px-4 py-3 font-semibold text-court-700">Serve</th>
                <th className="px-4 py-3 font-semibold text-clay-700">Overhead</th>
              </tr>
            </thead>
            <tbody>
              {OVERHEAD_VS_SERVE.map((d) => (
                <tr key={d.dim} className="border-b border-line last:border-0 align-top">
                  <td className="px-4 py-3 font-medium text-court-900">{d.dim}</td>
                  <td className="px-4 py-3 leading-relaxed text-court-700">{d.serve}</td>
                  <td className="px-4 py-3 leading-relaxed text-court-700">{d.overhead}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Prose>
        <Aside title="The #1 overhead sin">
          Backpedaling straight backwards while staring up. It’s slow, it’s how ankles roll,
          and you arrive off balance. Turn sideways FIRST, then sidestep — you’ll cover more
          ground faster and arrive ready to swing.
        </Aside>
        <p>
          One tactical note: you don’t have to crush every overhead. A three-quarter-pace
          smash placed into the open court wins the point just as dead — and misses far
          less. Save the highlight-reel swing for balls landing inside the service line.
        </p>
      </Prose>

      <TryThis>
        <p>
          <strong>Point-catch-smash ladder:</strong> partner feeds easy lobs. Round one:
          don’t hit — turn sideways, sidestep under the ball, and CATCH it with your free
          hand above your head, slightly in front. Ten catches. Round two: same footwork,
          now let it bounce and smash after the bounce (much easier timing). Round three:
          take it out of the air. You’ll be shocked how much round one fixes round three.
        </p>
      </TryThis>

      <NextUp moduleId="overhead" />
    </div>
  )
}
