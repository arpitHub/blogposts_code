import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import CourtExplorer from '../../components/widgets/CourtExplorer'

const ETIQUETTE = [
  ['Calling lines', 'You call the lines on YOUR side, honestly, out loud, immediately. If you’re not sure a ball was out — it was in. That’s the code.'],
  ['Between points', 'The server should see you ready before serving; the receiver should play at the server’s reasonable pace. Neither rushes the other.'],
  ['Stray balls', 'Ball rolls onto your court mid-point? Stop play, call a let, replay the point. Return neighbors’ balls at a break, not mid-rally.'],
  ['Crossing courts', 'Never walk behind a court mid-point. Wait for their point to finish, then cross quickly along the back.'],
  ['Calling the score', 'The server announces the score before each point (“30–15”). It prevents 90% of scoring arguments before they exist.'],
  ['Spin for serve', 'Racket spin (rough/smooth or logo up/down) decides who serves first. Winner picks serve, return, side — or makes the other player choose first.'],
]

export default function Court() {
  return (
    <div>
      <PageIntro moduleId="court" kicker="Fundamentals">
        <p>
          The court looks like a lot of lines until you realize each one answers exactly one
          question: <em>where can this ball land?</em> Tap around the diagram — every region
          has a job, a dimension, and usually a rule hiding in it.
        </p>
      </PageIntro>

      <WidgetFrame wide title="The interactive court" hint="tap any region — toggle singles vs doubles">
        <CourtExplorer />
      </WidgetFrame>

      <Prose title="The rules people actually trip over">
        <p>
          You don’t need the full ITF rulebook to play. You need these:
        </p>
        <ul className="list-disc space-y-2 pl-5">
          <li>
            <strong>Serving:</strong> two attempts per point, diagonal into the service box,
            feet behind the baseline until contact (else it’s a <em>foot fault</em>). Serve
            clips the net but lands in? <em>Let</em> — replay it, free of charge.
          </li>
          <li>
            <strong>Line = in.</strong> Any part of the ball touching any part of the line
            it’s aimed at counts. Even the outermost molecule of fuzz.
          </li>
          <li>
            <strong>One bounce.</strong> The ball may bounce once on your side (zero is fine
            — that’s a volley). Twice and the point is over, even if you get it back.
          </li>
          <li>
            <strong>Don’t touch.</strong> You lose the point if the ball touches you or your
            clothes, or if you (or your racket) touch the net while the ball is in play.
          </li>
          <li>
            <strong>Around the post is legal.</strong> A ball curving around the net post —
            even below net height — is a legitimate (and spectacular) shot.
          </li>
        </ul>
        <Aside title="Alternating sides">
          Players swap ends after every odd-numbered game (1st, 3rd, 5th…) so nobody gets
          stuck with the sun or wind. In a tiebreak, you swap every six points.
        </Aside>
      </Prose>

      <Prose title="On-court etiquette (the unwritten rulebook)">
        <div className="grid gap-3 sm:grid-cols-2">
          {ETIQUETTE.map(([title, text]) => (
            <div key={title} className="rounded-xl border border-line bg-white px-4 py-3 shadow-sm">
              <div className="text-sm font-bold text-court-900">{title}</div>
              <p className="mt-1 text-sm leading-relaxed text-court-700">{text}</p>
            </div>
          ))}
        </div>
      </Prose>

      <TryThis>
        <p>
          <strong>Court-walk ritual:</strong> next time you’re on a real court, walk it once
          with this page in mind — stand on the center mark, pace out the 6.4 m from net to
          service line, stand in no man’s land and feel how awkward it is. Courts stop being
          abstract diagrams the day your feet have measured one.
        </p>
      </TryThis>

      <NextUp moduleId="court" overrideId="grips" />
    </div>
  )
}
