import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import PositionLab from '../../components/widgets/PositionLab'

export default function Positioning() {
  return (
    <div>
      <PageIntro moduleId="positioning" kicker="Movement & tactics">
        <p>
          Before every ball is struck, you make a bet with your feet: “the next shot will
          come somewhere I can reach.” Good positioning is just making better bets. Drag
          yourself around the court below — each scenario grades your spot in real time.
        </p>
      </PageIntro>

      <WidgetFrame wide title="The positioning simulator" hint="drag the orange dot — switch scenarios">
        <PositionLab />
      </WidgetFrame>

      <Prose title="The three bands of the court">
        <p>
          Singles positioning largely reduces to three horizontal bands. The{' '}
          <strong>baseline band</strong> (a step behind the line) is home — you can handle
          deep balls and short ones from there. The <strong>net band</strong> (inside the
          service line) is where you finish points with volleys. Between them lies{' '}
          <strong>no man’s land</strong> — fine to travel through, terrible to live in,
          because good deep balls land exactly at your feet there.
        </p>
        <Aside title="Doubles is a different sport (positionally)">
          In doubles, one up + one back is the starting formation, and the net player is not
          a spectator: their job is to drift and poach. The pair that controls the net wins
          the point roughly twice as often — which is why every doubles tactic is ultimately
          about getting both players forward first.
        </Aside>
        <p>
          And a note on <strong>court geography by shot type</strong>: after a short, weak
          ball you hit, move up; after a deep defensive lob, drop back. Your position should
          breathe with the quality of your own shots — static feet are the real mistake, more
          than any single wrong spot.
        </p>
      </Prose>

      <TryThis>
        <p>
          <strong>The coin drill:</strong> place a coin one racket-length behind the center
          mark. Rally with a partner, and after EVERY shot you hit, touch the coin area with
          your foot before their ball comes back. It will feel frantic for ten minutes —
          then your legs learn that recovery isn’t optional, and rallies suddenly feel
          slower.
        </p>
      </TryThis>

      <NextUp moduleId="positioning" />
    </div>
  )
}
