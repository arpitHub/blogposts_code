import { Link } from 'react-router-dom'
import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import PhaseExplorer from '../../components/widgets/PhaseExplorer'
import MistakeLab from '../../components/widgets/MistakeLab'
import { VOLLEY_PHASES, VOLLEY_MISTAKES } from '../../data/strokes/volley'

export default function Volley() {
  return (
    <div>
      <PageIntro moduleId="volley" kicker="Stroke technique">
        <p>
          The volley is the anti-groundstroke: no backswing, no spin brush, no time. It’s a
          catch with strings — you block the ball out of the air before it bounces and let
          the incoming pace do the work. Players who “can’t volley” are almost always trying
          to do too much.
        </p>
      </PageIntro>

      <Prose>
        <p>
          Volleys are hit with the <strong>continental grip</strong> — the same “hammer” grip
          as the serve (<Link to="/grips" className="font-medium text-clay-600 underline">Grip
          Types</Link> has the picture). One grip for both sides, because at the net there is
          no time to change. Notice how short this sequence is compared to the forehand —
          four phases, and two of them are “be ready”:
        </p>
      </Prose>

      <WidgetFrame wide title="The volley, phase by phase" hint="note how little the racket travels">
        <PhaseExplorer phases={VOLLEY_PHASES} />
      </WidgetFrame>

      <Prose title="The footwork IS the volley">
        <p>
          Watch phase 3 again: the step and the contact happen <em>together</em>. Good
          volleyers don’t reach with the arm — they move the whole body diagonally forward
          into the ball. The punch comes from the step; the arm mostly holds the racket
          still. If your volleys feel weak, add step, not swing.
        </p>
        <Aside title="Firm wrist test">
          After a volley, your racket face should still point where the ball went. If it’s
          facing the side fence, your wrist flicked. Squeeze the grip to “firm handshake” at
          contact — never floppy, never white-knuckled.
        </Aside>
      </Prose>

      <WidgetFrame wide title="Good form vs. common mistake" hint="the classic error: treating it like a groundstroke">
        <MistakeLab good={{ title: 'Compact block', phases: VOLLEY_PHASES }} mistakes={VOLLEY_MISTAKES} />
      </WidgetFrame>

      <TryThis>
        <p>
          <strong>The wall catch:</strong> stand two meters from a wall, no racket. Have a
          partner (or your own throw) send balls at you; catch each one with your hitting
          hand out in front of your face, stepping diagonally into the catch. That catching
          position — hand out front, body moving forward — IS your volley. Then repeat with
          the racket, and change nothing.
        </p>
      </TryThis>

      <NextUp moduleId="volley" />
    </div>
  )
}
