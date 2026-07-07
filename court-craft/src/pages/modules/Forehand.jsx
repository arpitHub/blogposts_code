import { Link } from 'react-router-dom'
import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import PhaseExplorer from '../../components/widgets/PhaseExplorer'
import MistakeLab from '../../components/widgets/MistakeLab'
import StanceDiagram from '../../components/widgets/StanceDiagram'
import ContactWindow from '../../components/widgets/ContactWindow'
import { FOREHAND_PHASES, FOREHAND_MISTAKES } from '../../data/strokes/forehand'

export default function Forehand() {
  return (
    <div>
      <PageIntro moduleId="forehand" kicker="Stroke technique">
        <p>
          For most players the forehand is the biggest weapon on the court — the shot you’ll
          hit more than any other, and the one you build points around. The good news: it’s
          also the most natural swing in tennis. Scrub through it below, one phase at a time.
        </p>
      </PageIntro>

      <Prose>
        <p>
          Hold the racket with an <strong>Eastern or semi-Western grip</strong> (if those
          words mean nothing yet, take the two-minute detour to{' '}
          <Link to="/grips" className="font-medium text-clay-600 underline">Grip Types</Link>{' '}
          — it matters). Then meet the six phases every good forehand shares:
        </p>
      </Prose>

      <WidgetFrame wide title="The forehand, phase by phase" hint="press play, or drag the slider to scrub">
        <PhaseExplorer phases={FOREHAND_PHASES} />
      </WidgetFrame>

      <Prose title="Your feet decide before your arm swings">
        <p>
          Before every forehand you make one silent decision: how to set your feet. There are
          three answers, and each trades power for time differently. Flip between them:
        </p>
      </Prose>

      <WidgetFrame title="Open vs. neutral vs. closed stance" hint="viewed from above — note the hips line">
        <StanceDiagram />
      </WidgetFrame>

      <Prose title="Flat or topspin? It’s decided at contact">
        <p>
          The same six phases produce a flat missile or a heavy, kicking rally ball. The
          difference is the <strong>direction the racket travels through contact</strong> —
          level, or brushing up the back of the ball:
        </p>
      </Prose>

      <WidgetFrame title="The contact window: drive vs. brush" hint="slide between flat and heavy topspin">
        <ContactWindow />
      </WidgetFrame>

      <Prose title="The two mistakes that cause most bad forehands">
        <p>
          Watch the same swing with one ingredient removed. The left figure does it right;
          the right figure makes the error. One slider drives both, so you can freeze the
          exact moment things go wrong.
        </p>
        <Aside title="Coach’s eye">
          Almost every “my forehand broke down” story is one of these two. When your forehand
          sprays, check contact point first, body turn second — in that order.
        </Aside>
      </Prose>

      <WidgetFrame wide title="Good form vs. common mistake" hint="pick a mistake, then scrub the shared slider">
        <MistakeLab good={{ title: 'Good form', phases: FOREHAND_PHASES }} mistakes={FOREHAND_MISTAKES} />
      </WidgetFrame>

      <TryThis>
        <p>
          <strong>Drop-feed contact check:</strong> stand on the service line, drop a ball
          from your non-hitting hand, let it bounce, and hit a relaxed forehand — but freeze
          your finish for two full seconds. Check three things: did you contact the ball in
          front of your front hip? Is your racket over your opposite shoulder? Are you
          balanced? Twenty balls, three checks each. Boring — and transformative.
        </p>
      </TryThis>

      <NextUp moduleId="forehand" />
    </div>
  )
}
