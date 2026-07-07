import { Link } from 'react-router-dom'
import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import PhaseExplorer from '../../components/widgets/PhaseExplorer'
import MistakeLab from '../../components/widgets/MistakeLab'
import TossMeter from '../../components/widgets/TossMeter'
import { SERVE_PHASES, SERVE_MISTAKES } from '../../data/strokes/serve'

export default function Serve() {
  return (
    <div>
      <PageIntro moduleId="serve" kicker="Stroke technique">
        <p>
          The serve is the only shot in tennis you control completely — no opponent, no
          incoming ball, just you and your routine. That makes it the most learnable shot in
          the game… and the most personal. Under the surface, every good serve shares the
          same six-phase skeleton.
        </p>
      </PageIntro>

      <Prose>
        <p>
          One thing before you swing: the serve is hit with the{' '}
          <strong>continental grip</strong> — the “hammer” grip (
          <Link to="/grips" className="font-medium text-clay-600 underline">see Grip Types</Link>).
          Beginners who serve with a forehand grip hit a “waiter’s tray” serve that caps out
          fast. Learn it continental from day one, even if it feels awkward for a week.
        </p>
      </Prose>

      <WidgetFrame wide title="The serve, phase by phase" hint="press play, or drag the slider to scrub">
        <PhaseExplorer phases={SERVE_PHASES} />
      </WidgetFrame>

      <Prose title="The toss is half the serve">
        <p>
          Coaches joke that serving problems are toss problems wearing a disguise. The toss
          decides where contact can happen — and where contact happens decides everything
          else. Slide the toss around and watch what it forces your body to do:
        </p>
      </Prose>

      <WidgetFrame title="The toss laboratory" hint="drag the toss forward and back">
        <TossMeter />
      </WidgetFrame>

      <Prose title="The two serve-killers">
        <p>
          Compare a sound motion against the two most common breakdowns. One slider drives
          both figures — freeze it right at the moment the two swings diverge.
        </p>
        <Aside title="Why the drop matters">
          The racket drop is where “rhythm servers” beat “muscle servers”. A loose arm whips;
          a tight arm pushes. If your serve feels like effort, you’re probably skipping the
          drop.
        </Aside>
      </Prose>

      <WidgetFrame wide title="Good form vs. common mistake" hint="pick a mistake, then scrub the shared slider">
        <MistakeLab good={{ title: 'Good form', phases: SERVE_PHASES }} mistakes={SERVE_MISTAKES} />
      </WidgetFrame>

      <TryThis>
        <p>
          <strong>Toss-only practice (no hitting):</strong> stand on the baseline, place your
          racket on the ground with the head at your 1 o’clock, just inside the court. Toss
          and let the ball drop. If it lands on the strings, that toss was a serve you could
          have crushed. Ten in a row before you’re allowed to actually serve. Your toss will
          stop wandering within a week.
        </p>
      </TryThis>

      <NextUp moduleId="serve" />
    </div>
  )
}
