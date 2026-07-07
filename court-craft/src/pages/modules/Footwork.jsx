import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import SplitStepLab from '../../components/widgets/SplitStepLab'
import RecoveryLab from '../../components/widgets/RecoveryLab'

export default function Footwork() {
  return (
    <div>
      <PageIntro moduleId="footwork" kicker="Movement & tactics">
        <p>
          Watch club players and pros hit the same forehand and the swing looks 80% the same —
          the difference is that the pro was <em>there</em> a half-second earlier. Tennis is a
          running sport that happens to involve a racket. Two habits carry almost all of it:
          the split step and the recovery.
        </p>
      </PageIntro>

      <Prose>
        <p>
          The split step is a small bounce you take <strong>every single time your opponent
          hits</strong> — hundreds per match. It looks like nothing. It’s everything: landing
          just as they strike means your legs are loaded exactly when you learn where the
          ball is going. Try timing it yourself:
        </p>
      </Prose>

      <WidgetFrame wide title="Split-step timing drill" hint="spacebar works too — the wind-up varies every round">
        <SplitStepLab />
      </WidgetFrame>

      <Prose title="After the shot: recover, but not to the middle">
        <p>
          Beginners run back to the center mark after every ball. Better players recover to
          the spot that <strong>halves the opponent’s possible angles</strong> — which is
          usually near the middle, but rarely exactly on it. Switch between the scenarios and
          watch the cone:
        </p>
      </Prose>

      <WidgetFrame wide title="The recovery position" hint="pick where your shot landed">
        <RecoveryLab />
      </WidgetFrame>

      <Prose>
        <Aside title="First move, not first step">
          When the ball goes wide, the first motion isn’t a step — it’s a hip turn out of the
          split-step landing. Step-first movers cross their feet and arrive tangled; hip-turn
          movers glide. If you only remember one cue: <em>land, turn, then run.</em>
        </Aside>
        <p>
          One more piece of vocabulary: <strong>adjustment steps</strong>. The last meter to
          the ball is covered in several small stutter steps, not one big lunge — that’s how
          good players are always the right distance from the bounce. Big steps to travel,
          small steps to arrive.
        </p>
      </Prose>

      <TryThis>
        <p>
          <strong>Shadow split-steps:</strong> watch any tennis match tonight. Every time a
          player strikes the ball, hop where you sit — a tiny split step, timed to their
          contact. You’ll be wrong (late) for the first few minutes, then something clicks:
          you’ll start reading the swing instead of the ball. That read is the entire skill,
          and you just trained it from a couch.
        </p>
      </TryThis>

      <NextUp moduleId="footwork" />
    </div>
  )
}
