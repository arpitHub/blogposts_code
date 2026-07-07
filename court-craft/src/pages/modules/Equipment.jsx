import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import RacketTuner from '../../components/widgets/RacketTuner'

export default function Equipment() {
  return (
    <div>
      <PageIntro moduleId="equipment" kicker="Equipment & fitness">
        <p>
          No racket will fix your forehand — but the wrong one will absolutely fight it.
          Racket shopping is three sliders in a trench coat: weight, head size, and string
          tension, each trading power against control. Move them yourself and watch the
          trade-offs breathe:
        </p>
      </PageIntro>

      <WidgetFrame wide title="The racket tuner" hint="every spec is a trade — there is no free power">
        <RacketTuner />
      </WidgetFrame>

      <Prose title="How to actually buy a racket">
        <p>
          The meters above compress a lot of physics, but the buying advice is short:{' '}
          <strong>beginners</strong> want light (270–285 g) and roomy (102–107 in²) —
          maximum forgiveness while your contact point stabilizes. <strong>Improvers</strong>{' '}
          move toward 295–305 g and ~100 in² as swings get fuller. And{' '}
          <strong>everyone</strong> should demo before buying — two rackets with identical
          specs can feel completely different because of balance and stiffness.
        </p>
        <Aside title="Strings matter more than you think">
          A racket is half strings. Soft multifilament or natural gut = comfort and power
          (most club players should be here). Stiff polyester = spin and control for fast,
          full swings — and a sore elbow for everyone else. Restring at least twice a year:
          strings die quietly long before they break.
        </Aside>
        <p>
          The unglamorous rest: <strong>shoes</strong> must be actual tennis shoes — running
          shoes have zero lateral support and courts eat ankles. <strong>Balls</strong>:
          fresh pressurized balls for matches (they go flat in weeks); pressureless for the
          practice basket. <strong>Grip</strong>: replace your overgrip monthly — a slippery
          grip quietly ruins technique by making you squeeze.
        </p>
      </Prose>

      <TryThis>
        <p>
          <strong>The demo test:</strong> when trying a racket, don’t judge it on winners —
          judge it on your <em>bad</em> swings. Hit twenty balls slightly late on purpose.
          The racket that keeps those in the court is the one that will win you matches,
          because mishits are half of real tennis.
        </p>
      </TryThis>

      <NextUp moduleId="equipment" />
    </div>
  )
}
