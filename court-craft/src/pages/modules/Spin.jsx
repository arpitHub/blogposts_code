import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import TrajectoryLab, { SpinComparison } from '../../components/widgets/TrajectoryLab'
import MagnusDiagram from '../../components/widgets/MagnusDiagram'

export default function Spin() {
  return (
    <div>
      <PageIntro moduleId="spin" kicker="Flagship interactive">
        <p>
          Spin is the single biggest idea separating beginners from everyone else. It’s why pros
          can swing at full speed and still land the ball in, why some bounces jump at your
          shoulders and others slide under your racket. You don’t need physics equations —
          you need to <em>see</em> it. Grab the sliders below and find out for yourself.
        </p>
      </PageIntro>

      <WidgetFrame
        wide
        title="The Ball Flight Lab"
        hint="drag the sliders — the shot replays instantly"
      >
        <TrajectoryLab />
      </WidgetFrame>

      <Prose title="What you just discovered">
        <p>
          Play with the <strong>swing path</strong> slider and watch what happens. Brushing up
          the back of the ball (low-to-high) creates <strong>topspin</strong>: the ball can fly
          well above the net and still dive down inside the baseline. Chopping down
          (high-to-low) creates <strong>slice</strong>: the ball floats on a shallow arc and
          skids low off the bounce.
        </p>
        <p>
          Now try to hit a <strong>flat</strong> ball hard and deep. Notice how narrow the
          window is — a touch too low and it clips the net, a touch too high and it sails long.
          That window is called <em>margin for error</em>, and spin is how you buy more of it.
          Topspin lets you swing faster <em>and</em> miss less. That’s not a trick; it’s the
          core deal of modern tennis.
        </p>
        <Aside title="Rule of thumb">
          Rally balls should cross the net about 1–2 meters above the tape. If your shots
          are skimming the net cord, you’re not playing with enough spin — you’re playing
          with luck.
        </Aside>
      </Prose>

      <WidgetFrame title="Why spin bends the ball — the Magnus effect" hint="switch between the three spin types">
        <MagnusDiagram />
      </WidgetFrame>

      <Prose title="The bounce tells the story">
        <p>
          Spin doesn’t stop working when the ball lands. A topspin ball grips the court and
          <strong> kicks up and forward</strong> — that’s what pushes opponents behind the
          baseline. A slice <strong>stays low and slows down</strong>, forcing your opponent to
          bend and hit up. Compare all three at rally pace:
        </p>
      </Prose>

      <WidgetFrame wide title="Same rally, three spins" hint="all three shots land in — look at the arcs and the bounces">
        <SpinComparison />
      </WidgetFrame>

      <TryThis>
        <p>
          <strong>The one-meter rule:</strong> rally cross-court and aim every ball at
          least one meter over the net — no exceptions. If a ball lands long, don’t aim lower;
          add more low-to-high brush instead (finish with your racket by your opposite
          shoulder). You’ll feel the ball dip. Ten balls in a row over the meter mark and
          inside the baseline means you’re creating real topspin.
        </p>
      </TryThis>

      <NextUp moduleId="spin" />
    </div>
  )
}
