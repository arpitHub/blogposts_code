import { Link } from 'react-router-dom'
import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import GripWheel from '../../components/widgets/GripWheel'

export default function Grips() {
  return (
    <div>
      <PageIntro moduleId="grips" kicker="Fundamentals">
        <p>
          How you hold the racket quietly decides what your shots <em>can</em> be: which way
          the strings face at contact, how much spin comes naturally, which balls feel easy
          and which feel impossible. A racket handle has eight flat sides — “bevels” — and
          every grip is just a choice of which bevel your index knuckle rests on.
        </p>
      </PageIntro>

      <WidgetFrame wide title="The grip wheel" hint="pick a grip — watch the knuckle move and the face tilt">
        <GripWheel />
      </WidgetFrame>

      <Prose title="Why one bevel changes everything">
        <p>
          Look at the middle panel as you switch grips: rotating your hand just one bevel
          tilts the racket face at contact by 10–15 degrees. That tilt is destiny. An open
          face slides under the ball (slice); a square face drives through it (flat); a
          closed face forces you to brush up (topspin — see{' '}
          <Link to="/spin" className="font-medium text-clay-600 underline">why that matters</Link>).
          Nobody “chooses” to hit flat or spinny on a given ball as much as their grip
          chooses for them.
        </p>
        <Aside title="Which grips do you actually need?">
          Just two to start: <strong>continental</strong> for serves and volleys, and{' '}
          <strong>eastern or semi-western</strong> for forehands. Add a backhand grip when
          you pick your backhand style. Western can wait until high, heavy topspin is
          genuinely your game.
        </Aside>
        <p>
          One habit to build from day one: <strong>change grips between shots</strong>. Rest
          the racket in your non-hitting hand at the throat, and let that hand spin the
          handle while you move. By the time you split-step, the right grip is already there.
        </p>
      </Prose>

      <TryThis>
        <p>
          <strong>The blind grip drill:</strong> close your eyes, spin the racket in your
          hand, then find continental by feel alone — hammer grip, knuckle on bevel 2. Open
          your eyes and check. Then do the same for your forehand grip. Ten times each.
          Match play never gives you time to look at your hand; the feel has to be automatic.
        </p>
      </TryThis>

      <NextUp moduleId="grips" overrideId="forehand" />
    </div>
  )
}
