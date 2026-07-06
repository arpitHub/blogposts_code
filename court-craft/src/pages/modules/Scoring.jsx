import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import ScoreSimulator from '../../components/widgets/ScoreSimulator'

function NestingDiagram() {
  return (
    <svg viewBox="0 0 640 180" className="w-full" role="img" aria-label="Points nest inside games, games inside sets, sets inside the match">
      <rect x="4" y="8" width="632" height="164" rx="14" fill="none" stroke="#cf5f38" strokeWidth="2.5" />
      <text x="20" y="34" fontSize="15" fontWeight="bold" fill="#b94a26">MATCH</text>
      <text x="86" y="34" fontSize="12" fill="#9a3a1f">= best of 3 sets (first to 2)</text>

      <rect x="20" y="48" width="600" height="112" rx="12" fill="#fdf5f2" stroke="#3f7a54" strokeWidth="2" />
      <text x="36" y="72" fontSize="14" fontWeight="bold" fill="#2e6142">SET</text>
      <text x="72" y="72" fontSize="12" fill="#254e37">= first to 6 games, win by 2 (6–6 → tiebreak)</text>

      <rect x="36" y="84" width="568" height="62" rx="10" fill="#f0f7f2" stroke="#1f3f2e" strokeWidth="1.5" />
      <text x="52" y="107" fontSize="13" fontWeight="bold" fill="#1a3427">GAME</text>
      <text x="102" y="107" fontSize="12" fill="#254e37">= first to 4 points, win by 2</text>
      <g fontSize="12" fill="#1a3427" fontFamily="monospace">
        <text x="52" y="132">0 → 15 → 30 → 40 → game</text>
        <text x="270" y="132" fill="#9a3a1f">(tied at 40 = deuce → advantage → game)</text>
      </g>
    </svg>
  )
}

export default function Scoring() {
  return (
    <div>
      <PageIntro moduleId="scoring" kicker="Fundamentals">
        <p>
          Tennis scoring trips up more newcomers than any swing ever will: numbers that jump
          0–15–30–40, a word for zero borrowed from eggs (maybe), and games inside sets inside
          matches. Instead of memorizing rules, <strong>play a match right here</strong> —
          award points and let the scoreboard teach you as it goes.
        </p>
      </PageIntro>

      <WidgetFrame wide title="The living scoreboard" hint="award points and read the commentary">
        <ScoreSimulator />
      </WidgetFrame>

      <Prose title="The structure: three boxes, one inside another">
        <p>
          Everything the scoreboard just did fits in one picture. Points build games, games
          build sets, sets build the match — and each layer resets when it's won:
        </p>
      </Prose>

      <section className="mx-auto max-w-3xl px-6 py-2">
        <div className="rounded-2xl border border-line bg-white p-5 shadow-sm">
          <NestingDiagram />
        </div>
      </section>

      <Prose>
        <p>
          Two consequences worth noticing. First, <strong>not all points are equal</strong>:
          winning 30–40 down saves a game; winning at 40–30 banks one. Big-match players are
          famous precisely for winning the important points. Second, <strong>the score can
          lie</strong>: you can win fewer total points than your opponent and still win the
          match, if you spend your wins wisely. Tennis rewards timing, not volume.
        </p>
        <Aside title='Why is zero called "love"?'>
          The favorite theory: French players called zero <em>l’œuf</em> — “the egg” — for its
          shape, and English tongues mangled it into “love”. (Compare “duck” for zero in
          cricket, short for “duck’s egg”.) The competing theory says it’s from playing “for
          love” — for nothing. Nobody knows for sure; both make good pub trivia.
        </Aside>
        <Aside title="And why 15, 30, 40?">
          Most likely a medieval clock face: points moved a hand a quarter turn — 15, 30, 45 —
          and 45 (<em>quarante-cinq</em>) got lazily shortened to 40. Six hundred years later,
          we’re all still saying it.
        </Aside>
      </Prose>

      <TryThis>
        <p>
          <strong>Scoreboard fluency test:</strong> in the simulator above, get the score to
          exactly <em>deuce</em>, then win the game from there (it takes two in a row). Then
          force a tiebreak at 6–6 and finish it. If you can do both without reading the
          commentary, you’ll never be lost on a real court — and you’re ready to call the
          score out loud when you serve, like players are expected to.
        </p>
      </TryThis>

      <NextUp moduleId="scoring" overrideId="court" />
    </div>
  )
}
