import { PageIntro, WidgetFrame, Prose, Aside, TryThis, NextUp } from '../../components/ui'
import InjuryMap from '../../components/widgets/InjuryMap'

const WARMUP = [
  ['0:00–2:00', 'Raise the temperature', 'Easy jog around the court, sidesteps, backwards jog, skips. Sweat is optional; warmth isn’t.'],
  ['2:00–4:00', 'Open the hips & shoulders', 'Leg swings (front and side, 10 each), walking lunges with a twist, big arm circles both ways.'],
  ['4:00–5:30', 'Wake up the reflexes', 'Ten explosive split steps, five short sprints to the net with shuffle-back recovery.'],
  ['5:30–8:00', 'Progressive hitting', 'Start in the service boxes with mini-tennis — soft rally, exaggerated spin — then walk back to the baseline as the ball speeds up.'],
]

export default function Fitness() {
  return (
    <div>
      <PageIntro moduleId="fitness" kicker="Equipment & fitness">
        <p>
          Tennis fitness isn’t marathon fitness. A match is hundreds of 3–8 second sprints
          with direction changes, separated by rests — closer to interval training with a
          racket in your hand. Train for <em>that</em>, and protect the seven body parts
          tennis loves to pick on:
        </p>
      </PageIntro>

      <WidgetFrame wide title="The tennis injury map" hint="tap a hotspot — each has a cause and a fix">
        <InjuryMap />
      </WidgetFrame>

      <Prose title="Train movements, not muscles">
        <p>
          The gym work that transfers to court fits in three buckets.{' '}
          <strong>Legs & push-off:</strong> split squats, lateral lunges, calf raises —
          tennis lives on one leg at a time. <strong>Core & rotation:</strong> side planks,
          medicine-ball rotational throws — every groundstroke is a rotation, and the core
          is the gearbox between legs and arm. <strong>Shoulder durability:</strong> band
          external rotations and face pulls — small muscles, huge job on every serve.
        </p>
        <Aside title="The 2-to-1 rule">
          For every pushing/hitting exercise, do two pulling/decelerating ones. Tennis
          already trains acceleration all day; injuries happen in the braking. Your rotator
          cuff, hamstrings, and mid-back are your brakes — maintain them like you drive.
        </Aside>
      </Prose>

      <Prose title="The 8-minute warm-up that prevents most of it">
        <div className="space-y-2">
          {WARMUP.map(([time, title, text]) => (
            <div key={time} className="flex gap-4 rounded-xl border border-line bg-white px-4 py-3 shadow-sm">
              <span className="shrink-0 pt-0.5 font-mono text-xs text-clay-600">{time}</span>
              <div>
                <div className="text-sm font-bold text-court-900">{title}</div>
                <p className="text-sm leading-relaxed text-court-700">{text}</p>
              </div>
            </div>
          ))}
        </div>
      </Prose>

      <TryThis>
        <p>
          <strong>The spider drill (court-specific conditioning):</strong> place five balls —
          on both baseline corners, both service-line/sideline junctions, and the center
          service T. Starting at the center mark, sprint to each ball one at a time and
          carry it back. Time yourself. Rest two minutes, repeat three rounds. It trains
          exactly the sprint-brake-turn pattern that wins third sets.
        </p>
      </TryThis>

      <NextUp moduleId="fitness" />
    </div>
  )
}
