import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { PageIntro, Prose, Aside, NextUp } from '../../components/ui'
import { byId } from '../../data/modules'

const STAGES = [
  {
    id: 'beginner',
    name: 'Beginner',
    tagline: 'From “which end do I hold” to real rallies',
    color: '#3f7a54',
    milestones: [
      'Know the scoring system cold — can call the score out loud all match',
      'Find continental and forehand grips by feel, without looking',
      'Rally 10 balls cooperatively from the service line (mini-tennis)',
      'Get 5 of 10 serves into the correct box from the baseline',
      'Complete a full set against a friend, scored correctly',
    ],
    modules: ['scoring', 'court', 'grips', 'forehand', 'backhand'],
    drillLevel: 'beginner',
  },
  {
    id: 'rallier',
    name: 'Consistent Rallier',
    tagline: 'The ball goes back — on purpose, with shape',
    color: '#2e6142',
    milestones: [
      'Rally 10+ balls cross-court from the baseline without an error',
      'First serve lands in 60%+ of the time at rally pace',
      'Use topspin intentionally on the forehand (ball clears net by 1 m+ and still lands in)',
      'Split-step on every opponent hit — without thinking about it',
      'Recover toward the correct spot (not just the center mark) after each shot',
    ],
    modules: ['spin', 'serve', 'footwork'],
    drillLevel: 'beginner',
  },
  {
    id: 'club',
    name: 'Competitive Club Player',
    tagline: 'Points are won by plans, not accidents',
    color: '#b94a26',
    milestones: [
      'Hold serve more often than you lose it',
      'Approach the net behind short balls and finish with a volley',
      'Run the Serve + 1 pattern deliberately on your service points',
      'Own a reliable defensive slice for stretched positions',
      'Beat a same-level opponent using a pre-match plan you wrote down',
    ],
    modules: ['volley', 'overhead', 'positioning', 'strategy'],
    drillLevel: 'intermediate',
  },
  {
    id: 'advanced',
    name: 'Advanced',
    tagline: 'Weapons, disguise, and a body that lasts',
    color: '#7d311e',
    milestones: [
      'Vary spin, pace, and height deliberately by opponent and score',
      'Second serve kicks with enough spin that opponents can’t attack it',
      'Win points with all four foundational patterns in match play',
      'Follow a weekly tennis-specific conditioning + injury-prevention routine',
      'Compete in a league or box ladder — and enjoy the pressure',
    ],
    modules: ['strategy', 'fitness', 'equipment'],
    drillLevel: 'advanced',
  },
]

const STORAGE_KEY = 'cc-roadmap-v1'

function useChecklist() {
  const [done, setDone] = useState(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? {} } catch { return {} }
  })
  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(done)) } catch { /* private mode */ }
  }, [done])
  const toggle = (key) => setDone((d) => ({ ...d, [key]: !d[key] }))
  return [done, toggle]
}

function Stage({ stage, index, done, toggle }) {
  const doneCount = stage.milestones.filter((_, i) => done[`${stage.id}:${i}`]).length
  const pct = Math.round((doneCount / stage.milestones.length) * 100)

  return (
    <div className="relative pl-12">
      {/* path spine */}
      {index < STAGES.length - 1 && (
        <div className="absolute left-[15px] top-10 h-full w-1 rounded bg-court-100" />
      )}
      <div
        className="absolute left-0 top-1 flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold text-white shadow"
        style={{ background: stage.color }}
      >
        {index + 1}
      </div>

      <div className="mb-10 rounded-2xl border border-line bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h3 className="font-display text-xl font-bold text-court-950">{stage.name}</h3>
            <p className="text-sm text-court-600">{stage.tagline}</p>
          </div>
          <span className="font-mono text-xs text-court-500">{doneCount}/{stage.milestones.length} milestones</span>
        </div>

        <div className="mt-3 h-2 overflow-hidden rounded-full bg-court-100">
          <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: stage.color }} />
        </div>

        <ul className="mt-4 space-y-2">
          {stage.milestones.map((m, i) => {
            const key = `${stage.id}:${i}`
            return (
              <li key={i}>
                <label className="flex cursor-pointer items-start gap-3 rounded-lg px-2 py-1.5 transition hover:bg-court-50">
                  <input
                    type="checkbox"
                    checked={!!done[key]}
                    onChange={() => toggle(key)}
                    className="mt-1 h-4 w-4 accent-clay-500"
                  />
                  <span className={`text-sm leading-relaxed ${done[key] ? 'text-court-400 line-through' : 'text-court-800'}`}>
                    {m}
                  </span>
                </label>
              </li>
            )
          })}
        </ul>

        <div className="mt-4 flex flex-wrap items-center gap-1.5 border-t border-line pt-3">
          <span className="mr-1 text-[11px] font-bold uppercase tracking-wide text-court-500">Study for this stage:</span>
          {stage.modules.map((id) => {
            const m = byId(id)
            return (
              <Link key={id} to={m.path} className="rounded-full bg-court-50 px-2.5 py-1 text-xs font-medium text-court-700 transition hover:bg-clay-100 hover:text-clay-700">
                {m.title}
              </Link>
            )
          })}
          <Link
            to={`/drills?level=${stage.drillLevel}`}
            className="rounded-full bg-clay-500 px-2.5 py-1 text-xs font-semibold text-white transition hover:bg-clay-600"
          >
            Drills for this stage →
          </Link>
        </div>
      </div>
    </div>
  )
}

export default function Roadmap() {
  const [done, toggle] = useChecklist()

  return (
    <div>
      <PageIntro moduleId="roadmap" kicker="Progression">
        <p>
          “Getting good at tennis” is vague enough to be discouraging. This roadmap replaces
          it with four stages and twenty concrete, checkable milestones. Your progress saves
          in this browser — tick things off as they become true, and let the unticked boxes
          choose what you practice next.
        </p>
      </PageIntro>

      <section className="mx-auto max-w-3xl px-6 py-6">
        {STAGES.map((s, i) => (
          <Stage key={s.id} stage={s} index={i} done={done} toggle={toggle} />
        ))}
      </section>

      <Prose>
        <Aside title="How long does each stage take?">
          With one or two sessions a week: reaching Consistent Rallier typically takes a
          season; Competitive Club Player, a year or two. But the spread is enormous — and
          almost all of it is explained by one variable: deliberate practice (drills with a
          goal) versus just playing sets. The players who drill pass the players who don’t,
          every time.
        </Aside>
      </Prose>

      <NextUp moduleId="roadmap" />
    </div>
  )
}
