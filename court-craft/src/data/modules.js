// Central registry of all content modules. Pages, nav, landing grid, and the
// roadmap all read from this single source of truth.

export const CATEGORIES = [
  { id: 'fundamentals', label: 'Fundamentals', blurb: 'Scoring, grips, and the rules of the court — everything you need before your first rally.' },
  { id: 'strokes', label: 'Strokes', blurb: 'Phase-by-phase breakdowns of every shot in the game, from serve to smash.' },
  { id: 'movement', label: 'Movement & Tactics', blurb: 'Footwork, positioning, and the patterns that win points.' },
  { id: 'equipment', label: 'Equipment & Fitness', blurb: 'Gear tradeoffs and the conditioning that keeps you on court.' },
  { id: 'progression', label: 'Progression', blurb: 'A staged roadmap from first rally to competitive play, with drills for every level.' },
]

export const LEVELS = {
  beginner: { label: 'Beginner', color: 'bg-court-100 text-court-700' },
  intermediate: { label: 'Intermediate', color: 'bg-clay-100 text-clay-700' },
  advanced: { label: 'Advanced', color: 'bg-court-800 text-court-50' },
  all: { label: 'All levels', color: 'bg-line text-court-600' },
}

export const MODULES = [
  {
    id: 'scoring', path: '/scoring', title: 'The Scoring System', category: 'fundamentals', level: 'beginner',
    icon: '15', order: 1,
    blurb: 'Love, deuce, tiebreaks — play through live points on an interactive scoreboard until the strangest scoring system in sport makes sense.',
  },
  {
    id: 'grips', path: '/grips', title: 'Grip Types', category: 'fundamentals', level: 'beginner',
    icon: '✊', order: 2,
    blurb: 'Continental, Eastern, Western — rotate the handle, see where your hand sits, and learn which shots use which grip.',
  },
  {
    id: 'court', path: '/court', title: 'Court & Rules', category: 'fundamentals', level: 'beginner',
    icon: '▭', order: 3,
    blurb: 'Explore a labeled court diagram: dimensions, service boxes, let serves, foot faults, and on-court etiquette.',
  },
  {
    id: 'serve', path: '/serve', title: 'The Serve', category: 'strokes', level: 'all',
    icon: '🎾', order: 4,
    blurb: 'The only shot you fully control. Scrub through toss, trophy position, contact, and follow-through — and see the classic errors side by side.',
  },
  {
    id: 'forehand', path: '/forehand', title: 'The Forehand', category: 'strokes', level: 'all',
    icon: '➚', order: 5,
    blurb: 'Your biggest weapon. Unit turn, stance, swing path, and contact point — explored phase by phase with a topspin vs. flat comparison.',
  },
  {
    id: 'backhand', path: '/backhand', title: 'The Backhand', category: 'strokes', level: 'all',
    icon: '⬔', order: 6,
    blurb: 'One hand or two? Toggle both variants side by side and see the mechanical differences that matter.',
  },
  {
    id: 'volley', path: '/volley', title: 'The Volley', category: 'strokes', level: 'intermediate',
    icon: '◇', order: 7,
    blurb: 'No backswing, all block. The compact punch at the net, and the footwork that gets you there.',
  },
  {
    id: 'overhead', path: '/overhead', title: 'Overhead & Smash', category: 'strokes', level: 'intermediate',
    icon: '⌄', order: 8,
    blurb: 'A serve you hit on the move. Same motion, new footwork — with the key differences called out.',
  },
  {
    id: 'spin', path: '/spin', title: 'Spin & Ball Flight', category: 'strokes', level: 'all',
    icon: '↻', order: 9, flagship: true,
    blurb: 'Why topspin dives and slice floats. Control the racket face and swing path, and watch the ball flight and bounce change live.',
  },
  {
    id: 'footwork', path: '/footwork', title: 'Footwork & Court Movement', category: 'movement', level: 'intermediate',
    icon: '👣', order: 10,
    blurb: 'The split step, first move, and recovery — timed against an incoming shot on an animated court.',
  },
  {
    id: 'positioning', path: '/positioning', title: 'Court Positioning', category: 'movement', level: 'intermediate',
    icon: '⊹', order: 11,
    blurb: 'Where should you stand? Drag your player around the court and get live feedback for singles and doubles scenarios.',
  },
  {
    id: 'strategy', path: '/strategy', title: 'Match Strategy & Shot Selection', category: 'movement', level: 'advanced',
    icon: '♟', order: 12,
    blurb: 'Patterns of play, attack vs. rally decisions, and reading your opponent — animated point patterns on a live court.',
  },
  {
    id: 'equipment', path: '/equipment', title: 'Equipment', category: 'equipment', level: 'all',
    icon: '⚙', order: 13,
    blurb: 'Racket weight, head size, string tension — slide the tradeoffs and see how power, control, and comfort shift.',
  },
  {
    id: 'fitness', path: '/fitness', title: 'Fitness for Tennis', category: 'equipment', level: 'all',
    icon: '♥', order: 14,
    blurb: 'Tennis-specific conditioning and the injury-prevention basics every player should know.',
  },
  {
    id: 'roadmap', path: '/roadmap', title: 'Skill Roadmap', category: 'progression', level: 'all',
    icon: '⚑', order: 15,
    blurb: 'Beginner → Consistent Rallier → Club Competitor → Advanced. Concrete milestones at every stage, linked to the modules and drills that get you there.',
  },
  {
    id: 'drills', path: '/drills', title: 'Practice Drills Library', category: 'progression', level: 'all',
    icon: '▦', order: 16,
    blurb: 'A searchable, filterable library of drills tagged by stroke and level, each with an animated court pattern.',
  },
]

export const byId = (id) => MODULES.find((m) => m.id === id)
export const byCategory = (catId) => MODULES.filter((m) => m.category === catId).sort((a, b) => a.order - b.order)

// The "Start here" linear path for total beginners.
export const BEGINNER_PATH = ['scoring', 'court', 'grips', 'forehand', 'backhand', 'serve', 'spin', 'footwork', 'roadmap']

export const nextModule = (id) => {
  const m = byId(id)
  if (!m) return null
  return MODULES.find((x) => x.order === m.order + 1) ?? null
}
