// Overhead/smash keyframes — a serve you hit on the move. Poses deliberately
// echo the serve's trophy → contact → follow-through chain.

import { SERVE_PHASES } from './serve'

export const OVERHEAD_PHASES = [
  {
    name: 'Read the lob',
    note: 'The moment you recognize a lob, everything starts: shoulders turn sideways and the free hand points up at the ball like a sight. That pointing arm is your rangefinder — it tracks the ball all the way down.',
    checkpoint: 'Sideways to the net, free hand pointing at the ball.',
    pose: {
      head: [106, 78], neck: [106, 92], hip: [107, 140],
      kneeB: [95, 173], ankleB: [91, 205], kneeF: [119, 173], ankleF: [123, 205],
      elbowR: [90, 100], wristR: [84, 78], racketTip: [74, 50],
      elbowL: [112, 78], wristL: [118, 60],
      ball: [196, 22],
    },
  },
  {
    name: 'Shuffle under it',
    note: 'Move with sidesteps — never backpedal straight back (that’s how ankles get rolled). Position yourself so the ball would land slightly in FRONT of you, exactly like a good serve toss.',
    checkpoint: 'Ball would drop at your 1 o’clock, not on your head.',
    pose: {
      head: [104, 77], neck: [104, 91], hip: [106, 141],
      kneeB: [93, 174], ankleB: [88, 205], kneeF: [117, 174], ankleF: [122, 205],
      elbowR: [88, 96], wristR: [83, 72], racketTip: [74, 44],
      elbowL: [111, 74], wristL: [117, 56],
      ball: [162, 16],
    },
  },
  {
    name: 'Trophy — abbreviated',
    note: 'Take the racket straight up into the trophy position — skip the big wind-up your serve uses; there’s no time and you don’t need it. Knees load as the ball drops toward your window.',
    checkpoint: 'Same trophy as your serve, reached by the shortcut.',
    pose: {
      ...SERVE_PHASES[2].pose,
      ball: [138, 14],
    },
  },
  {
    name: 'Contact — high & in front',
    note: 'Identical to the serve: legs drive, racket drops behind the back and whips up to full extension. Meet the ball as high as you can reach, slightly in front. Snap the wrist through — this is the one shot where you swing at 100%.',
    checkpoint: 'Full arm extension; contact in front of your head.',
    pose: {
      ...SERVE_PHASES[4].pose,
      ball: [128, 12],
    },
  },
  {
    name: 'Follow-through',
    note: 'The racket finishes across your body as you land on the front foot, already moving back toward the net for the next ball — one smash rarely ends the point against good lobbers.',
    checkpoint: 'Balanced landing, eyes already back on your opponent.',
    pose: {
      ...SERVE_PHASES[5].pose,
      ball: [256, 100],
    },
  },
]

export const OVERHEAD_VS_SERVE = [
  {
    dim: 'The toss',
    serve: 'You place the ball exactly where you want it.',
    overhead: 'Your opponent “tosses” for you — badly, on purpose. Footwork replaces the toss.',
  },
  {
    dim: 'The wind-up',
    serve: 'Full, rhythmic take-back — you have all the time in the world.',
    overhead: 'Abbreviated: racket goes straight up to trophy. No time for the loop.',
  },
  {
    dim: 'Footwork',
    serve: 'Feet planted; energy comes from the ground up in place.',
    overhead: 'Sidesteps to get under and slightly behind the ball while it falls.',
  },
  {
    dim: 'Targets',
    serve: 'Constrained to the service box.',
    overhead: 'The whole court — hit to open space or bounce it over the fence.',
  },
]
