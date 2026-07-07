// Volley keyframes — deliberately compact: the whole point is how little
// swing there is. 4 phases instead of 6.

export const VOLLEY_PHASES = [
  {
    name: 'Ready at the net',
    note: 'Racket head up at chest height, hands out front, weight forward on the balls of your feet. At the net you have half the reaction time — the racket starts where the ball will be.',
    checkpoint: 'Racket head above your wrists, elbows in front of your body.',
    pose: {
      head: [108, 76], neck: [108, 90], hip: [108, 140],
      kneeB: [96, 173], ankleB: [92, 205], kneeF: [122, 173], ankleF: [126, 205],
      elbowR: [124, 110], wristR: [132, 96], racketTip: [142, 66],
      elbowL: [92, 110], wristL: [104, 98],
      ball: [235, 96],
    },
  },
  {
    name: 'Turn — not back',
    note: 'The shoulders make a small quarter turn and the racket sets slightly to the side. That’s the entire “backswing”. If the racket goes behind your body at the net, you’ve already lost the exchange.',
    checkpoint: 'Racket stays in front of your shoulder line — always visible in your peripheral vision.',
    pose: {
      head: [106, 76], neck: [106, 90], hip: [107, 141],
      kneeB: [95, 174], ankleB: [91, 205], kneeF: [121, 174], ankleF: [125, 205],
      elbowR: [112, 110], wristR: [118, 94], racketTip: [126, 63],
      elbowL: [98, 110], wristL: [110, 100],
      ball: [198, 98],
    },
  },
  {
    name: 'Step & punch',
    note: 'Step diagonally forward with the opposite foot and block the ball out front — a short punch from the shoulder, wrist firm, like pushing a door shut. The ball’s incoming pace does the work; you just redirect it.',
    checkpoint: 'Contact in front of your eyes; wrist locked; step lands as you hit.',
    pose: {
      head: [110, 74], neck: [110, 88], hip: [111, 139],
      kneeB: [98, 172], ankleB: [94, 204], kneeF: [128, 171], ankleF: [136, 205],
      elbowR: [128, 106], wristR: [146, 98], racketTip: [158, 70],
      elbowL: [94, 108], wristL: [102, 100],
      ball: [153, 82],
    },
  },
  {
    name: 'Short finish, reset',
    note: 'The racket travels only a little further — maybe 30 cm — then snaps back to ready position. Volleys come in pairs; the reset IS the follow-through.',
    checkpoint: 'Racket back up at chest height before the ball crosses the net.',
    pose: {
      head: [109, 75], neck: [109, 89], hip: [110, 140],
      kneeB: [97, 172], ankleB: [93, 204], kneeF: [126, 172], ankleF: [133, 205],
      elbowR: [126, 108], wristR: [138, 96], racketTip: [148, 67],
      elbowL: [93, 109], wristL: [104, 99],
      ball: [242, 94],
    },
  },
]

export const VOLLEY_MISTAKES = [
  {
    id: 'swing',
    title: 'Swinging at it',
    why: 'A groundstroke swing needs time you don’t have at the net. The big take-back means late contact, mistimed hits, and volleys dumped into the net or ballooned long. Fix: imagine a pane of glass just behind your shoulders — the racket must never break it.',
    phases: [
      { name: 'Ready', note: 'Everything starts the same…', pose: VOLLEY_PHASES[0].pose },
      {
        name: 'Big take-back',
        note: 'The racket swings back behind the body — groundstroke habits at the net.',
        pose: {
          head: [105, 77], neck: [105, 91], hip: [106, 141],
          kneeB: [94, 174], ankleB: [90, 205], kneeF: [120, 174], ankleF: [124, 205],
          elbowR: [90, 112], wristR: [76, 122], racketTip: [58, 104],
          elbowL: [100, 110], wristL: [112, 100],
          ball: [198, 98],
        },
      },
      {
        name: 'Contact — late & wild',
        note: 'The racket is still accelerating through a long arc; contact happens beside the body with a moving face.',
        pose: {
          head: [108, 75], neck: [108, 89], hip: [109, 140],
          kneeB: [96, 172], ankleB: [92, 204], kneeF: [124, 172], ankleF: [130, 205],
          elbowR: [118, 112], wristR: [132, 104], racketTip: [140, 74],
          elbowL: [96, 108], wristL: [104, 100],
          ball: [130, 88],
        },
      },
      {
        name: 'Long follow-through',
        note: 'The swing carries the racket way across the body — no chance to reset before the next ball.',
        pose: {
          head: [108, 75], neck: [108, 89], hip: [109, 140],
          kneeB: [97, 172], ankleB: [93, 204], kneeF: [125, 172], ankleF: [131, 205],
          elbowR: [118, 104], wristR: [104, 96], racketTip: [80, 82],
          elbowL: [95, 109], wristL: [103, 100],
          ball: [225, 110],
        },
      },
    ],
  },
]
