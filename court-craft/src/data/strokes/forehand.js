// Forehand keyframes (right-handed, side view, hitting toward the right).
// Coordinate space: 240x220, ground at y=205 — see figures.jsx.

export const FOREHAND_PHASES = [
  {
    name: 'Ready',
    note: 'Athletic base: knees flexed, weight on the balls of your feet, racket held out front with the non-hitting hand supporting the throat. From here you can move to either side.',
    checkpoint: 'Could someone push you over? If yes, get lower.',
    pose: {
      head: [108, 78], neck: [108, 92], hip: [108, 140],
      kneeB: [96, 172], ankleB: [92, 205], kneeF: [122, 172], ankleF: [126, 205],
      elbowR: [126, 112], wristR: [134, 128], racketTip: [152, 112],
      elbowL: [90, 112], wristL: [104, 126],
      ball: [232, 118],
    },
  },
  {
    name: 'Unit turn',
    note: 'The single most important move: turn your shoulders and hips together — as a unit — the moment you read “forehand”. The racket goes back because your body turns, not because your arm pulls it.',
    checkpoint: 'Non-hitting hand points across at the incoming ball.',
    pose: {
      head: [104, 78], neck: [104, 92], hip: [106, 141],
      kneeB: [94, 173], ankleB: [90, 205], kneeF: [120, 173], ankleF: [124, 205],
      elbowR: [86, 108], wristR: [70, 116], racketTip: [52, 96],
      elbowL: [122, 108], wristL: [140, 112],
      ball: [206, 122],
    },
  },
  {
    name: 'Racket drop',
    note: 'As the ball approaches, the racket head drops below the hand and below the incoming ball. Knees load deeper. This “low point” is what lets you swing low-to-high for topspin.',
    checkpoint: 'Racket head below your wrist, below ball height.',
    pose: {
      head: [104, 80], neck: [104, 94], hip: [106, 144],
      kneeB: [92, 175], ankleB: [88, 205], kneeF: [120, 175], ankleF: [124, 205],
      elbowR: [90, 124], wristR: [76, 148], racketTip: [58, 168],
      elbowL: [126, 112], wristL: [142, 118],
      ball: [182, 126],
    },
  },
  {
    name: 'Contact',
    note: 'Meet the ball out in front of your body, around waist height, with the racket face vertical. Your legs drive up, hips rotate through, and the racket is accelerating — not decelerating — through this point.',
    checkpoint: 'Contact is in front of your front hip, not beside you.',
    pose: {
      head: [106, 76], neck: [106, 90], hip: [108, 140],
      kneeB: [96, 173], ankleB: [92, 203], kneeF: [122, 172], ankleF: [126, 205],
      elbowR: [124, 116], wristR: [146, 124], racketTip: [154, 92],
      elbowL: [88, 110], wristL: [80, 122],
      ball: [152, 100],
    },
  },
  {
    name: 'Extension',
    note: 'Keep the racket moving out toward your target after contact — imagine hitting three balls in a row lined up toward the net. This is where depth and pace come from.',
    checkpoint: 'Racket reaches toward the target before it wraps.',
    pose: {
      head: [108, 76], neck: [108, 90], hip: [110, 139],
      kneeB: [96, 172], ankleB: [93, 203], kneeF: [122, 171], ankleF: [126, 205],
      elbowR: [132, 106], wristR: [154, 104], racketTip: [186, 96],
      elbowL: [88, 112], wristL: [82, 124],
      ball: [232, 84],
    },
  },
  {
    name: 'Follow-through',
    note: 'The racket finishes up and across, over your opposite shoulder. A full, relaxed finish is proof you accelerated through the ball instead of poking at it. Then: recover for the next shot.',
    checkpoint: 'Racket ends near your opposite shoulder; you can “answer the phone” with it.',
    pose: {
      head: [106, 76], neck: [106, 90], hip: [108, 139],
      kneeB: [98, 172], ankleB: [96, 203], kneeF: [122, 171], ankleF: [126, 205],
      elbowR: [116, 98], wristR: [96, 92], racketTip: [64, 100],
      elbowL: [88, 112], wristL: [86, 130],
      ball: [262, 62],
    },
  },
]

// --- Common mistakes (same 6-phase skeleton so one slider drives both) -----

export const FOREHAND_MISTAKES = [
  {
    id: 'late',
    title: 'Late contact',
    why: 'Hitting beside (or behind) your body means the racket face is still opening when it meets the ball — power leaks away, the ball sprays, and your wrist absorbs the shock. Fix: prepare earlier (unit turn the moment you see “forehand”) and meet the ball a full step further in front.',
    phases: [
      { name: 'Ready', note: 'Everything starts the same…', pose: FOREHAND_PHASES[0].pose },
      {
        name: 'Slow preparation',
        note: 'The shoulders stay square too long — the racket is still in front while the ball closes in.',
        pose: {
          ...FOREHAND_PHASES[0].pose,
          elbowR: [120, 114], wristR: [126, 130], racketTip: [144, 116],
          ball: [200, 122],
        },
      },
      {
        name: 'Rushed take-back',
        note: 'Now the arm yanks the racket back at the last moment — no time for a real drop.',
        pose: {
          head: [104, 79], neck: [104, 93], hip: [106, 142],
          kneeB: [94, 174], ankleB: [90, 205], kneeF: [120, 174], ankleF: [124, 205],
          elbowR: [90, 112], wristR: [76, 126], racketTip: [58, 112],
          elbowL: [110, 112], wristL: [120, 122],
          ball: [172, 124],
        },
      },
      {
        name: 'Contact — too late',
        note: 'The ball is already level with the hip. The racket face is open and still accelerating.',
        pose: {
          head: [106, 78], neck: [106, 92], hip: [108, 141],
          kneeB: [96, 173], ankleB: [92, 204], kneeF: [122, 172], ankleF: [126, 205],
          elbowR: [116, 122], wristR: [124, 132], racketTip: [130, 100],
          elbowL: [90, 112], wristL: [84, 124],
          ball: [128, 106],
        },
      },
      {
        name: 'Cramped extension',
        note: 'There’s no room to extend toward the target — the swing jams into the body.',
        pose: {
          head: [107, 77], neck: [107, 91], hip: [109, 140],
          kneeB: [96, 172], ankleB: [93, 204], kneeF: [122, 171], ankleF: [126, 205],
          elbowR: [122, 112], wristR: [136, 112], racketTip: [158, 118],
          elbowL: [89, 112], wristL: [83, 124],
          ball: [210, 110],
        },
      },
      {
        name: 'Short finish',
        note: 'The follow-through stops early — the ball floats without spin or direction.',
        pose: {
          head: [106, 77], neck: [106, 91], hip: [108, 140],
          kneeB: [98, 172], ankleB: [96, 204], kneeF: [122, 171], ankleF: [126, 205],
          elbowR: [118, 104], wristR: [108, 100], racketTip: [84, 116],
          elbowL: [89, 113], wristL: [85, 128],
          ball: [252, 96],
        },
      },
    ],
  },
  {
    id: 'allarm',
    title: 'All arm, no turn',
    why: 'Swinging with only the arm caps your power at a fraction of what your body can produce, and it’s the #1 source of tennis elbow. The kinetic chain goes legs → hips → shoulders → arm; skip the first three links and the last one has to do everything. Fix: exaggerate the unit turn until your back almost faces the side fence.',
    phases: [
      { name: 'Ready', note: 'Everything starts the same…', pose: FOREHAND_PHASES[0].pose },
      {
        name: 'Arm-only take-back',
        note: 'The arm pulls the racket back but the shoulders and hips never turn — chest still faces the net.',
        pose: {
          head: [108, 78], neck: [108, 92], hip: [108, 141],
          kneeB: [96, 173], ankleB: [92, 205], kneeF: [122, 173], ankleF: [126, 205],
          elbowR: [98, 116], wristR: [82, 132], racketTip: [64, 120],
          elbowL: [92, 114], wristL: [98, 128],
          ball: [206, 122],
        },
      },
      {
        name: 'Shallow drop',
        note: 'Without a shoulder turn there’s nothing to uncoil — the racket drops only a little.',
        pose: {
          head: [108, 79], neck: [108, 93], hip: [108, 142],
          kneeB: [96, 174], ankleB: [92, 205], kneeF: [122, 174], ankleF: [126, 205],
          elbowR: [96, 122], wristR: [82, 140], racketTip: [66, 152],
          elbowL: [92, 114], wristL: [98, 128],
          ball: [182, 126],
        },
      },
      {
        name: 'Contact — weak',
        note: 'The arm alone pushes at the ball. Contact position looks okay, but there’s no body behind it.',
        pose: {
          head: [108, 77], neck: [108, 91], hip: [108, 141],
          kneeB: [96, 173], ankleB: [92, 204], kneeF: [122, 172], ankleF: [126, 205],
          elbowR: [122, 118], wristR: [142, 126], racketTip: [150, 94],
          elbowL: [90, 114], wristL: [96, 128],
          ball: [148, 102],
        },
      },
      {
        name: 'Push extension',
        note: 'The “swing” is really a push — the racket decelerates through contact instead of accelerating.',
        pose: {
          head: [108, 77], neck: [108, 91], hip: [108, 140],
          kneeB: [96, 172], ankleB: [92, 204], kneeF: [122, 171], ankleF: [126, 205],
          elbowR: [128, 110], wristR: [148, 110], racketTip: [176, 104],
          elbowL: [90, 114], wristL: [96, 128],
          ball: [220, 96],
        },
      },
      {
        name: 'Stiff finish',
        note: 'The follow-through is short and stiff; the elbow takes the strain the body should have absorbed.',
        pose: {
          head: [108, 77], neck: [108, 91], hip: [108, 140],
          kneeB: [96, 172], ankleB: [92, 204], kneeF: [122, 171], ankleF: [126, 205],
          elbowR: [124, 102], wristR: [118, 92], racketTip: [96, 100],
          elbowL: [90, 114], wristL: [94, 128],
          ball: [252, 88],
        },
      },
    ],
  },
]
