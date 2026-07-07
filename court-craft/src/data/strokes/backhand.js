// Backhand keyframes — one-handed and two-handed variants share the same
// six-phase structure so a single scrubber can drive both.

export const BACKHAND_1H = [
  {
    name: 'Ready',
    note: 'Same athletic base as always. The non-hitting hand cradles the racket throat — on the backhand side it will do the take-back for you.',
    checkpoint: 'Grip change happens here: rotate to an Eastern backhand grip.',
    pose: {
      head: [108, 78], neck: [108, 92], hip: [108, 140],
      kneeB: [96, 172], ankleB: [92, 205], kneeF: [122, 172], ankleF: [126, 205],
      elbowR: [126, 112], wristR: [134, 128], racketTip: [152, 112],
      elbowL: [92, 112], wristL: [126, 122],
      ball: [232, 118],
    },
  },
  {
    name: 'Unit turn',
    note: 'Shoulders turn until your back almost faces the net — more turn than the forehand needs. The left hand pulls the racket back; the hitting arm just comes along.',
    checkpoint: 'Chin over your front shoulder, watching the ball.',
    pose: {
      head: [104, 78], neck: [104, 92], hip: [106, 141],
      kneeB: [94, 173], ankleB: [90, 205], kneeF: [120, 173], ankleF: [124, 205],
      elbowR: [92, 112], wristR: [76, 118], racketTip: [56, 104],
      elbowL: [96, 116], wristL: [82, 122],
      ball: [206, 122],
    },
  },
  {
    name: 'Drop & separate',
    note: 'The racket drops below the ball line and the hands separate — the left hand releases and sweeps back for balance. Knees loaded.',
    checkpoint: 'Racket head below your hands, hitting arm still relaxed.',
    pose: {
      head: [104, 80], neck: [104, 94], hip: [106, 144],
      kneeB: [92, 175], ankleB: [88, 205], kneeF: [120, 175], ankleF: [124, 205],
      elbowR: [88, 130], wristR: [72, 146], racketTip: [52, 162],
      elbowL: [94, 120], wristL: [80, 128],
      ball: [182, 126],
    },
  },
  {
    name: 'Contact — way out front',
    note: 'The one-hander’s golden rule: contact happens further in front than any other groundstroke, with the arm fully straight. Meanwhile the free arm stretches backward — that counterbalance is what keeps the shoulders closed.',
    checkpoint: 'Straight arm, ball ahead of your front foot, free arm back.',
    pose: {
      head: [106, 76], neck: [106, 90], hip: [108, 140],
      kneeB: [96, 173], ankleB: [92, 203], kneeF: [122, 172], ankleF: [126, 205],
      elbowR: [128, 108], wristR: [148, 112], racketTip: [158, 82],
      elbowL: [88, 106], wristL: [70, 100],
      ball: [154, 88],
    },
  },
  {
    name: 'Extension',
    note: 'Both arms open like wings — racket extending toward the target, free arm reaching back. If your free arm chases the racket, your shoulders fly open and the ball sprays wide.',
    checkpoint: 'Feel the stretch across your chest.',
    pose: {
      head: [107, 76], neck: [107, 90], hip: [109, 139],
      kneeB: [96, 172], ankleB: [93, 203], kneeF: [122, 171], ankleF: [126, 205],
      elbowR: [136, 100], wristR: [160, 98], racketTip: [190, 84],
      elbowL: [86, 104], wristL: [66, 96],
      ball: [226, 74],
    },
  },
  {
    name: 'High finish',
    note: 'The racket finishes high and in front, edge to the sky. Hold it for a beat — the classic one-hander pose. Then recover.',
    checkpoint: 'Racket head above shoulder height, body still sideways-ish.',
    pose: {
      head: [107, 76], neck: [107, 90], hip: [109, 139],
      kneeB: [98, 172], ankleB: [96, 203], kneeF: [122, 171], ankleF: [126, 205],
      elbowR: [128, 86], wristR: [150, 70], racketTip: [176, 46],
      elbowL: [86, 106], wristL: [68, 102],
      ball: [258, 58],
    },
  },
]

export const BACKHAND_2H = [
  {
    name: 'Ready',
    note: 'Both hands already on the grip: hitting hand at the bottom (continental-ish), other hand above it (like its own little forehand grip). No grip change needed under pressure.',
    checkpoint: 'Hands touching, relaxed, racket centered.',
    pose: {
      head: [108, 78], neck: [108, 92], hip: [108, 140],
      kneeB: [96, 172], ankleB: [92, 205], kneeF: [122, 172], ankleF: [126, 205],
      elbowR: [124, 112], wristR: [132, 128], racketTip: [150, 112],
      elbowL: [112, 110], wristL: [128, 122],
      ball: [232, 118],
    },
  },
  {
    name: 'Unit turn',
    note: 'Shoulders and hips coil together, both hands taking the racket back as one piece. The two-hander’s take-back is naturally more compact — that’s why it handles fast balls so well.',
    checkpoint: 'Both hands still on the racket, shoulders fully turned.',
    pose: {
      head: [104, 78], neck: [104, 92], hip: [106, 141],
      kneeB: [94, 173], ankleB: [90, 205], kneeF: [120, 173], ankleF: [124, 205],
      elbowR: [94, 114], wristR: [78, 122], racketTip: [58, 106],
      elbowL: [98, 110], wristL: [84, 116],
      ball: [206, 122],
    },
  },
  {
    name: 'Drop',
    note: 'Both hands drop the racket head below the ball. Think of it as a left-handed forehand drop with the right hand along for the ride.',
    checkpoint: 'Racket head below wrists, knees loaded.',
    pose: {
      head: [104, 80], neck: [104, 94], hip: [106, 144],
      kneeB: [92, 175], ankleB: [88, 205], kneeF: [120, 175], ankleF: [124, 205],
      elbowR: [92, 132], wristR: [74, 148], racketTip: [56, 162],
      elbowL: [96, 126], wristL: [80, 142],
      ball: [182, 126],
    },
  },
  {
    name: 'Contact — closer in',
    note: 'Contact sits closer to the body and can happen a touch later than the one-hander — the second hand powers through even when you’re jammed. This forgiveness is the two-hander’s superpower.',
    checkpoint: 'Both elbows slightly bent, ball in front of the hip.',
    pose: {
      head: [106, 76], neck: [106, 90], hip: [108, 140],
      kneeB: [96, 173], ankleB: [92, 203], kneeF: [122, 172], ankleF: [126, 205],
      elbowR: [120, 116], wristR: [138, 116], racketTip: [148, 86],
      elbowL: [112, 110], wristL: [132, 110],
      ball: [144, 90],
    },
  },
  {
    name: 'Extension',
    note: 'The non-dominant arm drives the racket through the ball — coaches say the two-hander is “a left-handed forehand” for a righty. Both hands stay on.',
    checkpoint: 'Racket face stays on the target line as long as possible.',
    pose: {
      head: [107, 76], neck: [107, 90], hip: [109, 139],
      kneeB: [96, 172], ankleB: [93, 203], kneeF: [122, 171], ankleF: [126, 205],
      elbowR: [130, 108], wristR: [152, 102], racketTip: [178, 88],
      elbowL: [120, 102], wristL: [146, 96],
      ball: [222, 76],
    },
  },
  {
    name: 'Wrap finish',
    note: 'The racket wraps over the hitting shoulder, both hands still attached, hips fully open to the net. More rotational than the one-hander’s frozen high finish.',
    checkpoint: 'Belt buckle faces the net; racket over your shoulder.',
    pose: {
      head: [107, 76], neck: [107, 90], hip: [109, 139],
      kneeB: [98, 172], ankleB: [96, 203], kneeF: [122, 171], ankleF: [126, 205],
      elbowR: [120, 96], wristR: [114, 82], racketTip: [90, 68],
      elbowL: [112, 92], wristL: [108, 80],
      ball: [256, 60],
    },
  },
]

export const BACKHAND_DIFFERENCES = [
  {
    dim: 'Contact point',
    oneH: 'Further out front; arm fully straight — unforgiving if you’re late.',
    twoH: 'Closer to the body; the second hand rescues late contact.',
  },
  {
    dim: 'Reach & slice',
    oneH: 'Longer reach, and the grip flows naturally into a beautiful slice.',
    twoH: 'Shorter reach — wide balls often force a one-handed slice anyway.',
  },
  {
    dim: 'High balls',
    oneH: 'Shoulder-height contact is the one-hander’s weak spot.',
    twoH: 'The extra hand muscles through high, heavy topspin comfortably.',
  },
  {
    dim: 'Return of serve',
    oneH: 'Needs more preparation time — big serves rush it.',
    twoH: 'Compact and stable — the tour’s default for a reason.',
  },
  {
    dim: 'Learning curve',
    oneH: 'Harder to time early on; pays off in versatility and style.',
    twoH: 'Quicker to reach reliability, especially for juniors and beginners.',
  },
]
