// Serve keyframes (right-handed, side view, serving toward the right).

export const SERVE_PHASES = [
  {
    name: 'Stance',
    note: 'Feet still, weight settled: front foot angled toward the net post, back foot roughly parallel to the baseline. Racket and ball hand start together out front. The serve is a rhythm shot — it begins from stillness.',
    checkpoint: 'Shoulders sideways to the net, weight slightly on the front foot.',
    pose: {
      head: [104, 78], neck: [104, 92], hip: [106, 140],
      kneeB: [95, 173], ankleB: [90, 205], kneeF: [118, 173], ankleF: [122, 205],
      elbowR: [116, 116], wristR: [122, 136], racketTip: [136, 164],
      elbowL: [94, 118], wristL: [112, 138],
      ball: [116, 142],
    },
  },
  {
    name: 'Ball toss',
    note: 'Both arms move together: the ball arm rises straight up like a lift, releasing the ball at eye level — no wrist flick. Meanwhile the racket arm swings down and back. Knees begin to load.',
    checkpoint: 'Toss arm stays up after release, pointing at the ball.',
    pose: {
      head: [103, 78], neck: [103, 92], hip: [106, 141],
      kneeB: [93, 175], ankleB: [90, 205], kneeF: [117, 175], ankleF: [122, 205],
      elbowR: [92, 116], wristR: [80, 136], racketTip: [64, 158],
      elbowL: [110, 96], wristL: [116, 70],
      ball: [119, 48],
    },
  },
  {
    name: 'Trophy position',
    note: 'The classic pose on every tennis trophy: tossing arm extended up, racket arm bent about 90° with the racket head up, knees fully loaded. You are a drawn bow — everything from here is release.',
    checkpoint: 'If someone took a photo now, you’d look like the trophy.',
    pose: {
      head: [103, 76], neck: [103, 90], hip: [107, 142],
      kneeB: [92, 177], ankleB: [90, 205], kneeF: [116, 177], ankleF: [122, 205],
      elbowR: [88, 86], wristR: [84, 62], racketTip: [76, 32],
      elbowL: [111, 73], wristL: [118, 58],
      ball: [122, 30],
    },
  },
  {
    name: 'Racket drop',
    note: 'As the legs drive up, the racket head drops behind your back — the “back-scratch” position. This isn’t a pose you hold; it happens on its own when your arm stays loose. It’s the slingshot being stretched.',
    checkpoint: 'Racket head points at the ground behind you while your body rises.',
    pose: {
      head: [103, 72], neck: [103, 86], hip: [107, 136],
      kneeB: [94, 172], ankleB: [92, 203], kneeF: [117, 171], ankleF: [121, 204],
      elbowR: [90, 80], wristR: [94, 64], racketTip: [86, 96],
      elbowL: [104, 98], wristL: [112, 86],
      ball: [122, 26],
    },
  },
  {
    name: 'Contact',
    note: 'Full extension: legs straight, body stretched tall, arm reaching up and slightly into the court. Contact happens at the highest point you can comfortably reach — every centimeter higher buys you a safer angle over the net.',
    checkpoint: 'Hitting arm, shoulder, and hip form one straight line.',
    pose: {
      head: [106, 64], neck: [106, 78], hip: [110, 126],
      kneeB: [98, 162], ankleB: [96, 200], kneeF: [118, 162], ankleF: [120, 200],
      elbowR: [116, 56], wristR: [122, 34], racketTip: [130, 4],
      elbowL: [96, 100], wristL: [94, 114],
      ball: [128, 12],
    },
  },
  {
    name: 'Follow-through',
    note: 'The racket swings down and across to your opposite hip as you land inside the court on your front foot, back leg kicking back for balance. Let the momentum finish — cutting it short costs pace and strains the shoulder.',
    checkpoint: 'You land balanced inside the baseline, ready to move.',
    pose: {
      head: [110, 72], neck: [110, 86], hip: [112, 134],
      kneeB: [96, 168], ankleB: [86, 190], kneeF: [124, 170], ankleF: [130, 203],
      elbowR: [104, 110], wristR: [92, 130], racketTip: [76, 152],
      elbowL: [96, 110], wristL: [92, 126],
      ball: [256, 88],
    },
  },
]

export const SERVE_MISTAKES = [
  {
    id: 'toss',
    title: 'Toss drifts behind',
    why: 'A toss behind your head forces your back to arch and your swing to push the ball up instead of out — you lose all forward drive, and your lower back pays for it. Fix: toss with a straight arm “lift”, releasing at eye level, aiming for a spot inside the court at 1 o’clock.',
    phases: [
      { name: 'Stance', note: 'Everything starts the same…', pose: SERVE_PHASES[0].pose },
      {
        name: 'Flicked toss',
        note: 'The wrist flicks at release — the ball drifts back over the head instead of up and forward.',
        pose: { ...SERVE_PHASES[1].pose, wristL: [112, 72], elbowL: [107, 96], ball: [108, 52] },
      },
      {
        name: 'Trophy — ball behind',
        note: 'The trophy looks fine, but the ball is already hanging behind the head.',
        pose: { ...SERVE_PHASES[2].pose, ball: [100, 34] },
      },
      {
        name: 'Back arches',
        note: 'To reach the ball, the back has to arch and the hips push forward — the “banana” shape.',
        pose: {
          head: [96, 74], neck: [98, 88], hip: [106, 138],
          kneeB: [93, 173], ankleB: [91, 204], kneeF: [116, 172], ankleF: [121, 204],
          elbowR: [88, 82], wristR: [92, 66], racketTip: [84, 98],
          elbowL: [102, 100], wristL: [108, 88],
          ball: [97, 30],
        },
      },
      {
        name: 'Contact — overhead/behind',
        note: 'Contact is directly overhead or behind. The swing can only push up, not out into the court.',
        pose: {
          head: [98, 66], neck: [100, 80], hip: [108, 128],
          kneeB: [98, 163], ankleB: [96, 200], kneeF: [118, 163], ankleF: [120, 200],
          elbowR: [104, 58], wristR: [104, 36], racketTip: [102, 6],
          elbowL: [94, 102], wristL: [92, 116],
          ball: [101, 14],
        },
      },
      {
        name: 'Falling backward',
        note: 'Momentum goes backward — no weight into the court, and the back absorbs the strain.',
        pose: {
          head: [98, 76], neck: [99, 90], hip: [104, 136],
          kneeB: [92, 170], ankleB: [82, 194], kneeF: [116, 172], ankleF: [120, 204],
          elbowR: [96, 112], wristR: [86, 130], racketTip: [70, 150],
          elbowL: [92, 112], wristL: [88, 128],
          ball: [240, 100],
        },
      },
    ],
  },
  {
    id: 'nodrop',
    title: 'No racket drop',
    why: 'If the racket never drops behind your back, you’re serving with a push instead of a whip — the “slingshot” never gets stretched. This usually comes from gripping too tight and muscling the motion. Fix: loosen your grip to 3/10, pause in trophy, and let gravity start the drop before you swing up.',
    phases: [
      { name: 'Stance', note: 'Everything starts the same…', pose: SERVE_PHASES[0].pose },
      { name: 'Ball toss', note: 'The toss is fine…', pose: SERVE_PHASES[1].pose },
      { name: 'Trophy', note: 'Trophy position is fine too. The problem comes next.', pose: SERVE_PHASES[2].pose },
      {
        name: 'Racket stays up',
        note: 'The arm is stiff — the racket head never drops. The stretch that powers the serve is skipped.',
        pose: {
          head: [103, 72], neck: [103, 86], hip: [107, 136],
          kneeB: [94, 172], ankleB: [92, 203], kneeF: [117, 171], ankleF: [121, 204],
          elbowR: [88, 82], wristR: [84, 58], racketTip: [76, 28],
          elbowL: [104, 98], wristL: [112, 86],
          ball: [122, 26],
        },
      },
      {
        name: 'Contact — bent arm',
        note: 'The ball is met with a bent elbow and a stiff wrist: a push, not a throw. Pace tops out early.',
        pose: {
          head: [105, 68], neck: [105, 82], hip: [109, 128],
          kneeB: [98, 163], ankleB: [96, 200], kneeF: [118, 163], ankleF: [120, 200],
          elbowR: [112, 64], wristR: [117, 45], racketTip: [122, 15],
          elbowL: [96, 102], wristL: [94, 116],
          ball: [121, 24],
        },
      },
      {
        name: 'Short finish',
        note: 'With no whip there’s nothing to release — the follow-through dies early.',
        pose: {
          head: [108, 72], neck: [108, 86], hip: [111, 134],
          kneeB: [96, 168], ankleB: [90, 194], kneeF: [122, 170], ankleF: [127, 203],
          elbowR: [110, 104], wristR: [104, 120], racketTip: [94, 142],
          elbowL: [96, 110], wristL: [92, 126],
          ball: [246, 96],
        },
      },
    ],
  },
]
