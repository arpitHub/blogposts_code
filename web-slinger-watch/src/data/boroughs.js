// Abstracted, non-geographic borough silhouettes for a 1200x800 viewBox.
// These are original stylized shapes — not real geodata.

function pointsToPath(points) {
  return (
    points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x},${y}`).join(' ') +
    ' Z'
  );
}

function boundingBox(points) {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

const manhattanPoints = [
  [560, 110],
  [610, 100],
  [640, 160],
  [660, 260],
  [655, 360],
  [630, 440],
  [590, 500],
  [555, 520],
  [530, 480],
  [520, 400],
  [525, 300],
  [535, 200],
  [545, 140],
];

const brooklynPoints = [
  [650, 520],
  [720, 490],
  [800, 500],
  [860, 540],
  [900, 600],
  [910, 670],
  [880, 730],
  [800, 760],
  [720, 750],
  [670, 700],
  [645, 630],
  [640, 570],
];

const queensPoints = [
  [750, 180],
  [850, 140],
  [950, 150],
  [1050, 190],
  [1110, 260],
  [1130, 350],
  [1110, 440],
  [1050, 500],
  [960, 520],
  [880, 500],
  [820, 460],
  [780, 400],
  [760, 320],
  [745, 240],
];

export const BOROUGHS = {
  manhattan: {
    id: 'manhattan',
    name: 'Manhattan',
    points: manhattanPoints,
    path: pointsToPath(manhattanPoints),
    bbox: boundingBox(manhattanPoints),
  },
  brooklyn: {
    id: 'brooklyn',
    name: 'Brooklyn',
    points: brooklynPoints,
    path: pointsToPath(brooklynPoints),
    bbox: boundingBox(brooklynPoints),
  },
  queens: {
    id: 'queens',
    name: 'Queens',
    points: queensPoints,
    path: pointsToPath(queensPoints),
    bbox: boundingBox(queensPoints),
  },
};

export const BOROUGH_LIST = Object.values(BOROUGHS);

// Ray-casting point-in-polygon test.
export function isPointInBorough(borough, x, y) {
  const { points } = borough;
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i];
    const [xj, yj] = points[j];
    const intersects =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

// Rejection-samples a point inside the borough's silhouette.
export function randomPointInBorough(borough) {
  const { minX, maxX, minY, maxY } = borough.bbox;
  for (let attempt = 0; attempt < 200; attempt++) {
    const x = minX + Math.random() * (maxX - minX);
    const y = minY + Math.random() * (maxY - minY);
    if (isPointInBorough(borough, x, y)) {
      return { x, y };
    }
  }
  // Fallback: bbox center, in case of a pathological polygon.
  return { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
}

export function randomBorough() {
  return BOROUGH_LIST[Math.floor(Math.random() * BOROUGH_LIST.length)];
}
