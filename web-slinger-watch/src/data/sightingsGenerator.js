import { randomBorough, randomPointInBorough } from './boroughs';
import { generateFlavorText } from './flavorText';

export const MIN_SPAWN_INTERVAL_MS = 4000;
export const MAX_SPAWN_INTERVAL_MS = 9000;
export const CLUSTER_THRESHOLD_PX = 40;
export const ALERT_DISMISS_MS = 4000;
export const TICKER_MAX_ENTRIES = 20;

export function getRandomSpawnDelay() {
  return (
    MIN_SPAWN_INTERVAL_MS +
    Math.random() * (MAX_SPAWN_INTERVAL_MS - MIN_SPAWN_INTERVAL_MS)
  );
}

function distance(ax, ay, bx, by) {
  return Math.hypot(ax - bx, ay - by);
}

function makeId() {
  return `sighting-${Date.now()}-${Math.floor(Math.random() * 100000)}`;
}

// Finds an existing sighting close enough to (x, y) to absorb a new one
// into the same cluster marker.
function findNearbySighting(existingSightings, x, y) {
  return existingSightings.find(
    (s) => distance(s.x, s.y, x, y) <= CLUSTER_THRESHOLD_PX
  );
}

// Either bumps an existing cluster's count or creates a brand-new sighting.
// Returns { sighting, isNew } where `sighting` is the object to upsert into
// the caller's sightings array (matched by `id`).
export function generateSighting(existingSightings) {
  const borough = randomBorough();
  const { x, y } = randomPointInBorough(borough);
  const flavorText = generateFlavorText(borough.id);
  const timestamp = Date.now();

  const nearby = findNearbySighting(existingSightings, x, y);

  if (nearby) {
    return {
      isNew: false,
      sighting: {
        ...nearby,
        count: nearby.count + 1,
        timestamp,
        flavorText,
      },
    };
  }

  return {
    isNew: true,
    sighting: {
      id: makeId(),
      x,
      y,
      borough: borough.id,
      count: 1,
      timestamp,
      flavorText,
    },
  };
}
