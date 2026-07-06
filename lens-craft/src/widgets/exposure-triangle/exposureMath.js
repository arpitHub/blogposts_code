// Real camera full-stop values. Index position in each array = "stops" apart,
// so the maths below is just index arithmetic (2^stops = brightness multiplier).

export const APERTURE_STOPS = [1.4, 2, 2.8, 4, 5.6, 8, 11, 16, 22];
export const SHUTTER_STOPS = [
  { label: '1/4000', seconds: 1 / 4000 },
  { label: '1/2000', seconds: 1 / 2000 },
  { label: '1/1000', seconds: 1 / 1000 },
  { label: '1/500', seconds: 1 / 500 },
  { label: '1/250', seconds: 1 / 250 },
  { label: '1/125', seconds: 1 / 125 },
  { label: '1/60', seconds: 1 / 60 },
  { label: '1/30', seconds: 1 / 30 },
  { label: '1/15', seconds: 1 / 15 },
  { label: '1/8', seconds: 1 / 8 },
  { label: '1/4', seconds: 1 / 4 },
  { label: '1/2', seconds: 1 / 2 },
  { label: '1"', seconds: 1 },
];
export const ISO_STOPS = [100, 200, 400, 800, 1600, 3200, 6400, 12800];

// "Correctly exposed" reference combo for our demo scene.
export const BASELINE = {
  apertureIndex: 4, // f/5.6
  shutterIndex: 5, // 1/125
  isoIndex: 0, // ISO 100
};

export function formatAperture(index) {
  return `f/${APERTURE_STOPS[index]}`;
}

export function formatShutter(index) {
  return SHUTTER_STOPS[index].label + 's';
}

export function formatIso(index) {
  return `ISO ${ISO_STOPS[index]}`;
}

// Positive stopsFromBaseline = brighter than reference, negative = darker.
export function computeStops({ apertureIndex, shutterIndex, isoIndex }) {
  const apertureStops = BASELINE.apertureIndex - apertureIndex; // smaller opening (higher index) = less light
  const shutterStops = shutterIndex - BASELINE.shutterIndex; // longer time (higher index) = more light
  const isoStops = isoIndex - BASELINE.isoIndex; // higher ISO = more (amplified) light
  return {
    apertureStops,
    shutterStops,
    isoStops,
    totalStops: apertureStops + shutterStops + isoStops,
  };
}

export function brightnessMultiplier(totalStops) {
  return Math.pow(2, totalStops);
}

// 0 (wide open, f/1.4) .. 1 (fully stopped down, f/22) -> background blur amount
export function apertureToBlurAmount(apertureIndex) {
  const t = apertureIndex / (APERTURE_STOPS.length - 1);
  return 1 - t;
}

// 0 (1/4000, frozen) .. 1 (1", maximum blur) on a log scale of actual seconds
export function shutterToMotionAmount(shutterIndex) {
  const seconds = SHUTTER_STOPS[shutterIndex].seconds;
  const minLog = Math.log2(SHUTTER_STOPS[0].seconds);
  const maxLog = Math.log2(SHUTTER_STOPS[SHUTTER_STOPS.length - 1].seconds);
  return (Math.log2(seconds) - minLog) / (maxLog - minLog);
}

// 0 (ISO 100, clean) .. 1 (ISO 12800, heavy grain)
export function isoToNoiseAmount(isoIndex) {
  return isoIndex / (ISO_STOPS.length - 1);
}

export function exposureLabel(totalStops) {
  if (totalStops <= -2) return { text: 'Underexposed', color: '#5aa9ff' };
  if (totalStops >= 2) return { text: 'Overexposed', color: '#ff5a4d' };
  if (Math.abs(totalStops) <= 0.4) return { text: 'Well exposed', color: '#52c97a' };
  return { text: totalStops < 0 ? 'Slightly dark' : 'Slightly bright', color: '#ffb020' };
}
