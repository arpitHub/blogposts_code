// Central registry of every module in Lens Craft.
// `order` drives the linear "Start Here" beginner track.
// `component` is set later (in pages/ModuleRoute.jsx) for modules that have a real widget;
// everything else falls back to the placeholder panel.

export const LEVELS = {
  beginner: { label: 'Beginner', color: 'var(--color-level-beginner)' },
  intermediate: { label: 'Intermediate', color: 'var(--color-level-intermediate)' },
  advanced: { label: 'Advanced', color: 'var(--color-level-advanced)' },
};

export const MODULES = [
  {
    slug: 'exposure-triangle',
    order: 1,
    title: 'The Exposure Triangle',
    tagline: 'Aperture, shutter speed, and ISO — one photo, three dials.',
    level: 'beginner',
    ready: true,
  },
  {
    slug: 'aperture-depth-of-field',
    order: 2,
    title: 'Aperture & Depth of Field',
    tagline: 'Why f/1.4 melts the background and f/16 keeps it all sharp.',
    level: 'beginner',
    ready: true,
  },
  {
    slug: 'shutter-speed-motion',
    order: 3,
    title: 'Shutter Speed & Motion',
    tagline: 'Freeze a splash or silk-smooth a waterfall — same water, different time.',
    level: 'beginner',
    ready: true,
  },
  {
    slug: 'iso-noise',
    order: 4,
    title: 'ISO & Noise',
    tagline: 'How much grain are you willing to trade for a usable shutter speed?',
    level: 'beginner',
    ready: true,
  },
  {
    slug: 'histogram-reading',
    order: 5,
    title: 'Reading the Histogram',
    tagline: "Your eyes lie in bright sunlight. The histogram doesn't.",
    level: 'beginner',
    ready: true,
  },
  {
    slug: 'composition',
    order: 6,
    title: 'Composition',
    tagline: 'Rule of thirds, leading lines, and symmetry — drag the grid, feel the difference.',
    level: 'beginner',
    ready: true,
  },
  {
    slug: 'light-and-direction',
    order: 7,
    title: 'Light & Direction',
    tagline: 'Move the light around a face and watch mood appear out of nowhere.',
    level: 'intermediate',
    ready: true,
  },
  {
    slug: 'white-balance',
    order: 8,
    title: 'White Balance & Color Temperature',
    tagline: 'Kelvin is just a dial between candlelight-orange and overcast-blue.',
    level: 'intermediate',
    ready: true,
  },
  {
    slug: 'focal-length',
    order: 9,
    title: 'Focal Length & Lens Choice',
    tagline: '16mm to 400mm: field of view, compression, and why backgrounds "come closer."',
    level: 'intermediate',
    ready: true,
  },
  {
    slug: 'focusing-modes',
    order: 10,
    title: 'Focusing Modes & AF Points',
    tagline: 'Single-shot vs. continuous, and why the AF point you pick matters.',
    level: 'intermediate',
    ready: true,
  },
  {
    slug: 'raw-vs-jpeg',
    order: 11,
    title: 'RAW vs. JPEG',
    tagline: 'Same sensor data, wildly different room to fix your mistakes later.',
    level: 'intermediate',
    ready: true,
  },
  {
    slug: 'post-processing-basics',
    order: 12,
    title: 'Post-Processing Basics',
    tagline: 'Exposure, contrast, and curves — the three edits that do 80% of the work.',
    level: 'intermediate',
    ready: true,
  },
  {
    slug: 'genre-guides',
    order: 13,
    title: 'Genre Guides',
    tagline: 'Portrait, landscape, astro — starting settings and the reasoning behind them.',
    level: 'advanced',
    ready: true,
  },
  {
    slug: 'gear-explainer',
    order: 14,
    title: 'Gear: Sensor Size & Crop Factor',
    tagline: 'Full-frame, APS-C, Micro Four Thirds — what actually changes, visually.',
    level: 'advanced',
    ready: true,
  },
];

export function getModule(slug) {
  return MODULES.find((m) => m.slug === slug);
}

export function getNextModule(slug) {
  const current = getModule(slug);
  if (!current) return undefined;
  return MODULES.find((m) => m.order === current.order + 1);
}

export function modulesByLevel() {
  return ['beginner', 'intermediate', 'advanced'].map((level) => ({
    level,
    ...LEVELS[level],
    modules: MODULES.filter((m) => m.level === level).sort((a, b) => a.order - b.order),
  }));
}
