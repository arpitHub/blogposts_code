import { useMemo } from 'react';
import { BOROUGH_LIST } from '../data/boroughs';
import SightingMarker from './SightingMarker';

export const VIEW_WIDTH = 1200;
export const VIEW_HEIGHT = 800;
const GRID_SPACING = 60;
const MICRO_GRID_SPACING = 16;
const TRAIL_LENGTH = 5;

function buildGridLines() {
  const lines = [];
  for (let x = GRID_SPACING; x < VIEW_WIDTH; x += GRID_SPACING) {
    lines.push({ key: `v-${x}`, x1: x, y1: 0, x2: x, y2: VIEW_HEIGHT });
  }
  for (let y = GRID_SPACING; y < VIEW_HEIGHT; y += GRID_SPACING) {
    lines.push({ key: `h-${y}`, x1: 0, y1: y, x2: VIEW_WIDTH, y2: y });
  }
  return lines;
}

// Deterministic pseudo-random so the micro-grid stays stable across renders.
function makeRng(seed) {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

// Denser, deliberately irregular street lines drawn inside each borough.
// Clipped to the silhouette at render time.
function buildMicroGrid(borough, seedOffset) {
  const rng = makeRng(1337 + seedOffset * 7919);
  const { minX, maxX, minY, maxY } = borough.bbox;
  const lines = [];

  for (let x = minX; x <= maxX; x += MICRO_GRID_SPACING) {
    const jitter = (rng() - 0.5) * MICRO_GRID_SPACING * 0.7;
    const skew = (rng() - 0.5) * 14;
    lines.push({
      key: `${borough.id}-mv-${x}`,
      x1: x + jitter,
      y1: minY,
      x2: x + jitter + skew,
      y2: maxY,
      opacity: 0.25 + rng() * 0.35,
    });
  }

  for (let y = minY; y <= maxY; y += MICRO_GRID_SPACING) {
    const jitter = (rng() - 0.5) * MICRO_GRID_SPACING * 0.7;
    const skew = (rng() - 0.5) * 14;
    lines.push({
      key: `${borough.id}-mh-${y}`,
      x1: minX,
      y1: y + jitter,
      x2: maxX,
      y2: y + jitter + skew,
      opacity: 0.25 + rng() * 0.35,
    });
  }

  return lines;
}

const GRID_LINES = buildGridLines();
const MICRO_GRIDS = BOROUGH_LIST.map((borough, i) => ({
  borough,
  lines: buildMicroGrid(borough, i),
}));

export default function MapCanvas({ sightings, highlightBorough }) {
  // Newest-last chronological path through the most recent sightings.
  const trailPoints = useMemo(() => {
    return [...sightings]
      .sort((a, b) => a.timestamp - b.timestamp)
      .slice(-TRAIL_LENGTH);
  }, [sightings]);

  return (
    <svg
      viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
      className="h-full w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <radialGradient id="boroughFill" cx="50%" cy="45%" r="72%">
          <stop offset="0%" stopColor="#081221" />
          <stop offset="100%" stopColor="#16293f" />
        </radialGradient>
        <radialGradient id="boroughFillHot" cx="50%" cy="45%" r="72%">
          <stop offset="0%" stopColor="#12283f" />
          <stop offset="100%" stopColor="#1d3a52" />
        </radialGradient>
        {BOROUGH_LIST.map((borough) => (
          <clipPath key={`clip-${borough.id}`} id={`clip-${borough.id}`}>
            <path d={borough.path} />
          </clipPath>
        ))}
      </defs>

      <rect x={0} y={0} width={VIEW_WIDTH} height={VIEW_HEIGHT} fill="#0a1628" />

      <g stroke="#1e3a5f" strokeWidth={1} opacity={0.45}>
        {GRID_LINES.map((line) => (
          <line
            key={line.key}
            x1={line.x1}
            y1={line.y1}
            x2={line.x2}
            y2={line.y2}
          />
        ))}
      </g>

      <g>
        {MICRO_GRIDS.map(({ borough, lines }) => {
          const isHot = highlightBorough === borough.id;
          return (
            <g key={borough.id}>
              <path
                d={borough.path}
                fill={`url(#${isHot ? 'boroughFillHot' : 'boroughFill'})`}
                stroke="#2dd4bf"
                strokeWidth={isHot ? 3 : 2}
                opacity={0.95}
                className="transition-all duration-500"
              />
              <g clipPath={`url(#clip-${borough.id})`} stroke="#173049">
                {lines.map((line) => (
                  <line
                    key={line.key}
                    x1={line.x1}
                    y1={line.y1}
                    x2={line.x2}
                    y2={line.y2}
                    strokeWidth={0.6}
                    opacity={line.opacity}
                  />
                ))}
              </g>
              <text
                x={borough.bbox.minX}
                y={borough.bbox.minY - 12}
                className="font-mono uppercase tracking-widest"
                fontSize={12}
                fill={isHot ? '#2dd4bf' : '#7d97ad'}
              >
                {borough.name}
              </text>
            </g>
          );
        })}
      </g>

      {/* Dashed trail through recent sightings; older segments fade out. */}
      <g fill="none" stroke="#2dd4bf" strokeDasharray="4 6" strokeLinecap="round">
        {trailPoints.slice(0, -1).map((point, i) => {
          const next = trailPoints[i + 1];
          const freshness = (i + 1) / (trailPoints.length - 1);
          return (
            <line
              key={`trail-${point.id}-${next.id}`}
              x1={point.x}
              y1={point.y}
              x2={next.x}
              y2={next.y}
              strokeWidth={1.2}
              opacity={0.08 + freshness * 0.32}
            />
          );
        })}
      </g>

      <g>
        {sightings.map((sighting) => (
          <SightingMarker key={sighting.id} sighting={sighting} />
        ))}
      </g>
    </svg>
  );
}
