import { BOROUGH_LIST } from '../data/boroughs';
import SightingMarker from './SightingMarker';

export const VIEW_WIDTH = 1200;
export const VIEW_HEIGHT = 800;
const GRID_SPACING = 60;

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

const GRID_LINES = buildGridLines();

export default function MapCanvas({ sightings }) {
  return (
    <svg
      viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
      className="h-full w-full"
      preserveAspectRatio="xMidYMid meet"
    >
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
        {BOROUGH_LIST.map((borough) => (
          <g key={borough.id}>
            <path
              d={borough.path}
              fill="#12233d"
              stroke="#2dd4bf"
              strokeWidth={2}
              opacity={0.9}
            />
            <text
              x={borough.bbox.minX}
              y={borough.bbox.minY - 12}
              className="font-mono uppercase tracking-widest"
              fontSize={12}
              fill="#7d97ad"
            >
              {borough.name}
            </text>
          </g>
        ))}
      </g>

      <g>
        {sightings.map((sighting) => (
          <SightingMarker key={sighting.id} sighting={sighting} />
        ))}
      </g>
    </svg>
  );
}
