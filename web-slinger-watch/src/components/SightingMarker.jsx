import { motion } from 'framer-motion';

const BOROUGH_FILL_CLASS = {
  manhattan: 'fill-marker-pink',
  brooklyn: 'fill-marker-green',
  queens: 'fill-marker-pink',
};

const BOROUGH_GLOW_COLOR = {
  manhattan: '#ff6b8a',
  brooklyn: '#4ade80',
  queens: '#ff6b8a',
};

// Small original geometric spider glyph — a body, a head, and eight
// radiating legs. Not a reproduction of any copyrighted mark.
function SpiderGlyph({ stroke }) {
  return (
    <g stroke={stroke} strokeWidth={1.2} strokeLinecap="round" fill="none">
      <circle cx={0} cy={-1} r={2.2} fill={stroke} stroke="none" />
      <circle cx={0} cy={2.5} r={3.2} fill={stroke} stroke="none" />
      <line x1={-2} y1={0} x2={-8} y2={-5} />
      <line x1={-2} y1={1} x2={-9} y2={0} />
      <line x1={-2} y1={2} x2={-8} y2={5} />
      <line x1={-2} y1={3} x2={-7} y2={8} />
      <line x1={2} y1={0} x2={8} y2={-5} />
      <line x1={2} y1={1} x2={9} y2={0} />
      <line x1={2} y1={2} x2={8} y2={5} />
      <line x1={2} y1={3} x2={7} y2={8} />
    </g>
  );
}

export default function SightingMarker({ sighting }) {
  const { x, y, borough, count } = sighting;
  const fillClass = BOROUGH_FILL_CLASS[borough] || 'fill-marker-pink';
  const glowColor = BOROUGH_GLOW_COLOR[borough] || '#ff6b8a';
  const radius = Math.min(14 + (count - 1) * 1.5, 22);

  return (
    <motion.g
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 300, damping: 15 }}
      style={{ transformOrigin: `${x}px ${y}px` }}
    >
      <circle cx={x} cy={y} r={radius + 6} fill={glowColor} opacity={0.15} />
      <circle
        cx={x}
        cy={y}
        r={radius}
        className={fillClass}
        stroke="#0a1628"
        strokeWidth={2}
      />
      <g transform={`translate(${x}, ${y})`}>
        <SpiderGlyph stroke="#0a1628" />
      </g>
      {count > 1 && (
        <motion.g
          key={count}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 400, damping: 12 }}
          style={{ transformOrigin: `${x + radius}px ${y - radius}px` }}
        >
          <circle
            cx={x + radius}
            cy={y - radius}
            r={9}
            fill="#f5b878"
            stroke="#0a1628"
            strokeWidth={1.5}
          />
          <text
            x={x + radius}
            y={y - radius + 3.5}
            textAnchor="middle"
            fontSize={9}
            className="font-mono"
            fill="#0a1628"
          >
            {count}
          </text>
        </motion.g>
      )}
    </motion.g>
  );
}
