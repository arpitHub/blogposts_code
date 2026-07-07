// Side-view diagram: camera at left, subject in the middle, and a shaded
// "in focus" band that widens as the aperture closes down.
export default function DofDiagram({ fNumber }) {
  const W = 440;
  const H = 90;
  const subjectX = W * 0.48;

  // band half-width grows roughly linearly with f-number; far limit grows faster
  // than near limit, like real DOF (about 1/3 in front, 2/3 behind).
  const t = (fNumber - 1.4) / (22 - 1.4);
  const near = 12 + t * 70;
  const far = 16 + t * 190;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {/* distance axis */}
      <line x1="30" y1={H - 18} x2={W - 8} y2={H - 18} stroke="var(--color-line)" strokeWidth="1" />
      <text x={W - 8} y={H - 6} textAnchor="end" fontSize="9" fill="var(--color-ink-3)" fontFamily="var(--font-mono)">
        distance →
      </text>

      {/* camera */}
      <rect x="8" y={H / 2 - 14} width="20" height="16" rx="3" fill="var(--color-panel-3)" stroke="var(--color-ink-3)" />
      <circle cx="28" cy={H / 2 - 6} r="5" fill="var(--color-panel)" stroke="var(--color-ink-3)" />

      {/* in-focus band */}
      <rect
        x={subjectX - near}
        y="12"
        width={near + far}
        height={H - 36}
        rx="4"
        fill="rgba(82,201,122,0.13)"
        stroke="rgba(82,201,122,0.55)"
        strokeDasharray="3 3"
      />
      <text
        x={subjectX + (far - near) / 2}
        y="9"
        textAnchor="middle"
        fontSize="9"
        fill="var(--color-level-beginner)"
        fontFamily="var(--font-mono)"
      >
        in focus
      </text>

      {/* subject marker */}
      <line x1={subjectX} y1="14" x2={subjectX} y2={H - 22} stroke="var(--color-accent)" strokeWidth="2" />
      <circle cx={subjectX} cy={H / 2 - 6} r="4" fill="var(--color-accent)" />
      <text x={subjectX} y={H - 6} textAnchor="middle" fontSize="9" fill="var(--color-accent)" fontFamily="var(--font-mono)">
        subject
      </text>
    </svg>
  );
}
