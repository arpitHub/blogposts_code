import { useMemo } from 'react';
import { BOROUGHS } from '../data/boroughs';

function StatCell({ label, value, accentClass }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col items-center gap-1 px-2">
      <span className="font-display text-[7px] uppercase tracking-widest text-text-muted sm:text-[8px]">
        {label}
      </span>
      <span
        className={`truncate font-mono text-sm font-medium tabular-nums sm:text-base ${accentClass}`}
      >
        {value}
      </span>
    </div>
  );
}

export default function StatsStrip({ sightings, casesSolved }) {
  const totalSightings = useMemo(
    () => sightings.reduce((sum, s) => sum + s.count, 0),
    [sightings]
  );

  const hottestZone = useMemo(() => {
    if (sightings.length === 0) return '—';
    const totals = sightings.reduce((acc, s) => {
      acc[s.borough] = (acc[s.borough] || 0) + s.count;
      return acc;
    }, {});
    const [topBorough] = Object.entries(totals).sort((a, b) => b[1] - a[1])[0];
    return BOROUGHS[topBorough]?.name || topBorough;
  }, [sightings]);

  return (
    <div className="flex items-stretch divide-x divide-grid border-y border-grid bg-navy/70 py-2">
      <StatCell
        label="Sightings"
        value={String(totalSightings).padStart(3, '0')}
        accentClass="text-marker-pink"
      />
      <StatCell
        label="Hottest Zone"
        value={hottestZone}
        accentClass="text-alert"
      />
      <StatCell
        label="Cases Solved"
        value={String(casesSolved).padStart(2, '0')}
        accentClass="text-marker-green"
      />
    </div>
  );
}
