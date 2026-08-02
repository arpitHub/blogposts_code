import { AnimatePresence, motion } from 'framer-motion';
import { BOROUGHS } from '../data/boroughs';

const BOROUGH_TAG_CLASS = {
  manhattan: 'text-marker-pink',
  brooklyn: 'text-marker-green',
  queens: 'text-marker-pink',
};

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString('en-US', { hour12: false });
}

function CaseBadge({ solved }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-sm px-1 py-[1px] font-display text-[6px] uppercase tracking-widest ${
        solved
          ? 'bg-marker-green/20 text-marker-green'
          : 'bg-text-muted/20 text-text-muted'
      }`}
    >
      <svg viewBox="0 0 8 8" className="h-2 w-2" fill="currentColor">
        {solved ? (
          <path d="M1 4 L3 6 L7 1 L6 0 L3 4 L2 3 Z" />
        ) : (
          <path d="M1 1 L2 0 L4 2 L6 0 L7 1 L5 3 L7 5 L6 6 L4 4 L2 6 L1 5 L3 3 Z" />
        )}
      </svg>
      {solved ? 'Case Solved' : 'Signal Lost'}
    </span>
  );
}

export default function Ticker({ entries }) {
  return (
    <div className="flex h-full flex-col overflow-hidden">
      <h2 className="border-b border-grid px-3 py-2 font-display text-[10px] uppercase tracking-widest text-teal">
        Sighting Log
      </h2>
      <ul className="flex-1 space-y-2 overflow-y-auto px-3 py-2">
        {entries.length === 0 && (
          <li className="font-body text-xs text-text-muted">
            No sightings logged yet. Stand by...
          </li>
        )}
        <AnimatePresence initial={false}>
          {entries.map((entry) => {
            const isCase = entry.kind === 'case-solved' || entry.kind === 'case-missed';
            return (
              <motion.li
                key={entry.logId}
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className={`border-b border-grid/60 pb-2 last:border-none ${
                  isCase ? 'border-l-2 border-l-teal/50 pl-2' : ''
                }`}
              >
                <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] text-text-muted">
                  <span>{formatTime(entry.timestamp)}</span>
                  {entry.borough && (
                    <span
                      className={`uppercase ${
                        BOROUGH_TAG_CLASS[entry.borough] || 'text-marker-pink'
                      }`}
                    >
                      {BOROUGHS[entry.borough]?.name || entry.borough}
                    </span>
                  )}
                  {isCase && <CaseBadge solved={entry.kind === 'case-solved'} />}
                </div>
                <p className="font-body text-xs text-text-primary">
                  {entry.flavorText}
                </p>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </ul>
    </div>
  );
}
