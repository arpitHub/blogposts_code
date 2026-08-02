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
          {entries.map((entry) => (
            <motion.li
              key={entry.logId}
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="border-b border-grid/60 pb-2 last:border-none"
            >
              <div className="flex items-center gap-2 font-mono text-[10px] text-text-muted">
                <span>{formatTime(entry.timestamp)}</span>
                <span
                  className={`uppercase ${
                    BOROUGH_TAG_CLASS[entry.borough] || 'text-marker-pink'
                  }`}
                >
                  {BOROUGHS[entry.borough]?.name || entry.borough}
                </span>
              </div>
              <p className="font-body text-xs text-text-primary">
                {entry.flavorText}
              </p>
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </div>
  );
}
