import { AnimatePresence, motion } from 'framer-motion';
import { useEffect } from 'react';
import { ALERT_DISMISS_MS } from '../data/sightingsGenerator';

// Original, simple geometric pixel-portrait sprite — not a licensed character.
function PixelPortrait() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="h-8 w-8 shrink-0"
      shapeRendering="crispEdges"
    >
      <rect width="16" height="16" fill="#ff6b8a" />
      <rect x="4" y="2" width="8" height="2" fill="#0a1628" />
      <rect x="3" y="4" width="10" height="8" fill="#0a1628" />
      <rect x="5" y="6" width="2" height="2" fill="#e8f4f8" />
      <rect x="9" y="6" width="2" height="2" fill="#e8f4f8" />
      <rect x="6" y="9" width="4" height="1" fill="#e8f4f8" />
    </svg>
  );
}

export default function AlertBanner({ alert, onDismiss }) {
  useEffect(() => {
    if (!alert) return undefined;
    const timer = setTimeout(() => onDismiss(), ALERT_DISMISS_MS);
    return () => clearTimeout(timer);
  }, [alert, onDismiss]);

  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-30 flex justify-center px-4 pt-4">
      <AnimatePresence>
        {alert && (
          <motion.div
            key={alert.id + alert.timestamp}
            initial={{ y: -80, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -80, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 260, damping: 24 }}
            className="pointer-events-auto flex max-w-xl items-center gap-3 rounded-lg border-2 border-navy bg-alert px-4 py-3 shadow-lg"
          >
            <PixelPortrait />
            <div className="min-w-0">
              <p className="font-display text-[9px] uppercase tracking-widest text-navy/70">
                Sighting Alert
              </p>
              <p className="font-body text-sm font-medium leading-snug text-navy">
                {alert.flavorText}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
