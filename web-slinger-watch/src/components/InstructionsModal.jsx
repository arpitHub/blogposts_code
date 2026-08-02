import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { encode } from '../utils/caesarCipher';
import { EXAMPLE_CIPHER, CIPHER_LABELS } from '../data/cipherMessages';

export const INTRO_STORAGE_KEY = 'hasSeenIntro';

const HALFTONE_STYLE = {
  backgroundImage:
    'radial-gradient(circle, rgba(232,244,248,0.5) 1px, transparent 1.2px)',
  backgroundSize: '7px 7px',
};

function PinIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4 shrink-0 text-marker-pink">
      <circle cx="8" cy="8" r="6" fill="currentColor" />
      <circle cx="8" cy="8" r="2.2" fill="#0a1628" />
    </svg>
  );
}

function RadarIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4 shrink-0 text-teal">
      <circle
        cx="8"
        cy="8"
        r="6.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <path d="M8 8 L8 1.5 A6.5 6.5 0 0 1 13.6 5 Z" fill="currentColor" />
    </svg>
  );
}

function SignalIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4 shrink-0 text-alert">
      <g fill="currentColor">
        <rect x="1" y="10" width="2.5" height="5" />
        <rect x="5" y="7" width="2.5" height="8" />
        <rect x="9" y="4" width="2.5" height="11" />
        <rect x="13" y="1" width="2.5" height="14" />
      </g>
    </svg>
  );
}

const SECTIONS = [
  {
    id: 'sightings',
    Icon: PinIcon,
    title: 'Sightings & Clusters',
    body: 'Pins drop wherever our hero turns up — pink and green by borough. When two sightings land close together they merge into one pin, and the amber badge counts how many reports it holds.',
  },
  {
    id: 'radar',
    Icon: RadarIcon,
    title: 'The Radar',
    body: 'The sweep in the corner never stops turning. It flashes the moment a fresh report lands, so you can catch a new pin even when your eyes are elsewhere on the board.',
  },
];

// Static, non-interactive demo of the decoder dial.
function DemoDial() {
  const ciphertext = encode(EXAMPLE_CIPHER.plaintext, EXAMPLE_CIPHER.shift);
  return (
    <div className="mt-2 rounded-md border border-teal/40 bg-navy/80 p-2">
      <p className="font-display text-[6px] uppercase tracking-widest text-text-muted">
        {CIPHER_LABELS.intercepted}
      </p>
      <p className="mt-1 font-mono text-xs tracking-wider text-marker-pink">
        {ciphertext}
      </p>
      <div className="mt-2 flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded border border-teal/50 font-mono text-[10px] text-teal/60">
          &lt;
        </span>
        <div className="flex flex-1 flex-col items-center">
          <span className="font-display text-[6px] uppercase tracking-widest text-text-muted">
            {CIPHER_LABELS.shift}
          </span>
          <span className="font-mono text-sm text-teal">
            {String(EXAMPLE_CIPHER.shift).padStart(2, '0')}
          </span>
        </div>
        <span className="flex h-6 w-6 items-center justify-center rounded border border-teal/50 font-mono text-[10px] text-teal/60">
          &gt;
        </span>
      </div>
      <p className="mt-2 font-display text-[6px] uppercase tracking-widest text-text-muted">
        {CIPHER_LABELS.decodedAs}
      </p>
      <p className="mt-1 font-mono text-xs tracking-wider text-marker-green">
        {EXAMPLE_CIPHER.plaintext}
      </p>
    </div>
  );
}

export default function InstructionsModal({ onClose }) {
  useEffect(() => {
    function handleKey(event) {
      if (event.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-navy/80 p-4 backdrop-blur-sm"
      onClick={onClose}
      role="presentation"
    >
      <motion.div
        role="dialog"
        aria-modal="true"
        aria-label="Field Manual"
        initial={{ opacity: 0, scale: 0.92, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: 'spring', stiffness: 260, damping: 22 }}
        onClick={(event) => event.stopPropagation()}
        className="relative w-full max-w-md rounded-lg border-4 border-teal bg-navy shadow-2xl"
      >
        {/* Tail lives outside the scroll container so it isn't clipped. */}
        <div
          aria-hidden="true"
          className="absolute -top-[14px] right-9 h-4 w-4 rotate-45 border-l-4 border-t-4 border-teal bg-navy"
        />

        {/* Halftone texture, confined to the panel. */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 rounded-md opacity-[0.06]"
          style={HALFTONE_STYLE}
        />

        <button
          type="button"
          onClick={onClose}
          aria-label="Close field manual"
          className="absolute right-0 top-0 h-9 w-9 bg-teal font-display text-[10px] text-navy transition-colors hover:bg-alert"
          style={{ clipPath: 'polygon(100% 0, 100% 100%, 0 0)' }}
        >
          <span className="absolute right-1.5 top-0.5">×</span>
        </button>

        <div className="relative max-h-[85vh] overflow-y-auto p-5">
          <h2 className="font-display text-sm uppercase tracking-widest text-teal">
            Field Manual
          </h2>
          <p className="mt-2 font-body text-xs text-text-muted">
            You&apos;re the guy in the chair. Here&apos;s how the board works.
          </p>

          <div className="mt-4 space-y-4">
            {SECTIONS.map(({ id, Icon, title, body }) => (
              <section key={id} className="flex gap-3">
                <Icon />
                <div className="min-w-0">
                  <h3 className="font-display text-[8px] uppercase tracking-widest text-text-primary">
                    {title}
                  </h3>
                  <p className="mt-1.5 font-body text-xs leading-relaxed text-text-muted">
                    {body}
                  </p>
                </div>
              </section>
            ))}

            <section className="flex gap-3">
              <SignalIcon />
              <div className="min-w-0 flex-1">
                <h3 className="font-display text-[8px] uppercase tracking-widest text-text-primary">
                  {CIPHER_LABELS.panelTitle}
                </h3>
                <p className="mt-1.5 font-body text-xs leading-relaxed text-text-muted">
                  Now and then a scrambled transmission comes through. Every
                  letter has been rotated by the same amount — spin the dial
                  until the preview reads like plain English, then hit{' '}
                  <span className="text-alert">{CIPHER_LABELS.lockIn}</span>{' '}
                  before the bar runs out. Here the dial is parked on{' '}
                  {EXAMPLE_CIPHER.shift}:
                </p>
                <DemoDial />
              </div>
            </section>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="mt-5 w-full rounded-full border-2 border-alert bg-alert px-4 py-2.5 font-display text-[10px] uppercase tracking-widest text-navy transition-colors hover:bg-navy hover:text-alert active:scale-95"
          >
            Got It
          </button>
        </div>
      </motion.div>
    </div>
  );
}
