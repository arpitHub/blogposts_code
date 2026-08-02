import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { decode, ALPHABET_SIZE } from '../utils/caesarCipher';
import { CIPHER_LABELS } from '../data/cipherMessages';

export const PUZZLE_DURATION_MS = 35000;
export const MIN_PUZZLE_INTERVAL_MS = 60000;
export const MAX_PUZZLE_INTERVAL_MS = 90000;
// How long the solved/lost flourish stays on screen before resolving.
const RESOLVE_DELAY_MS = 1100;

export function getRandomPuzzleDelay() {
  return (
    MIN_PUZZLE_INTERVAL_MS +
    Math.random() * (MAX_PUZZLE_INTERVAL_MS - MIN_PUZZLE_INTERVAL_MS)
  );
}

function ArrowButton({ direction, onClick, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={direction === -1 ? 'Decrease shift' : 'Increase shift'}
      className="flex h-9 w-9 items-center justify-center rounded-md border-2 border-teal text-teal transition-colors hover:bg-teal hover:text-navy active:scale-95 disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-teal"
    >
      <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor">
        {direction === -1 ? (
          <path d="M10 2 L4 8 L10 14 Z" />
        ) : (
          <path d="M6 2 L12 8 L6 14 Z" />
        )}
      </svg>
    </button>
  );
}

export default function SignalDecoder({ puzzle, onSolved, onMissed }) {
  const [shift, setShift] = useState(0);
  const [remaining, setRemaining] = useState(PUZZLE_DURATION_MS);
  const [status, setStatus] = useState('active');
  const [shakeKey, setShakeKey] = useState(0);

  const onSolvedRef = useRef(onSolved);
  const onMissedRef = useRef(onMissed);

  useEffect(() => {
    onSolvedRef.current = onSolved;
    onMissedRef.current = onMissed;
  }, [onSolved, onMissed]);

  // Countdown. Wall-clock based so a throttled tab can't stretch the timer.
  useEffect(() => {
    if (status !== 'active') return undefined;
    const startedAt = Date.now();
    const intervalId = setInterval(() => {
      const left = Math.max(0, PUZZLE_DURATION_MS - (Date.now() - startedAt));
      setRemaining(left);
      if (left === 0) setStatus('lost');
    }, 100);
    return () => clearInterval(intervalId);
  }, [status]);

  // Hand the outcome back to App once the flourish has played.
  useEffect(() => {
    if (status === 'active') return undefined;
    const timer = setTimeout(() => {
      if (status === 'solved') onSolvedRef.current(puzzle.borough);
      else onMissedRef.current();
    }, RESOLVE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [status, puzzle.borough]);

  const adjustShift = useCallback((delta) => {
    setShift((s) => (s + delta + ALPHABET_SIZE) % ALPHABET_SIZE);
  }, []);

  const handleLockIn = useCallback(() => {
    if (status !== 'active') return;
    if (decode(puzzle.ciphertext, shift) === puzzle.plaintext) {
      setStatus('solved');
    } else {
      // No penalty for a wrong guess — just a nudge.
      setShakeKey((k) => k + 1);
    }
  }, [status, puzzle.ciphertext, puzzle.plaintext, shift]);

  const preview = decode(puzzle.ciphertext, shift);
  const progress = (remaining / PUZZLE_DURATION_MS) * 100;
  const secondsLeft = Math.ceil(remaining / 1000);
  const isActive = status === 'active';

  return (
    <motion.div
      initial={{ opacity: 0, y: 24, scale: 0.96 }}
      animate={{ opacity: status === 'lost' ? 0.35 : 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 24, scale: 0.96 }}
      transition={{ type: 'spring', stiffness: 240, damping: 24 }}
      className="w-[19rem] rounded-lg border-2 border-teal/70 bg-navy/95 p-3 shadow-xl backdrop-blur-sm sm:w-[21rem]"
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <h2 className="font-display text-[8px] uppercase tracking-widest text-teal">
          {CIPHER_LABELS.panelTitle}
        </h2>
        <span className="font-mono text-[10px] tabular-nums text-text-muted">
          {String(secondsLeft).padStart(2, '0')}s
        </span>
      </div>

      <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-grid/60">
        <div
          className="h-full rounded-full bg-alert transition-[width] duration-100 ease-linear"
          style={{ width: `${progress}%` }}
        />
      </div>

      <AnimatePresence mode="wait">
        {status === 'solved' ? (
          <motion.div
            key="solved"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center gap-1 py-5"
          >
            <p className="font-display text-[11px] uppercase tracking-widest text-marker-green">
              {CIPHER_LABELS.solved}
            </p>
            <p className="font-mono text-xs text-text-primary">
              {puzzle.plaintext}
            </p>
          </motion.div>
        ) : status === 'lost' ? (
          <motion.div
            key="lost"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex flex-col items-center gap-1 py-5"
          >
            <p className="font-display text-[11px] uppercase tracking-widest text-text-muted">
              {CIPHER_LABELS.lost}
            </p>
          </motion.div>
        ) : (
          <motion.div key="active" exit={{ opacity: 0 }}>
            <p className="font-display text-[7px] uppercase tracking-widest text-text-muted">
              {CIPHER_LABELS.intercepted}
            </p>
            <p className="mt-1 break-words font-mono text-sm tracking-wider text-marker-pink">
              {puzzle.ciphertext}
            </p>

            <p className="mt-3 font-display text-[7px] uppercase tracking-widest text-text-muted">
              {CIPHER_LABELS.decodedAs}
            </p>
            <p className="mt-1 min-h-[1.25rem] break-words font-mono text-sm tracking-wider text-text-primary">
              {preview}
            </p>

            <div
              key={shakeKey}
              className={`mt-3 flex items-center gap-2 ${
                shakeKey > 0 ? 'animate-shake' : ''
              }`}
            >
              <ArrowButton
                direction={-1}
                onClick={() => adjustShift(-1)}
                disabled={!isActive}
              />
              <div className="flex flex-1 flex-col items-center">
                <span className="font-display text-[7px] uppercase tracking-widest text-text-muted">
                  {CIPHER_LABELS.shift}
                </span>
                <span className="font-mono text-lg tabular-nums text-teal">
                  {String(shift).padStart(2, '0')}
                </span>
              </div>
              <ArrowButton
                direction={1}
                onClick={() => adjustShift(1)}
                disabled={!isActive}
              />
            </div>

            <button
              type="button"
              onClick={handleLockIn}
              disabled={!isActive}
              className="mt-3 w-full rounded-full border-2 border-alert bg-navy px-4 py-2 font-body text-sm font-semibold text-alert transition-colors hover:bg-alert hover:text-navy active:scale-95 disabled:opacity-40"
            >
              {CIPHER_LABELS.lockIn}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
