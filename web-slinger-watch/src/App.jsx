import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import MapCanvas from './components/MapCanvas';
import Ticker from './components/Ticker';
import AlertBanner from './components/AlertBanner';
import RadarScan from './components/RadarScan';
import ControlBar from './components/ControlBar';
import TitleBar from './components/TitleBar';
import StatsStrip from './components/StatsStrip';
import ScanlineOverlay from './components/ScanlineOverlay';
import HelpButton from './components/HelpButton';
import InstructionsModal, {
  INTRO_STORAGE_KEY,
} from './components/InstructionsModal';
import SignalDecoder, {
  getRandomPuzzleDelay,
} from './components/SignalDecoder';
import {
  generateSighting,
  getRandomSpawnDelay,
  TICKER_MAX_ENTRIES,
} from './data/sightingsGenerator';
import {
  CIPHER_LABELS,
  randomCipherMessage,
  randomShift,
} from './data/cipherMessages';
import { encode } from './utils/caesarCipher';

const TAGLINE = 'Neighborhood sighting board — unofficial, always watching.';
const HIGHLIGHT_DURATION_MS = 2600;

// Tiny synthesized "blip" via Web Audio API — no audio files needed.
function playBlip(audioCtxRef) {
  try {
    if (!audioCtxRef.current) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      audioCtxRef.current = new Ctx();
    }
    const ctx = audioCtxRef.current;
    if (ctx.state === 'suspended') ctx.resume();

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 660;
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.15, ctx.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.26);
  } catch {
    // Web Audio unsupported or blocked — fail silently.
  }
}

let logIdCounter = 0;

export default function App() {
  const [sightings, setSightings] = useState([]);
  const [log, setLog] = useState([]);
  const [alert, setAlert] = useState(null);
  const [muted, setMuted] = useState(false);
  const [pulseTrigger, setPulseTrigger] = useState(0);
  const [casesSolved, setCasesSolved] = useState(0);
  const [activeCipher, setActiveCipher] = useState(null);
  const [highlightBorough, setHighlightBorough] = useState(null);
  const [showInstructions, setShowInstructions] = useState(false);

  const sightingsRef = useRef([]);
  const mutedRef = useRef(muted);
  const activeCipherRef = useRef(null);
  const audioCtxRef = useRef(null);
  const spawnTimeoutRef = useRef(null);
  const puzzleTimeoutRef = useRef(null);

  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  useEffect(() => {
    activeCipherRef.current = activeCipher;
  }, [activeCipher]);

  const appendLog = useCallback((entry) => {
    logIdCounter += 1;
    setLog((prevLog) =>
      [{ logId: logIdCounter, timestamp: Date.now(), ...entry }, ...prevLog].slice(
        0,
        TICKER_MAX_ENTRIES
      )
    );
  }, []);

  const spawnSighting = useCallback(
    (options = {}) => {
      const { sighting, isNew } = generateSighting(
        sightingsRef.current,
        options
      );
      const nextSightings = isNew
        ? [...sightingsRef.current, sighting]
        : sightingsRef.current.map((s) =>
            s.id === sighting.id ? sighting : s
          );
      sightingsRef.current = nextSightings;
      setSightings(nextSightings);

      appendLog({
        kind: 'sighting',
        borough: sighting.borough,
        flavorText: sighting.flavorText,
      });

      setAlert({
        id: sighting.id,
        timestamp: sighting.timestamp,
        flavorText: sighting.flavorText,
      });
      setPulseTrigger((p) => p + 1);

      if (!mutedRef.current) playBlip(audioCtxRef);
    },
    [appendLog]
  );

  // Randomized 4-9s sighting spawns.
  useEffect(() => {
    function scheduleNext() {
      spawnTimeoutRef.current = setTimeout(() => {
        spawnSighting();
        scheduleNext();
      }, getRandomSpawnDelay());
    }
    scheduleNext();
    return () => clearTimeout(spawnTimeoutRef.current);
  }, [spawnSighting]);

  // Randomized 60-90s decoder puzzles, never more than one at a time.
  useEffect(() => {
    function scheduleNext() {
      puzzleTimeoutRef.current = setTimeout(() => {
        if (!activeCipherRef.current) {
          const message = randomCipherMessage();
          const shift = randomShift();
          setActiveCipher({
            id: `cipher-${Date.now()}`,
            borough: message.borough,
            plaintext: message.plaintext,
            shift,
            ciphertext: encode(message.plaintext, shift),
          });
        }
        scheduleNext();
      }, getRandomPuzzleDelay());
    }
    scheduleNext();
    return () => clearTimeout(puzzleTimeoutRef.current);
  }, []);

  // Auto-open the field manual on a visitor's first session only.
  useEffect(() => {
    try {
      if (!window.localStorage.getItem(INTRO_STORAGE_KEY)) {
        setShowInstructions(true);
      }
    } catch {
      // localStorage blocked (private mode) — just skip the auto-open.
    }
  }, []);

  // Let a solved borough glow for a beat, then settle back.
  useEffect(() => {
    if (!highlightBorough) return undefined;
    const timer = setTimeout(
      () => setHighlightBorough(null),
      HIGHLIGHT_DURATION_MS
    );
    return () => clearTimeout(timer);
  }, [highlightBorough]);

  const handleCipherSolved = useCallback(
    (borough) => {
      setCasesSolved((c) => c + 1);
      setHighlightBorough(borough);
      setActiveCipher(null);
      appendLog({
        kind: 'case-solved',
        borough,
        flavorText: CIPHER_LABELS.logSolved,
      });
      // The tip pans out: a guaranteed sighting in the predicted borough.
      spawnSighting({ boroughId: borough });
    },
    [appendLog, spawnSighting]
  );

  const handleCipherMissed = useCallback(() => {
    setActiveCipher(null);
    appendLog({
      kind: 'case-missed',
      flavorText: CIPHER_LABELS.logMissed,
    });
  }, [appendLog]);

  const handleClear = useCallback(() => {
    sightingsRef.current = [];
    setSightings([]);
    setLog([]);
    setAlert(null);
  }, []);

  const handleDismissAlert = useCallback(() => setAlert(null), []);
  const handleToggleMute = useCallback(() => setMuted((m) => !m), []);
  const handleOpenInstructions = useCallback(
    () => setShowInstructions(true),
    []
  );

  const handleCloseInstructions = useCallback(() => {
    setShowInstructions(false);
    try {
      window.localStorage.setItem(INTRO_STORAGE_KEY, 'true');
    } catch {
      // Nothing to persist to — the modal simply reopens next visit.
    }
  }, []);

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-navy text-text-primary">
      <ScanlineOverlay />
      <AlertBanner alert={alert} onDismiss={handleDismissAlert} />

      <TitleBar tagline={TAGLINE} />

      <div className="flex flex-1 flex-col overflow-hidden md:flex-row">
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="relative flex-1 overflow-hidden">
            <MapCanvas
              sightings={sightings}
              highlightBorough={highlightBorough}
            />
            <div className="absolute right-4 top-4 flex flex-col items-center gap-3">
              <RadarScan pulseTrigger={pulseTrigger} />
              <HelpButton onClick={handleOpenInstructions} />
            </div>
            {/* Bottom-left keeps the decoder clear of the top alert banner. */}
            <div className="absolute bottom-4 left-4">
              <AnimatePresence>
                {activeCipher && (
                  <SignalDecoder
                    key={activeCipher.id}
                    puzzle={activeCipher}
                    onSolved={handleCipherSolved}
                    onMissed={handleCipherMissed}
                  />
                )}
              </AnimatePresence>
            </div>
          </div>
          <StatsStrip sightings={sightings} casesSolved={casesSolved} />
        </div>

        <aside className="h-48 border-t border-grid bg-navy/60 md:h-auto md:w-72 md:border-l md:border-t-0">
          <Ticker entries={log} />
        </aside>
      </div>

      <ControlBar
        muted={muted}
        onToggleMute={handleToggleMute}
        onClear={handleClear}
      />

      {showInstructions && (
        <InstructionsModal onClose={handleCloseInstructions} />
      )}
    </div>
  );
}
