import { useCallback, useEffect, useRef, useState } from 'react';
import MapCanvas from './components/MapCanvas';
import Ticker from './components/Ticker';
import AlertBanner from './components/AlertBanner';
import RadarScan from './components/RadarScan';
import ControlBar from './components/ControlBar';
import {
  generateSighting,
  getRandomSpawnDelay,
  TICKER_MAX_ENTRIES,
} from './data/sightingsGenerator';

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

  const sightingsRef = useRef([]);
  const mutedRef = useRef(muted);
  const audioCtxRef = useRef(null);
  const timeoutRef = useRef(null);

  useEffect(() => {
    mutedRef.current = muted;
  }, [muted]);

  const spawnSighting = useCallback(() => {
    const { sighting, isNew } = generateSighting(sightingsRef.current);
    const nextSightings = isNew
      ? [...sightingsRef.current, sighting]
      : sightingsRef.current.map((s) => (s.id === sighting.id ? sighting : s));
    sightingsRef.current = nextSightings;
    setSightings(nextSightings);

    logIdCounter += 1;
    setLog((prevLog) =>
      [
        {
          logId: logIdCounter,
          timestamp: sighting.timestamp,
          borough: sighting.borough,
          flavorText: sighting.flavorText,
        },
        ...prevLog,
      ].slice(0, TICKER_MAX_ENTRIES)
    );

    setAlert({
      id: sighting.id,
      timestamp: sighting.timestamp,
      flavorText: sighting.flavorText,
    });
    setPulseTrigger((p) => p + 1);

    if (!mutedRef.current) playBlip(audioCtxRef);
  }, []);

  useEffect(() => {
    function scheduleNext() {
      timeoutRef.current = setTimeout(() => {
        spawnSighting();
        scheduleNext();
      }, getRandomSpawnDelay());
    }
    scheduleNext();
    return () => clearTimeout(timeoutRef.current);
  }, [spawnSighting]);

  const handleClear = useCallback(() => {
    sightingsRef.current = [];
    setSightings([]);
    setLog([]);
    setAlert(null);
  }, []);

  const handleDismissAlert = useCallback(() => setAlert(null), []);
  const handleToggleMute = useCallback(() => setMuted((m) => !m), []);

  const totalSightings = sightings.reduce((sum, s) => sum + s.count, 0);

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-navy text-text-primary">
      <AlertBanner alert={alert} onDismiss={handleDismissAlert} />

      <header className="flex items-center justify-between border-b border-grid px-4 py-3">
        <h1 className="font-display text-xs uppercase tracking-widest text-teal sm:text-sm">
          Web-Slinger Watch
        </h1>
        <p className="hidden font-mono text-[10px] text-text-muted sm:block">
          Sightings tracked: {totalSightings}
        </p>
      </header>

      <div className="flex flex-1 flex-col overflow-hidden md:flex-row">
        <div className="relative flex-1 overflow-hidden">
          <MapCanvas sightings={sightings} />
          <div className="absolute right-4 top-4">
            <RadarScan pulseTrigger={pulseTrigger} />
          </div>
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
    </div>
  );
}
