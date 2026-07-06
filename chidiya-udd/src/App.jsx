import { useCallback, useEffect, useRef, useState } from 'react';
import KiteMark from './components/KiteMark.jsx';
import SetupScreen from './screens/SetupScreen.jsx';
import PlayingScreen from './screens/PlayingScreen.jsx';
import GameOverScreen from './screens/GameOverScreen.jsx';
import { createWordPicker, getIntervalMs, getLevel } from './data/words.js';
import { useBeep } from './hooks/useBeep.js';

const STARTING_LIVES = 3;

const KITE_POSITIONS = [
  { top: '8%', left: '6%', width: '3rem' },
  { top: '18%', right: '8%', width: '4rem' },
  { bottom: '14%', left: '10%', width: '2.5rem' },
  { bottom: '22%', right: '12%', width: '3.5rem' },
  { top: '46%', left: '2%', width: '2rem' },
];

export default function App() {
  const [screen, setScreen] = useState('setup');
  const [players, setPlayers] = useState([]);
  const [muted, setMuted] = useState(false);
  const [paused, setPaused] = useState(false);
  const [callCount, setCallCount] = useState(0);
  const [currentWord, setCurrentWord] = useState(null);
  const [wordKey, setWordKey] = useState(0);
  const [resumeGeneration, setResumeGeneration] = useState(0);
  const [toastQueue, setToastQueue] = useState([]);
  const [activeToast, setActiveToast] = useState(null);
  const [shakeId, setShakeId] = useState(null);
  const [gameOverStats, setGameOverStats] = useState(null);

  const pickerRef = useRef(null);
  const { beep, unlock } = useBeep(muted);

  const addPlayer = useCallback((name) => {
    setPlayers((prev) =>
      prev.length >= 8 ? prev : [...prev, { id: crypto.randomUUID(), name, lives: STARTING_LIVES }]
    );
  }, []);

  const removePlayer = useCallback((id) => {
    setPlayers((prev) => prev.filter((p) => p.id !== id));
  }, []);

  const startGame = useCallback(() => {
    unlock();
    pickerRef.current = createWordPicker();
    setCallCount(0);
    setCurrentWord(null);
    setScreen('playing');
  }, [unlock]);

  const advanceWord = useCallback(() => {
    const word = pickerRef.current();
    setCurrentWord(word);
    setWordKey((k) => k + 1);
    beep(word.flying);
    setCallCount((c) => c + 1);
  }, [beep]);

  useEffect(() => {
    if (screen !== 'playing' || paused) return undefined;
    const delay = callCount === 0 ? 0 : getIntervalMs(callCount);
    const t = setTimeout(advanceWord, delay);
    return () => clearTimeout(t);
  }, [screen, paused, callCount, advanceWord]);

  function togglePause() {
    setPaused((p) => {
      if (p) setResumeGeneration((g) => g + 1);
      return !p;
    });
  }

  function toggleMute() {
    setMuted((m) => !m);
  }

  function dockLife(id) {
    setPlayers((prev) => {
      const target = prev.find((p) => p.id === id);
      if (!target || target.lives === 0) return prev;
      const next = prev.map((p) => (p.id === id ? { ...p, lives: p.lives - 1 } : p));
      const justEliminated = next.find((p) => p.id === id && p.lives === 0);
      if (justEliminated) {
        setShakeId(id);
        setTimeout(() => setShakeId((cur) => (cur === id ? null : cur)), 420);
        setToastQueue((q) => [
          ...q,
          { id: crypto.randomUUID(), message: `${justEliminated.name} is out!` },
        ]);
      }
      return next;
    });
  }

  useEffect(() => {
    if (activeToast || toastQueue.length === 0) return undefined;
    const [next, ...rest] = toastQueue;
    setActiveToast(next);
    setToastQueue(rest);
    const t = setTimeout(() => setActiveToast(null), 2500);
    return () => clearTimeout(t);
  }, [toastQueue, activeToast]);

  useEffect(() => {
    if (screen !== 'playing' || players.length < 2) return;
    const active = players.filter((p) => p.lives > 0);
    if (active.length === 1) {
      setGameOverStats({
        winnerName: active[0].name,
        level: getLevel(callCount),
        totalCalls: callCount,
      });
      setScreen('gameover');
    }
  }, [players, screen, callCount]);

  function resetLivesAndWord() {
    pickerRef.current = createWordPicker();
    setPlayers((prev) => prev.map((p) => ({ ...p, lives: STARTING_LIVES })));
    setCallCount(0);
    setCurrentWord(null);
    setActiveToast(null);
    setToastQueue([]);
    setPaused(false);
  }

  function restartCurrentGame() {
    resetLivesAndWord();
  }

  function playAgain() {
    resetLivesAndWord();
    setScreen('playing');
  }

  function newPlayers() {
    setPlayers([]);
    setGameOverStats(null);
    setScreen('setup');
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      {KITE_POSITIONS.map((pos, i) => (
        <KiteMark key={i} className="absolute" style={{ ...pos, animationDelay: `${i * 0.7}s` }} />
      ))}

      {screen === 'setup' && (
        <SetupScreen
          players={players}
          onAddPlayer={addPlayer}
          onRemovePlayer={removePlayer}
          onStart={startGame}
        />
      )}

      {screen === 'playing' && (
        <PlayingScreen
          currentWord={currentWord}
          wordKey={wordKey}
          resumeGeneration={resumeGeneration}
          delayMs={getIntervalMs(callCount)}
          level={getLevel(callCount)}
          paused={paused}
          muted={muted}
          players={players}
          activeToast={activeToast}
          shakeId={shakeId}
          onDockLife={dockLife}
          onTogglePause={togglePause}
          onToggleMute={toggleMute}
          onRestart={restartCurrentGame}
        />
      )}

      {screen === 'gameover' && gameOverStats && (
        <GameOverScreen
          winnerName={gameOverStats.winnerName}
          level={gameOverStats.level}
          totalCalls={gameOverStats.totalCalls}
          onPlayAgain={playAgain}
          onNewPlayers={newPlayers}
        />
      )}
    </div>
  );
}
