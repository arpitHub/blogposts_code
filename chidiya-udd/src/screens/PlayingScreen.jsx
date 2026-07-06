import { Volume2, VolumeX, Pause, Play, RotateCcw } from 'lucide-react';

export default function PlayingScreen({
  currentWord,
  wordKey,
  resumeGeneration,
  delayMs,
  level,
  paused,
  muted,
  players,
  activeToast,
  shakeId,
  onDockLife,
  onTogglePause,
  onToggleMute,
  onRestart,
}) {
  return (
    <div className="relative z-10 flex min-h-screen flex-col items-center px-4 py-6">
      <div className="flex w-full max-w-2xl items-center justify-between">
        <span className="font-mono text-sm text-offwhite/60">Level {level}</span>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onToggleMute}
            aria-pressed={muted}
            aria-label={muted ? 'Unmute' : 'Mute'}
            className="rounded-lg p-2 text-offwhite/80 hover:text-offwhite focus-visible:ring-2 focus-visible:ring-marigold"
          >
            {muted ? <VolumeX size={20} /> : <Volume2 size={20} />}
          </button>
          <button
            type="button"
            onClick={onTogglePause}
            aria-pressed={paused}
            aria-label={paused ? 'Resume' : 'Pause'}
            className="rounded-lg p-2 text-offwhite/80 hover:text-offwhite focus-visible:ring-2 focus-visible:ring-marigold"
          >
            {paused ? <Play size={20} /> : <Pause size={20} />}
          </button>
          <button
            type="button"
            onClick={onRestart}
            aria-label="Restart game"
            className="rounded-lg p-2 text-offwhite/80 hover:text-offwhite focus-visible:ring-2 focus-visible:ring-marigold"
          >
            <RotateCcw size={20} />
          </button>
        </div>
      </div>

      <div className="mt-2 h-1 w-full max-w-2xl overflow-hidden rounded-full bg-offwhite/10">
        <div
          key={`${wordKey}-${resumeGeneration}`}
          className="progress-fill h-full bg-marigold"
          style={{
            animationDuration: `${delayMs}ms`,
            animationPlayState: paused ? 'paused' : 'running',
          }}
        />
      </div>

      <div className="flex flex-1 items-center justify-center py-10">
        {currentWord && (
          <div
            key={wordKey}
            className={`swoop-in flex w-72 flex-col items-center rounded-3xl border-2 px-8 py-10 text-center shadow-2xl sm:w-96 ${
              currentWord.flying
                ? 'border-parrot bg-parrot/20'
                : 'border-kite bg-kite/20'
            }`}
          >
            <span className="font-display text-4xl font-extrabold text-offwhite sm:text-5xl">
              {currentWord.word}
            </span>
            <span className="mt-3 font-body text-lg text-offwhite/70">
              {currentWord.translation}
            </span>
          </div>
        )}
      </div>

      <div className="grid w-full max-w-2xl grid-cols-2 gap-3 sm:grid-cols-4">
        {players.map((player) => {
          const eliminated = player.lives === 0;
          return (
            <button
              type="button"
              key={player.id}
              onClick={() => !eliminated && onDockLife(player.id)}
              disabled={eliminated}
              className={`flex flex-col items-center gap-2 rounded-2xl border border-offwhite/10 bg-dusk-800/50 px-3 py-4 transition-opacity focus-visible:ring-2 focus-visible:ring-marigold ${
                eliminated ? 'opacity-40 grayscale' : 'active:scale-95'
              } ${player.id === shakeId ? 'shake-out' : ''}`}
            >
              <span className="truncate font-body text-sm text-offwhite">{player.name}</span>
              <span className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className={`h-3 w-3 rounded-full border border-marigold/60 ${
                      i < player.lives ? 'bg-marigold' : 'bg-transparent'
                    }`}
                  />
                ))}
              </span>
            </button>
          );
        })}
      </div>

      {activeToast && (
        <div
          role="status"
          className="fixed bottom-6 left-1/2 -translate-x-1/2 rounded-xl bg-kite px-5 py-3 font-body text-sm font-semibold text-offwhite shadow-xl"
        >
          {activeToast.message}
        </div>
      )}
    </div>
  );
}
