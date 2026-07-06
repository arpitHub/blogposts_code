import { Trophy, RotateCcw, Users } from 'lucide-react';

export default function GameOverScreen({ winnerName, level, totalCalls, onPlayAgain, onNewPlayers }) {
  return (
    <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 py-10 text-center">
      <Trophy size={48} className="text-marigold" />
      <p className="mt-4 font-mono text-sm uppercase tracking-widest text-offwhite/60">
        Winner
      </p>
      <h1 className="mt-1 font-display text-4xl font-extrabold text-offwhite sm:text-5xl">
        {winnerName}
      </h1>

      <div className="mt-6 flex gap-8 font-mono text-sm text-offwhite/70">
        <span>Level {level} reached</span>
        <span>{totalCalls} calls made</span>
      </div>

      <div className="mt-10 flex w-full max-w-sm flex-col gap-3">
        <button
          type="button"
          onClick={onPlayAgain}
          className="flex items-center justify-center gap-2 rounded-2xl bg-parrot px-8 py-4 font-display text-lg font-bold text-offwhite shadow-lg focus-visible:ring-2 focus-visible:ring-offwhite"
        >
          <RotateCcw size={22} />
          Play Again
        </button>
        <button
          type="button"
          onClick={onNewPlayers}
          className="flex items-center justify-center gap-2 rounded-2xl border border-offwhite/20 px-8 py-4 font-display text-lg font-bold text-offwhite focus-visible:ring-2 focus-visible:ring-marigold"
        >
          <Users size={22} />
          New Players
        </button>
      </div>
    </div>
  );
}
