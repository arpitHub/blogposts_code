import { useState } from 'react';
import { Plus, Trash2, PlayCircle } from 'lucide-react';

const MAX_PLAYERS = 8;
const MIN_PLAYERS = 2;

export default function SetupScreen({ players, onAddPlayer, onRemovePlayer, onStart }) {
  const [nameInput, setNameInput] = useState('');
  const atMax = players.length >= MAX_PLAYERS;

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = nameInput.trim();
    if (!trimmed || atMax) return;
    onAddPlayer(trimmed);
    setNameInput('');
  }

  return (
    <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 py-10">
      <h1 className="font-display text-4xl font-extrabold text-marigold sm:text-5xl">
        Chidiya Udd
      </h1>
      <p className="mt-2 text-center font-body text-sm text-offwhite/70 sm:text-base">
        Bird, Fly! &mdash; raise a hand only when something flies.
      </p>

      <form onSubmit={handleSubmit} className="mt-8 flex w-full max-w-sm gap-2">
        <input
          type="text"
          value={nameInput}
          onChange={(e) => setNameInput(e.target.value)}
          placeholder="Player name"
          disabled={atMax}
          maxLength={20}
          className="flex-1 rounded-xl border border-offwhite/20 bg-dusk-800/60 px-4 py-3 font-body text-offwhite placeholder-offwhite/40 outline-none focus-visible:ring-2 focus-visible:ring-marigold disabled:opacity-40"
        />
        <button
          type="submit"
          disabled={atMax || !nameInput.trim()}
          aria-label="Add player"
          className="flex items-center justify-center rounded-xl bg-marigold px-4 py-3 text-dusk-900 transition-opacity focus-visible:ring-2 focus-visible:ring-offwhite disabled:opacity-40"
        >
          <Plus size={22} strokeWidth={3} />
        </button>
      </form>

      <p className="mt-2 font-mono text-xs text-offwhite/50">
        {players.length}/{MAX_PLAYERS} players
      </p>

      <ul className="mt-4 flex w-full max-w-sm flex-col gap-2">
        {players.map((player) => (
          <li
            key={player.id}
            className="flex items-center justify-between rounded-xl border border-offwhite/10 bg-dusk-800/40 px-4 py-3"
          >
            <span className="font-body text-offwhite">{player.name}</span>
            <button
              type="button"
              onClick={() => onRemovePlayer(player.id)}
              aria-label={`Remove ${player.name}`}
              className="rounded-lg p-1 text-kite/80 transition-colors hover:text-kite focus-visible:ring-2 focus-visible:ring-kite"
            >
              <Trash2 size={18} />
            </button>
          </li>
        ))}
      </ul>

      <button
        type="button"
        onClick={onStart}
        disabled={players.length < MIN_PLAYERS}
        className="mt-8 flex items-center gap-2 rounded-2xl bg-parrot px-8 py-4 font-display text-lg font-bold text-offwhite shadow-lg transition-opacity focus-visible:ring-2 focus-visible:ring-offwhite disabled:cursor-not-allowed disabled:opacity-30"
      >
        <PlayCircle size={24} />
        Start Game
      </button>
      {players.length < MIN_PLAYERS && (
        <p className="mt-2 font-mono text-xs text-offwhite/40">
          Add at least {MIN_PLAYERS} players to start
        </p>
      )}
    </div>
  );
}
