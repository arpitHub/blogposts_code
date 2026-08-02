function SpeakerIcon({ muted }) {
  return (
    <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor">
      <path d="M4 9v6h4l5 5V4L8 9H4z" />
      {muted ? (
        <path
          d="M16 8l6 8M22 8l-6 8"
          stroke="currentColor"
          strokeWidth="2"
          fill="none"
          strokeLinecap="round"
        />
      ) : (
        <path
          d="M17 8a5 5 0 010 8"
          stroke="currentColor"
          strokeWidth="2"
          fill="none"
          strokeLinecap="round"
        />
      )}
    </svg>
  );
}

export default function ControlBar({ muted, onToggleMute, onClear }) {
  return (
    <div className="flex items-center gap-3 border-t border-grid bg-navy/80 px-4 py-3">
      <button
        type="button"
        onClick={onToggleMute}
        className="flex items-center gap-2 rounded-full border-2 border-teal bg-navy px-4 py-2 font-body text-sm font-semibold text-teal transition-colors hover:bg-teal hover:text-navy active:scale-95"
      >
        <SpeakerIcon muted={muted} />
        {muted ? 'Unmute' : 'Mute'}
      </button>
      <button
        type="button"
        onClick={onClear}
        className="flex items-center gap-2 rounded-full border-2 border-alert bg-navy px-4 py-2 font-body text-sm font-semibold text-alert transition-colors hover:bg-alert hover:text-navy active:scale-95"
      >
        Clear Board
      </button>
    </div>
  );
}
