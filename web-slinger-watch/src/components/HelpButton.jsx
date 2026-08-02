export default function HelpButton({ onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Open field manual"
      className="group relative flex h-9 w-9 items-center justify-center rounded-full border-2 border-teal bg-navy font-display text-[10px] text-teal transition-colors hover:bg-teal hover:text-navy active:scale-95"
    >
      {/* Same breathing halo the sighting markers use. */}
      <span
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 animate-pulse-glow rounded-full bg-teal"
        style={{ filter: 'blur(7px)' }}
      />
      <span className="relative">?</span>
    </button>
  );
}
