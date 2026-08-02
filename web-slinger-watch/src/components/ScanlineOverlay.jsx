// Full-viewport CRT treatment: static scanlines, one slow travelling band,
// and a soft vignette. Purely decorative — never intercepts pointer events.
export default function ScanlineOverlay() {
  return (
    <div className="pointer-events-none fixed inset-0 z-50 overflow-hidden">
      <div
        className="absolute inset-0 opacity-[0.28]"
        style={{
          backgroundImage:
            'repeating-linear-gradient(to bottom, rgba(10,22,40,0.55) 0px, rgba(10,22,40,0.55) 1px, transparent 1px, transparent 3px)',
        }}
      />
      <div
        className="absolute inset-x-0 h-1/3 animate-scanline opacity-[0.05]"
        style={{
          background:
            'linear-gradient(to bottom, transparent, rgba(45,212,191,0.5), transparent)',
        }}
      />
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at center, transparent 42%, rgba(4,10,20,0.55) 100%)',
        }}
      />
    </div>
  );
}
