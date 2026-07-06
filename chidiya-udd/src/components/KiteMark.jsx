export default function KiteMark({ className = '', style }) {
  return (
    <svg
      viewBox="0 0 40 40"
      className={`float-slow pointer-events-none opacity-10 ${className}`}
      style={style}
      aria-hidden="true"
    >
      <path d="M20 2 L36 20 L20 38 L4 20 Z" fill="#FFA630" />
      <line x1="20" y1="2" x2="20" y2="38" stroke="#F5F0E6" strokeWidth="0.5" opacity="0.4" />
      <line x1="4" y1="20" x2="36" y2="20" stroke="#F5F0E6" strokeWidth="0.5" opacity="0.4" />
    </svg>
  );
}
