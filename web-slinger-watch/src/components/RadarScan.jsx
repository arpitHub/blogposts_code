import { useEffect, useState } from 'react';

export default function RadarScan({ pulseTrigger }) {
  const [flash, setFlash] = useState(false);

  useEffect(() => {
    if (pulseTrigger == null) return undefined;
    setFlash(true);
    const timer = setTimeout(() => setFlash(false), 300);
    return () => clearTimeout(timer);
  }, [pulseTrigger]);

  return (
    <div
      className={`relative h-24 w-24 rounded-full border-2 border-teal/40 bg-navy/60 transition-[filter] duration-200 ${
        flash ? 'brightness-150' : 'brightness-100'
      }`}
    >
      <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
        <circle
          cx="50"
          cy="50"
          r="46"
          fill="none"
          stroke="#2dd4bf"
          strokeOpacity="0.25"
          strokeWidth="1"
        />
        <circle
          cx="50"
          cy="50"
          r="30"
          fill="none"
          stroke="#2dd4bf"
          strokeOpacity="0.2"
          strokeWidth="1"
        />
        <circle
          cx="50"
          cy="50"
          r="14"
          fill="none"
          stroke="#2dd4bf"
          strokeOpacity="0.2"
          strokeWidth="1"
        />
        <line
          x1="50"
          y1="4"
          x2="50"
          y2="96"
          stroke="#2dd4bf"
          strokeOpacity="0.12"
          strokeWidth="1"
        />
        <line
          x1="4"
          y1="50"
          x2="96"
          y2="50"
          stroke="#2dd4bf"
          strokeOpacity="0.12"
          strokeWidth="1"
        />
        <g
          className="origin-center animate-[spin_4s_linear_infinite]"
          style={{ transformOrigin: '50px 50px' }}
        >
          <defs>
            <linearGradient id="radarSweep" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#2dd4bf" stopOpacity="0" />
              <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.55" />
            </linearGradient>
          </defs>
          <path d="M50,50 L50,4 A46,46 0 0,1 87.8,26.2 Z" fill="url(#radarSweep)" />
        </g>
        <circle cx="50" cy="50" r="2" fill="#2dd4bf" />
      </svg>
    </div>
  );
}
