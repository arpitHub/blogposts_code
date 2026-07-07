import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import ToggleGroup from '../../components/ToggleGroup.jsx';
import {
  APERTURE_STOPS,
  SHUTTER_STOPS,
  ISO_STOPS,
} from '../exposure-triangle/exposureMath.js';

// Each genre = a point in the exposure-triangle space plus the reasoning.
const GENRES = {
  portrait: {
    label: 'Portrait',
    icon: '👤',
    settings: { aperture: 1, shutter: 4, iso: 1 }, // f/2, 1/250, ISO 200
    priority: 'Aperture first',
    why: 'A wide aperture melts the background so the person is unmissable. Shutter only needs to beat subject sway (~1/250); ISO stays low in decent light.',
    tips: [
      'Focus on the nearest eye — always.',
      '85–135mm equivalent flatters faces; 24mm up close distorts noses.',
      'Overcast days are free softboxes. Midday sun is the enemy.',
      'Leave headroom in the frame, but not too much — crop at the chest or waist, never at joints.',
    ],
    tryThis: 'Golden hour, one person, f/2 or your widest, subject 3m from any background. Take the same photo at f/8 and compare how "snapshot" it suddenly feels.',
  },
  landscape: {
    label: 'Landscape',
    icon: '🏔',
    settings: { aperture: 6, shutter: 7, iso: 0 }, // f/11, 1/30, ISO 100
    priority: 'Aperture first (the other way)',
    why: 'You want front-to-back sharpness, so stop down to f/8–f/13. On a tripod, shutter speed is free — let it fall where it lands and keep ISO at base for maximum detail.',
    tips: [
      'Sharpest light is 30 min either side of sunrise/sunset.',
      'A foreground object (rock, flowers, path) gives the eye an entrance.',
      'Check the histogram — skies love to clip.',
      'f/16+ costs sharpness to diffraction; f/8–f/11 is usually the sweet spot.',
    ],
    tryThis: 'Find a scene with something 1m away and something 1km away. Shoot f/2.8 vs f/11 focused a third into the scene, and inspect corner sharpness on both.',
  },
  astro: {
    label: 'Astro',
    icon: '✨',
    settings: { aperture: 0, shutter: 12, iso: 5 }, // f/1.4, 1", ISO 3200 (representing 15-25s in spirit)
    priority: 'Everything maxed',
    why: 'Starlight is absurdly dim: you need every photon. Widest aperture, the longest shutter before stars trail (500 ÷ focal length ≈ seconds), and ISO 1600–6400 — noise is the entry fee.',
    tips: [
      'Manual focus on a bright star with magnified live view — AF is useless in the dark.',
      'The "500 rule": 500 ÷ focal length = max seconds before star trails.',
      'Moonless nights only. The moon is a giant streetlight.',
      'Shoot RAW: you will be pushing shadows hard, and JPEG falls apart.',
    ],
    tryThis: 'Tonight, even in a city: 20s, f/2.8, ISO 3200, pointed at the darkest patch of sky. You will capture stars your eyes cannot see — this is the photo that hooks people on astro forever.',
  },
};

// slider positions (0-1) for the mini triangle readout
function settingFractions(s) {
  return {
    aperture: 1 - s.aperture / (APERTURE_STOPS.length - 1), // 1 = wide open
    shutter: s.shutter / (SHUTTER_STOPS.length - 1), // 1 = longest
    iso: s.iso / (ISO_STOPS.length - 1), // 1 = highest
  };
}

export default function GenreGuidesModule({ module }) {
  const [genre, setGenre] = useState('portrait');
  const g = GENRES[genre];
  const f = settingFractions(g.settings);

  return (
    <ModulePage
      module={module}
      intro="Every genre is just a different answer to the same question: which corner of the exposure triangle matters most? Learn the reasoning, not the recipe."
      explanation={
        <p>
          The settings card shows a <em>starting point</em>, not gospel — light changes
          everything. What transfers between shoots is the{' '}
          <strong className="text-ink">priority order</strong>: portraits protect aperture,
          landscapes protect aperture in the opposite direction, astro sacrifices everything
          for light. Once you can say <em>why</em> each genre picks its corner, you can walk
          into any situation and derive your own settings.
        </p>
      }
      challenge={g.tryThis}
    >
      <div className="flex flex-col gap-6">
        <ToggleGroup
          value={genre}
          onChange={setGenre}
          options={Object.entries(GENRES).map(([k, v]) => ({
            value: k,
            label: `${v.icon} ${v.label}`,
          }))}
        />

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_1.2fr]">
          {/* settings card, styled like a camera top-plate LCD */}
          <div className="rounded-2xl border border-line bg-panel p-5">
            <div className="mb-4 flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase tracking-wider text-ink-3">
                Starting settings
              </span>
              <span className="rounded-full border border-accent/40 px-2.5 py-0.5 font-mono text-[11px] text-accent">
                {g.priority}
              </span>
            </div>
            <div className="mb-5 grid grid-cols-3 gap-2 rounded-xl bg-black/40 px-3 py-4 text-center font-mono">
              <div>
                <div className="text-lg text-[#ffb020]">f/{APERTURE_STOPS[g.settings.aperture]}</div>
                <div className="mt-1 text-[10px] uppercase text-ink-3">aperture</div>
              </div>
              <div>
                <div className="text-lg text-[#35d0ba]">{SHUTTER_STOPS[g.settings.shutter].label}</div>
                <div className="mt-1 text-[10px] uppercase text-ink-3">shutter</div>
              </div>
              <div>
                <div className="text-lg text-[#ff5a6a]">{ISO_STOPS[g.settings.iso]}</div>
                <div className="mt-1 text-[10px] uppercase text-ink-3">iso</div>
              </div>
            </div>

            {/* where this genre lives on each axis */}
            <div className="flex flex-col gap-3">
              <AxisBar label="Background blur" frac={f.aperture} color="#ffb020" left="deep focus" right="melted" />
              <AxisBar label="Motion capture" frac={f.shutter} color="#35d0ba" left="frozen" right="long exposure" />
              <AxisBar label="Light amplification" frac={f.iso} color="#ff5a6a" left="clean" right="grainy" />
            </div>

            <p className="mt-4 text-xs leading-relaxed text-ink-2">{g.why}</p>
          </div>

          {/* field tips */}
          <div className="rounded-2xl border border-line bg-panel p-5">
            <span className="mb-3 block font-mono text-[10px] uppercase tracking-wider text-ink-3">
              Field notes — {g.label.toLowerCase()}
            </span>
            <ul className="flex flex-col gap-3">
              {g.tips.map((tip, i) => (
                <li key={i} className="flex gap-3 text-sm leading-relaxed text-ink-2">
                  <span className="mt-0.5 font-mono text-xs text-accent">{String(i + 1).padStart(2, '0')}</span>
                  {tip}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </ModulePage>
  );
}

function AxisBar({ label, frac, color, left, right }) {
  return (
    <div>
      <div className="mb-1 flex justify-between font-mono text-[10px] text-ink-3">
        <span>{left}</span>
        <span className="text-ink-2">{label}</span>
        <span>{right}</span>
      </div>
      <div className="relative h-2 rounded-full bg-panel-2">
        <div
          className="absolute top-1/2 h-3.5 w-3.5 -translate-y-1/2 -translate-x-1/2 rounded-full border-2 border-void transition-all duration-300"
          style={{ left: `${8 + frac * 84}%`, background: color }}
        />
      </div>
    </div>
  );
}
