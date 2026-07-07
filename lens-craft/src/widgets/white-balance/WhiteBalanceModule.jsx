import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import WidgetShell, { Stat } from '../../components/WidgetShell.jsx';
import WbCanvas from './WbCanvas.jsx';

const PRESETS = [
  { k: 1900, label: 'Candle', icon: '🕯' },
  { k: 3200, label: 'Tungsten', icon: '💡' },
  { k: 4300, label: 'Fluorescent', icon: '🏢' },
  { k: 5500, label: 'Daylight', icon: '☀' },
  { k: 6500, label: 'Neutral', icon: '◎' },
  { k: 7500, label: 'Shade', icon: '⛅' },
  { k: 9500, label: 'Deep shade', icon: '🏔' },
];

export default function WhiteBalanceModule({ module }) {
  const [kelvin, setKelvin] = useState(6500);

  const cast =
    kelvin < 4500 ? 'Warm orange cast' : kelvin < 6000 ? 'Slightly warm' : kelvin <= 7000 ? 'Neutral' : kelvin <= 8500 ? 'Slightly cool' : 'Cool blue cast';

  const pct = ((kelvin - 1900) / (10000 - 1900)) * 100;

  return (
    <ModulePage
      module={module}
      intro="Light has a colour. Candles are orange, shade is blue, and your brain silently corrects for all of it — but your camera records what's actually there. White balance is how you tell it what 'white' meant."
      explanation={
        <p>
          Colour temperature is measured in <strong className="text-ink">Kelvin</strong>:
          candlelight ≈ 1900K (orange), tungsten bulbs ≈ 3200K, midday sun ≈ 5500K, open shade ≈
          7500K+ (blue). Watch the <strong className="text-ink">white mug</strong> in the scene —
          when it looks truly white, the white balance matches the light. Get it wrong and every
          colour in the photo shifts together. Shoot RAW and you can fix it losslessly later;
          shoot JPEG and the cast is baked in.
        </p>
      }
      challenge="Slide to 1900K and notice how your eyes 'accept' the orange after a few seconds — that's the adaptation your camera doesn't have. Then find the temperature where the mug and the grey cat look truly neutral. Check against the Neutral preset: how close were you?"
    >
      <WidgetShell
        preview={<WbCanvas kelvin={kelvin} />}
        previewFooter={
          <div className="flex items-center justify-between">
            <span className="font-mono text-sm text-accent">{kelvin}K</span>
            <span className="text-xs text-ink-2">{cast}</span>
          </div>
        }
        controls={
          <>
            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm font-medium text-ink-2">Colour temperature</span>
                <span className="rounded-md border border-accent px-2 py-0.5 font-mono text-sm text-accent">
                  {kelvin}K
                </span>
              </div>
              {/* the track itself is the lesson: orange -> white -> blue */}
              <input
                type="range"
                className="dial"
                style={{
                  '--slider-color': '#f3f2ee',
                  '--slider-fill': `${pct}%`,
                  background: 'linear-gradient(to right, #ff9d3c, #ffd9a0, #f8f8f4, #cfe0f8, #8ab4f0)',
                }}
                min={1900}
                max={10000}
                step={100}
                value={kelvin}
                onChange={(e) => setKelvin(Number(e.target.value))}
              />
              <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-3">
                <span>1900K candle</span>
                <span>10000K deep shade</span>
              </div>
            </div>

            <div>
              <div className="mb-2 font-mono text-[10px] uppercase tracking-wide text-ink-3">
                Presets (like your camera's WB menu)
              </div>
              <div className="flex flex-wrap gap-1.5">
                {PRESETS.map((p) => (
                  <button
                    key={p.k}
                    onClick={() => setKelvin(p.k)}
                    className={`rounded-lg border px-2.5 py-1.5 text-xs transition-colors ${
                      Math.abs(kelvin - p.k) < 150
                        ? 'border-accent text-accent'
                        : 'border-line text-ink-2 hover:border-ink-3'
                    }`}
                  >
                    {p.icon} {p.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 border-t border-line pt-4">
              <Stat label="The mug looks" value={kelvin >= 6000 && kelvin <= 7000 ? 'White ✓' : kelvin < 6000 ? 'Orange-ish' : 'Blue-ish'} />
              <Stat label="Fixable later?" value="RAW: yes · JPEG: partly" />
            </div>
          </>
        }
      />
    </ModulePage>
  );
}
