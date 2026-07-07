import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import ToggleGroup from '../../components/ToggleGroup.jsx';
import WidgetShell, { Stat } from '../../components/WidgetShell.jsx';
import IsoSceneCanvas from './IsoSceneCanvas.jsx';
import { ISO_STOPS } from '../exposure-triangle/exposureMath.js';

// For a fixed f/2.8, the shutter speed each ISO buys you in each context.
// One stop of ISO = one stop of shutter; night starts 10 stops darker.
const DAY_BASE_SHUTTER = ['1/2000', '1/4000', '1/8000', '1/8000', '1/8000', '1/8000', '1/8000', '1/8000'];
const NIGHT_BASE_SHUTTER = ['2"', '1"', '1/2', '1/4', '1/8', '1/15', '1/30', '1/60'];

export default function IsoNoiseModule({ module }) {
  const [isoIndex, setIsoIndex] = useState(0);
  const [night, setNight] = useState(false);

  const iso = ISO_STOPS[isoIndex];
  const noiseAmount = isoIndex / (ISO_STOPS.length - 1);
  const shutterYouGet = night ? NIGHT_BASE_SHUTTER[isoIndex] : DAY_BASE_SHUTTER[isoIndex];
  const nightHandheld = night && isoIndex >= 7;

  return (
    <ModulePage
      module={module}
      intro="ISO doesn't add light — it amplifies the signal your sensor already captured, and amplifies its imperfections along with it. That's the grain."
      explanation={
        <p>
          In daylight there's so much signal that ISO 100 is all you ever need — raising it just
          adds noise for no benefit. At night the calculus flips: at ISO 100 you'd need a
          2-second exposure (tripod, frozen subjects), but ISO 12800 buys you{' '}
          <strong className="text-ink">1/60 handheld</strong> at the cost of visible grain.{' '}
          <strong className="text-ink">A noisy sharp photo beats a clean blurry one</strong> —
          that's the whole trade.
        </p>
      }
      challenge="Switch to night and raise ISO until the 'shutter you get' readout says you can shoot handheld (1/60). Look at how much grain you accepted. Now switch back to day at that same ISO — same grain, zero benefit. This is why ISO is the dial you touch last."
    >
      <WidgetShell
        preview={<IsoSceneCanvas night={night} noiseAmount={noiseAmount} />}
        previewFooter={
          <div className="flex items-center justify-between">
            <ToggleGroup
              value={night ? 'night' : 'day'}
              onChange={(v) => setNight(v === 'night')}
              options={[
                { value: 'day', label: '☀ Day' },
                { value: 'night', label: '☾ Night' },
              ]}
            />
            <span className="font-mono text-xs text-ink-2">
              f/2.8 · shutter you get: <span className="text-accent">{shutterYouGet}s</span>
            </span>
          </div>
        }
        controls={
          <>
            <Slider
              label="ISO"
              color="#ff5a6a"
              value={isoIndex}
              min={0}
              max={ISO_STOPS.length - 1}
              display={`ISO ${iso}`}
              onChange={setIsoIndex}
            />

            <div className="grid grid-cols-2 gap-2 border-t border-line pt-4">
              <Stat label="Grain" value={noiseAmount < 0.15 ? 'Clean' : noiseAmount < 0.45 ? 'Slight' : noiseAmount < 0.75 ? 'Visible' : 'Heavy'} />
              <Stat
                label={night ? 'Handheld at night?' : 'Benefit in daylight?'}
                value={night ? (nightHandheld ? 'Yes — 1/60' : 'No — too slow') : 'None'}
              />
            </div>

            <div className="rounded-lg bg-panel-2 px-3 py-2.5 text-xs leading-relaxed text-ink-2">
              {night
                ? 'Night: every stop of ISO doubles your shutter speed. Grain is the price of a sharp handheld shot.'
                : 'Day: there is already plenty of light. Raising ISO here only adds grain — keep it at base ISO.'}
            </div>

            <p className="text-xs leading-relaxed text-ink-3">
              Modern cameras are usable up to ISO 3200–6400. Set aperture and shutter for the
              photo you want, then raise ISO only as far as the meter demands.
            </p>
          </>
        }
      />
    </ModulePage>
  );
}
