import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import SceneCanvas from './SceneCanvas.jsx';
import ExposureMeter from './ExposureMeter.jsx';
import {
  APERTURE_STOPS,
  SHUTTER_STOPS,
  ISO_STOPS,
  formatAperture,
  formatShutter,
  formatIso,
  computeStops,
  brightnessMultiplier,
  apertureToBlurAmount,
  shutterToMotionAmount,
  isoToNoiseAmount,
} from './exposureMath.js';

const ACCENT_APERTURE = '#ffb020';
const ACCENT_SHUTTER = '#35d0ba';
const ACCENT_ISO = '#ff5a6a';

export default function ExposureTriangleModule({ module }) {
  const [apertureIndex, setApertureIndex] = useState(1); // f/2
  const [shutterIndex, setShutterIndex] = useState(2); // 1/1000
  const [isoIndex, setIsoIndex] = useState(0); // ISO 100

  const stops = computeStops({ apertureIndex, shutterIndex, isoIndex });
  const brightness = brightnessMultiplier(stops.totalStops);
  const blurAmount = apertureToBlurAmount(apertureIndex);
  const motionAmount = shutterToMotionAmount(shutterIndex);
  const noiseAmount = isoToNoiseAmount(isoIndex);

  return (
    <ModulePage
      module={module}
      intro="Every photo is a compromise between three controls. Change one, and to keep the same brightness you must change another — but each one also leaves its own fingerprint on the image."
      explanation={
        <p>
          <strong className="text-ink">Aperture</strong> controls how much of the scene is in
          focus (depth of field) as well as light.{' '}
          <strong className="text-ink">Shutter speed</strong> controls whether motion freezes or
          blurs, as well as light. <strong className="text-ink">ISO</strong> makes the sensor more
          sensitive to light at the cost of grain. The exposure meter below shows whether your
          combination is too dark, too bright, or balanced — there's no single "correct" setting,
          only trade-offs.
        </p>
      }
      challenge={
        'Set the shutter to 1" (one full second) to blur the car completely, then raise ISO ' +
        'until the exposure meter reads close to 0 EV again. Now try the opposite: freeze the ' +
        'car at 1/4000 using only aperture and ISO to compensate.'
      }
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.3fr_1fr]">
        <div className="overflow-hidden rounded-2xl border border-line bg-panel">
          <div className="aspect-[8/5] w-full">
            <SceneCanvas
              blurAmount={blurAmount}
              motionAmount={motionAmount}
              noiseAmount={noiseAmount}
              brightness={brightness}
            />
          </div>
          <div className="border-t border-line px-4 py-3">
            <ExposureMeter totalStops={stops.totalStops} />
          </div>
        </div>

        <div className="flex flex-col gap-6 rounded-2xl border border-line bg-panel p-5">
          <Slider
            label="Aperture"
            color={ACCENT_APERTURE}
            value={apertureIndex}
            min={0}
            max={APERTURE_STOPS.length - 1}
            display={formatAperture(apertureIndex)}
            onChange={setApertureIndex}
          />
          <Slider
            label="Shutter speed"
            color={ACCENT_SHUTTER}
            value={shutterIndex}
            min={0}
            max={SHUTTER_STOPS.length - 1}
            display={formatShutter(shutterIndex)}
            onChange={setShutterIndex}
          />
          <Slider
            label="ISO"
            color={ACCENT_ISO}
            value={isoIndex}
            min={0}
            max={ISO_STOPS.length - 1}
            display={formatIso(isoIndex)}
            onChange={setIsoIndex}
          />

          <div className="mt-1 grid grid-cols-3 gap-2 border-t border-line pt-4 text-center">
            <Stat label="Depth of field" value={blurAmount < 0.3 ? 'Deep' : blurAmount < 0.65 ? 'Medium' : 'Shallow'} />
            <Stat label="Motion" value={motionAmount < 0.15 ? 'Frozen' : motionAmount < 0.5 ? 'Slight blur' : 'Streaked'} />
            <Stat label="Grain" value={noiseAmount < 0.15 ? 'Clean' : noiseAmount < 0.5 ? 'Visible' : 'Heavy'} />
          </div>
        </div>
      </div>
    </ModulePage>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg bg-panel-2 px-2 py-2">
      <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{label}</div>
      <div className="mt-0.5 text-sm font-medium text-ink">{value}</div>
    </div>
  );
}
