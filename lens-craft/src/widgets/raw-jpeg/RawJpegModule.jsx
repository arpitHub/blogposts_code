import { useEffect, useRef, useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import ToggleGroup from '../../components/ToggleGroup.jsx';
import { clamp255 } from '../../lib/canvasUtils.js';

const W = 236;
const H = 300;

// Sunset scene with detail hidden in both highlights (clouds) and shadows (rocks).
// We render it in linear "sensor" space with values well beyond 0..255 so RAW
// recovery has real data to pull from.
function sceneValue(x, y) {
  const u = x / W;
  const v = y / H;

  // sky gradient, very bright near the sun
  let r = 500 - v * 380;
  let g = 380 - v * 300;
  let b = 300 - v * 210;

  // sun disk + glow, heavily overexposed
  const dx = u - 0.62;
  const dy = v - 0.3;
  const d = Math.sqrt(dx * dx * 1.6 + dy * dy);
  const glow = Math.max(0, 1 - d * 2.6);
  r += glow * 900;
  g += glow * 800;
  b += glow * 600;

  // cloud bands with subtle structure (visible only when highlights recovered)
  const cloud = Math.sin(u * 21 + Math.sin(v * 31) * 2.2) * Math.sin(v * 17 + u * 5);
  if (v < 0.55) {
    const c = Math.max(0, cloud) * 90 * (1 - v);
    r -= c;
    g -= c * 0.85;
    b -= c * 0.6;
  }

  // sea
  if (v > 0.55) {
    r = 40 + (1 - v) * 90 + glow * 220;
    g = 45 + (1 - v) * 80 + glow * 190;
    b = 62 + (1 - v) * 90 + glow * 130;
    const wave = Math.sin(u * 60 + v * 90) * 8;
    r += wave;
    g += wave;
    b += wave;
  }

  // dark foreground rocks with structure hidden in shadow
  if (v > 0.78) {
    const rock = Math.sin(u * 33 + v * 55) * Math.cos(u * 17 - v * 23);
    const base = 6 + Math.max(0, rock) * 26;
    r = base * 1.15;
    g = base;
    b = base * 0.9;
  }

  return [r, g, b];
}

// JPEG stores 8 bits after a tone curve; deep shadows get very few levels,
// which is what posterizes when you push them in post.
function quantize(v) {
  const c = clamp255(v);
  return c < 48 ? Math.round(c / 10) * 10 : Math.round(c / 3) * 3;
}

export default function RawJpegModule({ module }) {
  const rawRef = useRef(null);
  const jpegRef = useRef(null);
  const [recovery, setRecovery] = useState(0); // 0..-30 -> EV*10
  const [zone, setZone] = useState('highlights');

  const ev = recovery / 10; // negative = pull highlights down / or push shadows up

  useEffect(() => {
    const rawCtx = rawRef.current.getContext('2d');
    const jpegCtx = jpegRef.current.getContext('2d');
    const rawImg = rawCtx.createImageData(W, H);
    const jpegImg = jpegCtx.createImageData(W, H);

    // Recovery direction depends on which zone we're rescuing.
    const mult = Math.pow(2, zone === 'highlights' ? -ev : ev);

    // Like a real Highlights/Shadows slider, the adjustment is weighted toward
    // the target tonal zone instead of shifting the whole frame.
    const weightFor = (lum) =>
      zone === 'highlights'
        ? Math.min(1, Math.max(0, (lum - 110) / 260))
        : Math.min(1, Math.max(0, 1 - lum / 150));

    let i = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const [r, g, b] = sceneValue(x, y);

        // RAW: full sensor data survives; recovery pulls from real values
        const rawLum = (r + g + b) / 3;
        const rawFactor = 1 + (mult - 1) * weightFor(rawLum);
        rawImg.data[i] = clamp255(r * rawFactor);
        rawImg.data[i + 1] = clamp255(g * rawFactor);
        rawImg.data[i + 2] = clamp255(b * rawFactor);
        rawImg.data[i + 3] = 255;

        // JPEG: clamped to 0..255 AND coarsely quantized at capture (8-bit +
        // compression) — clipped data is gone and shadows posterize when pushed
        const jr = quantize(r);
        const jg = quantize(g);
        const jb = quantize(b);
        const jpegLum = (jr + jg + jb) / 3;
        const jpegFactor = 1 + (mult - 1) * weightFor(jpegLum);
        jpegImg.data[i] = clamp255(jr * jpegFactor);
        jpegImg.data[i + 1] = clamp255(jg * jpegFactor);
        jpegImg.data[i + 2] = clamp255(jb * jpegFactor);
        jpegImg.data[i + 3] = 255;
        i += 4;
      }
    }
    rawCtx.putImageData(rawImg, 0, 0);
    jpegCtx.putImageData(jpegImg, 0, 0);
  }, [ev, zone]);

  const amount = Math.abs(ev);

  return (
    <ModulePage
      module={module}
      intro="RAW files keep everything the sensor saw — including detail your screen can't show. JPEGs throw the extremes away at the moment of capture. You only notice the difference when you try to edit."
      explanation={
        <p>
          Both frames start identical: blown-out sky, rocks crushed to black. Drag the recovery
          slider and watch them diverge. The <strong className="text-ink">RAW side</strong>{' '}
          reveals cloud structure and rock texture that was there all along; the{' '}
          <strong className="text-ink">JPEG side</strong> just turns grey mush — the data was
          deleted in-camera. The cost: RAW files are 3–5× larger and need processing before
          sharing. Most cameras can save both at once while you learn.
        </p>
      }
      challenge="Recover the highlights fully on both sides, then switch to shadows and do the same. Which zone survives JPEG better? (Shadows usually keep a little more — which is why the old JPEG-era advice was to underexpose slightly, the opposite of RAW-era advice.)"
    >
      <div className="flex flex-col gap-6">
        <div className="overflow-hidden rounded-2xl border border-line bg-panel">
          <div className="grid grid-cols-2">
            <div className="relative">
              <canvas ref={rawRef} width={W} height={H} className="block w-full" />
              <span className="absolute left-3 top-3 rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-[#52c97a]">
                RAW
              </span>
            </div>
            <div className="relative border-l border-line">
              <canvas ref={jpegRef} width={W} height={H} className="block w-full" />
              <span className="absolute left-3 top-3 rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-[#ff5a6a]">
                JPEG
              </span>
            </div>
          </div>
          <div className="flex flex-col gap-4 border-t border-line px-5 py-4 sm:flex-row sm:items-center">
            <ToggleGroup
              value={zone}
              onChange={(z) => {
                setZone(z);
                setRecovery(0);
              }}
              options={[
                { value: 'highlights', label: 'Rescue highlights' },
                { value: 'shadows', label: 'Rescue shadows' },
              ]}
            />
            <div className="flex-1">
              <Slider
                label={zone === 'highlights' ? 'Pull highlights down' : 'Push shadows up'}
                color="#35d0ba"
                value={recovery}
                min={0}
                max={30}
                display={`${amount.toFixed(1)} EV`}
                onChange={setRecovery}
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-xl border border-line bg-panel px-4 py-3 text-xs leading-relaxed text-ink-2">
            <span className="mb-1 block font-mono text-[10px] uppercase tracking-wide text-[#52c97a]">
              RAW at {amount.toFixed(1)} EV
            </span>
            {amount < 0.5
              ? 'Untouched — drag the slider to start recovering.'
              : zone === 'highlights'
                ? 'Cloud bands and sun edge re-appear: the sensor captured them, the file kept them.'
                : 'Rock texture lifts cleanly out of the dark with usable colour.'}
          </div>
          <div className="rounded-xl border border-line bg-panel px-4 py-3 text-xs leading-relaxed text-ink-2">
            <span className="mb-1 block font-mono text-[10px] uppercase tracking-wide text-[#ff5a6a]">
              JPEG at {amount.toFixed(1)} EV
            </span>
            {amount < 0.5
              ? 'Identical to RAW — until you try to edit it.'
              : zone === 'highlights'
                ? 'The sky just gets darker grey — the clipped whites hold no detail to reveal.'
                : 'Shadows lift but posterize and stay muddy; fine texture never comes back.'}
          </div>
        </div>
      </div>
    </ModulePage>
  );
}
