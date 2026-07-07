import { useEffect, useRef, useState, useCallback } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import { clamp255 } from '../../lib/canvasUtils.js';

const W = 480;
const H = 300;

// A flat, slightly underexposed straight-out-of-camera shot begging for an edit.
function drawBase(ctx) {
  const sky = ctx.createLinearGradient(0, 0, 0, H * 0.6);
  sky.addColorStop(0, '#5a6470');
  sky.addColorStop(1, '#8a8078');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H * 0.6);

  // hazy sun
  const sun = ctx.createRadialGradient(W * 0.28, H * 0.22, 4, W * 0.28, H * 0.22, 60);
  sun.addColorStop(0, 'rgba(220,205,180,0.7)');
  sun.addColorStop(1, 'rgba(220,205,180,0)');
  ctx.fillStyle = sun;
  ctx.fillRect(0, 0, W, H * 0.5);

  // lake
  const lake = ctx.createLinearGradient(0, H * 0.6, 0, H * 0.82);
  lake.addColorStop(0, '#5c6468');
  lake.addColorStop(1, '#42484e');
  ctx.fillStyle = lake;
  ctx.fillRect(0, H * 0.6, W, H * 0.22);

  // mountains reflected
  ctx.fillStyle = '#4a5052';
  ctx.beginPath();
  ctx.moveTo(W * 0.4, H * 0.6);
  ctx.lineTo(W * 0.62, H * 0.3);
  ctx.lineTo(W * 0.86, H * 0.6);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = 'rgba(74,80,82,0.5)';
  ctx.beginPath();
  ctx.moveTo(W * 0.44, H * 0.6);
  ctx.lineTo(W * 0.62, H * 0.78);
  ctx.lineTo(W * 0.82, H * 0.6);
  ctx.closePath();
  ctx.fill();

  // canoe with paddler
  ctx.fillStyle = '#5c4434';
  ctx.beginPath();
  ctx.moveTo(W * 0.14, H * 0.72);
  ctx.quadraticCurveTo(W * 0.22, H * 0.78, W * 0.3, H * 0.72);
  ctx.lineTo(W * 0.28, H * 0.7);
  ctx.lineTo(W * 0.16, H * 0.7);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#3c342c';
  ctx.beginPath();
  ctx.arc(W * 0.22, H * 0.665, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#3c342c';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(W * 0.225, H * 0.68);
  ctx.lineTo(W * 0.27, H * 0.75);
  ctx.stroke();

  // foreground pines
  ctx.fillStyle = '#2c3428';
  for (const [x, s] of [[0.04, 1], [0.94, 0.85], [0.99, 1.1]]) {
    for (let t = 0; t < 3; t++) {
      const w = (34 - t * 8) * s;
      const y0 = H * (0.98 - t * 0.13 * s);
      ctx.beginPath();
      ctx.moveTo(W * x - w / 2, y0);
      ctx.lineTo(W * x, y0 - 46 * s);
      ctx.lineTo(W * x + w / 2, y0);
      ctx.closePath();
      ctx.fill();
    }
  }
  // ground strip
  ctx.fillStyle = '#32302a';
  ctx.fillRect(0, H * 0.82, W, H * 0.18);
}

function editPixels(src, dst, { exposure, contrast, saturation }) {
  const expMult = Math.pow(2, exposure);
  const c = contrast; // -1..1
  for (let i = 0; i < src.length; i += 4) {
    let r = src[i] * expMult;
    let g = src[i + 1] * expMult;
    let b = src[i + 2] * expMult;

    // contrast: S-curve around mid grey
    if (c !== 0) {
      r = 128 + (r - 128) * (1 + c) + Math.sin(((r - 128) / 128) * Math.PI) * c * 14;
      g = 128 + (g - 128) * (1 + c) + Math.sin(((g - 128) / 128) * Math.PI) * c * 14;
      b = 128 + (b - 128) * (1 + c) + Math.sin(((b - 128) / 128) * Math.PI) * c * 14;
    }

    // saturation around per-pixel luminance
    if (saturation !== 0) {
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      r = lum + (r - lum) * (1 + saturation);
      g = lum + (g - lum) * (1 + saturation);
      b = lum + (b - lum) * (1 + saturation);
    }

    dst[i] = clamp255(r);
    dst[i + 1] = clamp255(g);
    dst[i + 2] = clamp255(b);
    dst[i + 3] = 255;
  }
}

export default function PostProcessingModule({ module }) {
  const canvasRef = useRef(null);
  const baseRef = useRef(null);
  const containerRef = useRef(null);
  const [exposure, setExposure] = useState(0);
  const [contrast, setContrast] = useState(0);
  const [saturation, setSaturation] = useState(0);
  const [wipe, setWipe] = useState(0.5);
  const draggingRef = useRef(false);

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d');
    if (!baseRef.current) {
      drawBase(ctx);
      baseRef.current = ctx.getImageData(0, 0, W, H);
    }
    const out = ctx.createImageData(W, H);
    editPixels(baseRef.current.data, out.data, {
      exposure,
      contrast,
      saturation,
    });
    ctx.putImageData(out, 0, 0);
    // paint the ORIGINAL back over the left portion, up to the wipe line
    const wipeX = Math.round(wipe * W);
    if (wipeX > 0) {
      ctx.putImageData(baseRef.current, 0, 0, 0, 0, wipeX, H);
    }
    // wipe handle line
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.fillRect(wipeX - 1, 0, 2, H);
  }, [exposure, contrast, saturation, wipe]);

  const updateWipe = useCallback((e) => {
    const rect = containerRef.current.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    setWipe(Math.min(1, Math.max(0, x / rect.width)));
  }, []);

  const applyPreset = (ex, co, sa) => {
    setExposure(ex);
    setContrast(co);
    setSaturation(sa);
  };

  return (
    <ModulePage
      module={module}
      intro="Editing isn't cheating — the camera already made choices for you; editing just puts you in charge of them. Three sliders do most of the work: exposure, contrast, saturation."
      explanation={
        <p>
          <strong className="text-ink">Exposure</strong> sets overall brightness — fix the
          meter's caution here first. <strong className="text-ink">Contrast</strong> spreads
          tones apart: flat files gain punch, hazy light gains depth.{' '}
          <strong className="text-ink">Saturation</strong> scales colour intensity — a little
          goes a long way; +20% reads as 'vivid', +60% reads as 'radioactive'. Drag the white
          line across the image to compare against the straight-out-of-camera original.
        </p>
      }
      challenge="This file is flat and half a stop dark — a typical safe camera exposure. Try +0.5 EV, +25 contrast, +20 saturation, then wipe the divider back and forth. That 10-second edit is 90% of what 'editing your photos' means. Then overdo everything to see why restraint is a skill."
    >
      <div className="flex flex-col gap-6">
        <div className="overflow-hidden rounded-2xl border border-line bg-panel">
          <div
            ref={containerRef}
            className="relative aspect-[8/5] w-full cursor-ew-resize touch-none select-none"
            onMouseDown={(e) => {
              draggingRef.current = true;
              updateWipe(e);
            }}
            onMouseMove={(e) => draggingRef.current && updateWipe(e)}
            onMouseUp={() => (draggingRef.current = false)}
            onMouseLeave={() => (draggingRef.current = false)}
            onTouchStart={(e) => {
              draggingRef.current = true;
              updateWipe(e);
            }}
            onTouchMove={(e) => draggingRef.current && updateWipe(e)}
            onTouchEnd={() => (draggingRef.current = false)}
          >
            <canvas ref={canvasRef} width={W} height={H} className="h-full w-full" />
            <span className="pointer-events-none absolute left-3 top-3 rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-ink-2">
              before
            </span>
            <span className="pointer-events-none absolute right-3 top-3 rounded bg-black/60 px-2 py-0.5 font-mono text-[11px] text-accent">
              after
            </span>
            {/* drag grip on the wipe line */}
            <div
              className="pointer-events-none absolute top-1/2 grid h-8 w-8 -translate-y-1/2 -translate-x-1/2 place-items-center rounded-full border border-white/60 bg-black/50 text-[10px] text-white"
              style={{ left: `${wipe * 100}%` }}
            >
              ⇔
            </div>
          </div>
          <div className="grid grid-cols-1 gap-5 border-t border-line px-5 py-4 sm:grid-cols-3">
            <Slider
              label="Exposure"
              color="#ffb020"
              value={exposure * 10}
              min={-15}
              max={15}
              display={`${exposure >= 0 ? '+' : ''}${exposure.toFixed(1)} EV`}
              onChange={(v) => setExposure(v / 10)}
            />
            <Slider
              label="Contrast"
              color="#35d0ba"
              value={contrast * 100}
              min={-60}
              max={60}
              display={`${contrast >= 0 ? '+' : ''}${Math.round(contrast * 100)}`}
              onChange={(v) => setContrast(v / 100)}
            />
            <Slider
              label="Saturation"
              color="#ff5a6a"
              value={saturation * 100}
              min={-100}
              max={100}
              display={`${saturation >= 0 ? '+' : ''}${Math.round(saturation * 100)}%`}
              onChange={(v) => setSaturation(v / 100)}
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-wide text-ink-3">
            One-click looks:
          </span>
          <PresetButton label="SOOC (reset)" onClick={() => applyPreset(0, 0, 0)} />
          <PresetButton label="Natural pop" onClick={() => applyPreset(0.5, 0.25, 0.2)} />
          <PresetButton label="Moody" onClick={() => applyPreset(-0.3, 0.4, -0.25)} />
          <PresetButton label="Instagram 2012" onClick={() => applyPreset(0.8, 0.55, 0.85)} />
        </div>
      </div>
    </ModulePage>
  );
}

function PresetButton({ label, onClick }) {
  return (
    <button
      onClick={onClick}
      className="rounded-lg border border-line px-3 py-1.5 text-xs text-ink-2 transition-colors hover:border-accent hover:text-accent"
    >
      {label}
    </button>
  );
}
