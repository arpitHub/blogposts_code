import { useEffect, useMemo, useRef, useState } from 'react';
import { Area, AreaChart, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from 'recharts';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import WidgetShell, { Stat } from '../../components/WidgetShell.jsx';
import { applyExposureAndNoise, computeHistogram, clamp255 } from '../../lib/canvasUtils.js';

const W = 480;
const H = 300;
const BUCKETS = 64;

function drawLandscape(ctx) {
  // deliberately wide tonal range: bright sky, white church, deep shadows
  const sky = ctx.createLinearGradient(0, 0, 0, H * 0.55);
  sky.addColorStop(0, '#b8cce0');
  sky.addColorStop(1, '#e8d8c0');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H * 0.55);

  // sun
  const sun = ctx.createRadialGradient(W * 0.78, H * 0.18, 4, W * 0.78, H * 0.18, 46);
  sun.addColorStop(0, '#fffbe8');
  sun.addColorStop(0.5, '#ffedb8');
  sun.addColorStop(1, 'rgba(255,237,184,0)');
  ctx.fillStyle = sun;
  ctx.fillRect(W * 0.6, 0, 180, 130);

  // hills
  ctx.fillStyle = '#5a6a4a';
  ctx.beginPath();
  ctx.moveTo(0, H * 0.55);
  ctx.quadraticCurveTo(W * 0.25, H * 0.42, W * 0.5, H * 0.52);
  ctx.quadraticCurveTo(W * 0.75, H * 0.6, W, H * 0.5);
  ctx.lineTo(W, H * 0.62);
  ctx.lineTo(0, H * 0.62);
  ctx.closePath();
  ctx.fill();

  // white church — the highlight anchor
  ctx.fillStyle = '#f2efe8';
  ctx.fillRect(W * 0.22, H * 0.34, 44, 62);
  ctx.beginPath();
  ctx.moveTo(W * 0.22 - 4, H * 0.34);
  ctx.lineTo(W * 0.22 + 22, H * 0.25);
  ctx.lineTo(W * 0.22 + 48, H * 0.34);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#c8c2b4';
  ctx.fillRect(W * 0.22 + 16, H * 0.42, 12, 20);

  // foreground meadow
  const meadow = ctx.createLinearGradient(0, H * 0.62, 0, H);
  meadow.addColorStop(0, '#46543a');
  meadow.addColorStop(1, '#232b1c');
  ctx.fillStyle = meadow;
  ctx.fillRect(0, H * 0.62, W, H * 0.38);

  // dark forest edge — the shadow anchor
  ctx.fillStyle = '#141a10';
  for (let i = 0; i < 8; i++) {
    const x = W * 0.62 + i * 24;
    ctx.beginPath();
    ctx.moveTo(x, H * 0.78);
    ctx.lineTo(x + 11, H * 0.52);
    ctx.lineTo(x + 22, H * 0.78);
    ctx.closePath();
    ctx.fill();
  }
  ctx.fillRect(W * 0.6, H * 0.74, W * 0.4, H * 0.26);
}

export default function HistogramModule({ module }) {
  const canvasRef = useRef(null);
  const baseRef = useRef(null);
  const [ev, setEv] = useState(0); // -3..+3
  const [hist, setHist] = useState(() => new Array(BUCKETS).fill(0));
  const [clip, setClip] = useState({ shadows: 0, highlights: 0 });

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!baseRef.current) {
      drawLandscape(ctx);
      baseRef.current = ctx.getImageData(0, 0, W, H);
    }

    const mult = Math.pow(2, ev);
    const src = baseRef.current.data;
    const out = ctx.createImageData(W, H);
    const dst = out.data;
    let shadowClipped = 0;
    let highlightClipped = 0;
    for (let i = 0; i < src.length; i += 4) {
      const r = src[i] * mult;
      const g = src[i + 1] * mult;
      const b = src[i + 2] * mult;
      dst[i] = clamp255(r);
      dst[i + 1] = clamp255(g);
      dst[i + 2] = clamp255(b);
      dst[i + 3] = 255;
      const lum = 0.2126 * dst[i] + 0.7152 * dst[i + 1] + 0.0722 * dst[i + 2];
      if (lum <= 2) shadowClipped++;
      if (lum >= 253) highlightClipped++;
    }
    ctx.putImageData(out, 0, 0);

    const total = src.length / 4;
    setHist(computeHistogram(ctx, W, H, BUCKETS));
    setClip({
      shadows: Math.round((shadowClipped / total) * 100),
      highlights: Math.round((highlightClipped / total) * 100),
    });
  }, [ev]);

  const chartData = useMemo(
    () => hist.map((v, i) => ({ bucket: i, count: Math.pow(v, 0.6) })),
    [hist]
  );

  return (
    <ModulePage
      module={module}
      intro="The histogram is a bar chart of your photo's brightness: shadows on the left, highlights on the right. It's the only exposure tool that doesn't lie — screens do, especially in sunlight."
      explanation={
        <p>
          There's no single "correct" histogram shape — a night scene should lean left, a snow
          scene right. What matters is <strong className="text-ink">clipping</strong>: bars
          slammed against either wall mean detail that's gone forever — pure black shadows or
          blown white highlights that no editing can recover. The classic advice is to expose
          as bright as possible <em>without</em> touching the right wall, because shadows
          recover far better than highlights.
        </p>
      }
      challenge="Push the exposure up until the highlight warning fires — notice the sky and church go featureless white while the histogram piles against the right wall. Come back down to the brightest setting with 0% highlight clipping. That's the 'expose to the right' sweet spot."
    >
      <WidgetShell
        preview={<canvas ref={canvasRef} width={W} height={H} className="h-full w-full" />}
        previewFooter={
          <div>
            {/* live histogram */}
            <div className="relative h-24 overflow-hidden rounded-md bg-panel-2 pt-1">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 4, right: 2, bottom: 0, left: 2 }}>
                  <defs>
                    <linearGradient id="histFill" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#5aa9ff" stopOpacity={0.8} />
                      <stop offset="30%" stopColor="#a8a7ad" stopOpacity={0.7} />
                      <stop offset="70%" stopColor="#a8a7ad" stopOpacity={0.7} />
                      <stop offset="100%" stopColor="#ffb020" stopOpacity={0.85} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="bucket" hide />
                  <YAxis hide />
                  <ReferenceLine x={Math.floor(BUCKETS / 3)} stroke="var(--color-line)" />
                  <ReferenceLine x={Math.floor((BUCKETS * 2) / 3)} stroke="var(--color-line)" />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke="rgba(243,242,238,0.5)"
                    strokeWidth={1}
                    fill="url(#histFill)"
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
              {/* clipping walls */}
              <div
                className="absolute inset-y-0 left-0 w-1 rounded-l"
                style={{ background: clip.shadows > 1 ? '#5aa9ff' : 'transparent' }}
              />
              <div
                className="absolute inset-y-0 right-0 w-1 rounded-r"
                style={{ background: clip.highlights > 1 ? '#ff5a4d' : 'transparent' }}
              />
            </div>
            <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-3">
              <span>◀ shadows</span>
              <span>midtones</span>
              <span>highlights ▶</span>
            </div>
          </div>
        }
        controls={
          <>
            <Slider
              label="Exposure"
              color="#ffb020"
              value={ev * 10}
              min={-30}
              max={30}
              step={1}
              display={`${ev >= 0 ? '+' : ''}${ev.toFixed(1)} EV`}
              onChange={(v) => setEv(v / 10)}
            />

            <div className="grid grid-cols-2 gap-2 border-t border-line pt-4">
              <Stat
                label="Shadow clipping"
                value={clip.shadows > 1 ? `${clip.shadows}% lost` : 'None'}
              />
              <Stat
                label="Highlight clipping"
                value={clip.highlights > 1 ? `${clip.highlights}% lost` : 'None'}
              />
            </div>

            {(clip.highlights > 1 || clip.shadows > 1) && (
              <div
                className="rounded-lg border px-3 py-2.5 text-xs leading-relaxed"
                style={{
                  borderColor: clip.highlights > 1 ? 'rgba(255,90,77,0.4)' : 'rgba(90,169,255,0.4)',
                  color: clip.highlights > 1 ? '#ff8a7d' : '#8ec3ff',
                }}
              >
                {clip.highlights > 1
                  ? '⚠ Highlights are clipping — that detail is unrecoverable, even in RAW.'
                  : '⚠ Shadows are clipping — blacks are crushed to pure 0.'}
              </div>
            )}

            <p className="text-xs leading-relaxed text-ink-3">
              Enable "blinkies" (highlight alert) on your camera — it overlays flashing marks on
              clipped areas during playback, which is this same warning in the field.
            </p>
          </>
        }
      />
    </ModulePage>
  );
}
