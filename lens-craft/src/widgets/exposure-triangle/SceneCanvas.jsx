import { useEffect, useRef } from 'react';

const W = 480;
const H = 300;

// Fixed "bokeh light" positions/colors so they don't reshuffle on every render.
const BOKEH_LIGHTS = [
  { x: 0.08, y: 0.22, r: 7, hue: '#ffd27a' },
  { x: 0.18, y: 0.38, r: 5, hue: '#ff9d6c' },
  { x: 0.28, y: 0.16, r: 6, hue: '#ffe08a' },
  { x: 0.4, y: 0.3, r: 8, hue: '#ffb0a0' },
  { x: 0.52, y: 0.14, r: 5, hue: '#ffd27a' },
  { x: 0.63, y: 0.24, r: 7, hue: '#ff9d6c' },
  { x: 0.74, y: 0.12, r: 6, hue: '#ffe9b0' },
  { x: 0.85, y: 0.28, r: 8, hue: '#ffb0a0' },
  { x: 0.93, y: 0.15, r: 5, hue: '#ffd27a' },
  { x: 0.15, y: 0.1, r: 4, hue: '#ffe9b0' },
  { x: 0.48, y: 0.08, r: 4, hue: '#ff9d6c' },
  { x: 0.8, y: 0.08, r: 4, hue: '#ffe08a' },
];

function drawScene(ctx, { blurAmount, motionAmount, carX }) {
  ctx.clearRect(0, 0, W, H);

  // sky
  const sky = ctx.createLinearGradient(0, 0, 0, H * 0.62);
  sky.addColorStop(0, '#1a2740');
  sky.addColorStop(1, '#3f4f6b');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H * 0.62);

  // sun
  ctx.save();
  ctx.filter = `blur(${2 + blurAmount * 3}px)`;
  const sunGrad = ctx.createRadialGradient(W * 0.82, H * 0.18, 2, W * 0.82, H * 0.18, 40);
  sunGrad.addColorStop(0, 'rgba(255,225,170,0.95)');
  sunGrad.addColorStop(1, 'rgba(255,225,170,0)');
  ctx.fillStyle = sunGrad;
  ctx.beginPath();
  ctx.arc(W * 0.82, H * 0.18, 40, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // distant bokeh lights (background) — the classic shallow-DOF demonstration
  ctx.save();
  const bgBlurPx = blurAmount * 16;
  ctx.filter = bgBlurPx > 0.3 ? `blur(${bgBlurPx}px)` : 'none';
  for (const light of BOKEH_LIGHTS) {
    const radius = light.r * (1 + blurAmount * 1.4);
    const grad = ctx.createRadialGradient(
      light.x * W,
      light.y * H,
      0,
      light.x * W,
      light.y * H,
      radius
    );
    grad.addColorStop(0, light.hue);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(light.x * W, light.y * H, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // hills silhouette
  ctx.fillStyle = '#141b26';
  ctx.beginPath();
  ctx.moveTo(0, H * 0.55);
  ctx.bezierCurveTo(W * 0.2, H * 0.42, W * 0.35, H * 0.5, W * 0.5, H * 0.46);
  ctx.bezierCurveTo(W * 0.68, H * 0.42, W * 0.8, H * 0.52, W, H * 0.47);
  ctx.lineTo(W, H * 0.62);
  ctx.lineTo(0, H * 0.62);
  ctx.closePath();
  ctx.fill();

  // road
  ctx.fillStyle = '#22262c';
  ctx.fillRect(0, H * 0.62, W, H * 0.16);
  ctx.strokeStyle = 'rgba(255,255,255,0.25)';
  ctx.lineWidth = 2;
  ctx.setLineDash([14, 12]);
  ctx.beginPath();
  ctx.moveTo(0, H * 0.7);
  ctx.lineTo(W, H * 0.7);
  ctx.stroke();
  ctx.setLineDash([]);

  // grass foreground band
  const grass = ctx.createLinearGradient(0, H * 0.78, 0, H);
  grass.addColorStop(0, '#26362a');
  grass.addColorStop(1, '#131a14');
  ctx.fillStyle = grass;
  ctx.fillRect(0, H * 0.78, W, H * 0.22);

  // moving car with motion-blur trail (this is the shutter-speed demonstration)
  const carY = H * 0.68;
  const carScale = 1;
  const trailFraction = motionAmount * 0.22; // how far back (as fraction of W) the trail reaches
  const ghosts = 10;
  ctx.save();
  for (let i = ghosts; i >= 0; i--) {
    if (i > 0 && motionAmount < 0.02) continue;
    const t = i / ghosts; // 0 = current position, 1 = oldest ghost
    const gx = (carX - t * trailFraction) * W;
    ctx.globalAlpha = i === 0 ? 1 : Math.max(0, (1 - t) * 0.55 * motionAmount);
    drawCar(ctx, gx, carY, carScale);
  }
  ctx.restore();

  // foreground subject — a flower, always in sharp focus, anchors the eye
  drawFlower(ctx, W * 0.14, H * 0.86, 1.15);
}

function drawCar(ctx, x, y, scale) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(scale, scale);
  ctx.fillStyle = '#e2483d';
  roundRect(ctx, -26, -14, 52, 16, 5);
  ctx.fill();
  ctx.fillStyle = '#c8362c';
  roundRect(ctx, -16, -22, 30, 12, 4);
  ctx.fill();
  ctx.fillStyle = '#12151a';
  ctx.beginPath();
  ctx.arc(-14, 4, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(14, 4, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawFlower(ctx, x, y, scale) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(scale, scale);
  ctx.strokeStyle = '#0d130d';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, -46);
  ctx.stroke();
  ctx.fillStyle = '#1a2a1a';
  ctx.beginPath();
  ctx.ellipse(-10, -20, 10, 4, -0.6, 0, Math.PI * 2);
  ctx.fill();
  const petalColors = ['#f4c04a', '#eaa93a', '#f4c04a', '#eaa93a', '#f4c04a', '#eaa93a'];
  for (let i = 0; i < 6; i++) {
    const angle = (i / 6) * Math.PI * 2;
    ctx.save();
    ctx.translate(0, -58);
    ctx.rotate(angle);
    ctx.fillStyle = petalColors[i];
    ctx.beginPath();
    ctx.ellipse(0, -13, 8, 14, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  ctx.fillStyle = '#5a3c1a';
  ctx.beginPath();
  ctx.arc(0, -58, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Apply overall exposure brightness + sensor-style ISO noise, pixel by pixel.
function applyExposureAndNoise(ctx, brightness, noiseAmount) {
  const imageData = ctx.getImageData(0, 0, W, H);
  const data = imageData.data;
  const noiseStrength = noiseAmount * 55;

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i] * brightness;
    let g = data[i + 1] * brightness;
    let b = data[i + 2] * brightness;

    if (noiseStrength > 0.5) {
      // shadows show noise more than highlights, like a real sensor
      const luminance = (r + g + b) / 3 / 255;
      const shadowBoost = 1.6 - luminance;
      const n = (Math.random() - 0.5) * noiseStrength * shadowBoost;
      r += n;
      g += n * 0.9;
      b += n * 1.1;
    }

    data[i] = clamp255(r);
    data[i + 1] = clamp255(g);
    data[i + 2] = clamp255(b);
  }

  ctx.putImageData(imageData, 0, 0);
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

export default function SceneCanvas({ blurAmount, motionAmount, noiseAmount, brightness }) {
  const canvasRef = useRef(null);
  const carXRef = useRef(0.15);
  const rafRef = useRef(null);
  const paramsRef = useRef({ blurAmount, motionAmount, noiseAmount, brightness });

  paramsRef.current = { blurAmount, motionAmount, noiseAmount, brightness };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    let lastTime = performance.now();

    const tick = (now) => {
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      const { blurAmount, motionAmount, noiseAmount, brightness } = paramsRef.current;

      carXRef.current += dt * 0.12;
      if (carXRef.current > 1.3) carXRef.current = -0.1;

      drawScene(ctx, { blurAmount, motionAmount, carX: carXRef.current });
      applyExposureAndNoise(ctx, brightness, noiseAmount);

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={W}
      height={H}
      className="h-full w-full rounded-xl"
      style={{ imageRendering: 'auto' }}
    />
  );
}
