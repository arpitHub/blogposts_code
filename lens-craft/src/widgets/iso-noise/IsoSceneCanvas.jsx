import { useEffect, useRef } from 'react';
import { applyExposureAndNoise } from '../../lib/canvasUtils.js';

const W = 480;
const H = 300;

function drawStreet(ctx, night) {
  // sky
  const sky = ctx.createLinearGradient(0, 0, 0, H * 0.6);
  if (night) {
    sky.addColorStop(0, '#0a0f1e');
    sky.addColorStop(1, '#1a2238');
  } else {
    sky.addColorStop(0, '#7db3e8');
    sky.addColorStop(1, '#b8d4ee');
  }
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H * 0.6);

  if (night) {
    // stars
    for (let i = 0; i < 40; i++) {
      const x = ((i * 137.5) % 480);
      const y = ((i * 89.3) % 140);
      ctx.fillStyle = `rgba(255,255,255,${0.3 + (i % 5) * 0.12})`;
      ctx.fillRect(x, y, 1.5, 1.5);
    }
    // moon
    ctx.fillStyle = '#e8e6da';
    ctx.beginPath();
    ctx.arc(W * 0.82, H * 0.14, 18, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#0a0f1e';
    ctx.beginPath();
    ctx.arc(W * 0.85, H * 0.125, 15, 0, Math.PI * 2);
    ctx.fill();
  } else {
    // sun
    const sun = ctx.createRadialGradient(W * 0.8, H * 0.15, 4, W * 0.8, H * 0.15, 42);
    sun.addColorStop(0, 'rgba(255,244,200,1)');
    sun.addColorStop(1, 'rgba(255,244,200,0)');
    ctx.fillStyle = sun;
    ctx.fillRect(W * 0.66, 0, 160, 120);
  }

  // row of houses
  const houses = [
    [0.02, '#8a5a44'], [0.2, '#5a6e8a'], [0.38, '#7a6a4a'], [0.56, '#6a4a5a'], [0.74, '#4a6a5a'], [0.9, '#71584a'],
  ];
  for (const [hx, color] of houses) {
    const x = hx * W;
    const hw = 0.17 * W;
    ctx.fillStyle = night ? shade(color, 0.35) : color;
    ctx.fillRect(x, H * 0.38, hw, H * 0.24);
    ctx.fillStyle = night ? '#1a1610' : '#3a2e24';
    ctx.beginPath();
    ctx.moveTo(x - 6, H * 0.38);
    ctx.lineTo(x + hw / 2, H * 0.27);
    ctx.lineTo(x + hw + 6, H * 0.38);
    ctx.closePath();
    ctx.fill();
    // windows glow at night
    ctx.fillStyle = night ? '#ffcf6a' : '#dfeefc';
    ctx.fillRect(x + hw * 0.15, H * 0.44, hw * 0.22, H * 0.08);
    ctx.fillRect(x + hw * 0.6, H * 0.44, hw * 0.22, H * 0.08);
  }

  // street
  ctx.fillStyle = night ? '#15161a' : '#3f434a';
  ctx.fillRect(0, H * 0.62, W, H * 0.38);
  ctx.strokeStyle = night ? 'rgba(255,255,255,0.16)' : 'rgba(255,255,255,0.45)';
  ctx.lineWidth = 3;
  ctx.setLineDash([18, 16]);
  ctx.beginPath();
  ctx.moveTo(0, H * 0.8);
  ctx.lineTo(W, H * 0.8);
  ctx.stroke();
  ctx.setLineDash([]);

  // streetlamp
  ctx.strokeStyle = '#2a2d33';
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(W * 0.14, H * 0.62);
  ctx.lineTo(W * 0.14, H * 0.3);
  ctx.quadraticCurveTo(W * 0.14, H * 0.24, W * 0.2, H * 0.24);
  ctx.stroke();
  if (night) {
    const lamp = ctx.createRadialGradient(W * 0.2, H * 0.26, 2, W * 0.2, H * 0.26, 55);
    lamp.addColorStop(0, 'rgba(255,214,130,0.95)');
    lamp.addColorStop(1, 'rgba(255,214,130,0)');
    ctx.fillStyle = lamp;
    ctx.fillRect(W * 0.08, H * 0.12, 120, 130);
  }
  ctx.fillStyle = night ? '#ffe3a0' : '#c8ccd4';
  ctx.beginPath();
  ctx.arc(W * 0.2, H * 0.26, 6, 0, Math.PI * 2);
  ctx.fill();
}

function shade(hex, factor) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.round(((n >> 16) & 255) * factor);
  const g = Math.round(((n >> 8) & 255) * factor);
  const b = Math.round((n & 255) * factor);
  return `rgb(${r},${g},${b})`;
}

export default function IsoSceneCanvas({ night, noiseAmount }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    let raf;
    const render = () => {
      drawStreet(ctx, night);
      applyExposureAndNoise(ctx, W, H, 1, noiseAmount);
      // keep re-rendering so the grain "crawls" like a live sensor feed
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [night, noiseAmount]);

  return <canvas ref={canvasRef} width={W} height={H} className="h-full w-full" />;
}
