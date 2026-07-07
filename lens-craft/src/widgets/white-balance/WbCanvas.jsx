import { useEffect, useRef } from 'react';
import { kelvinToRGB, clamp255 } from '../../lib/canvasUtils.js';

const W = 480;
const H = 300;

// A cosy interior scene with plenty of neutral tones so colour casts are obvious.
function drawRoom(ctx) {
  // wall
  ctx.fillStyle = '#b9b2a6';
  ctx.fillRect(0, 0, W, H * 0.72);
  // floor
  ctx.fillStyle = '#7a6248';
  ctx.fillRect(0, H * 0.72, W, H * 0.28);
  for (let i = 0; i < 7; i++) {
    ctx.strokeStyle = 'rgba(0,0,0,0.18)';
    ctx.beginPath();
    ctx.moveTo(0, H * (0.72 + 0.04 * i));
    ctx.lineTo(W, H * (0.72 + 0.045 * i));
    ctx.stroke();
  }

  // window with daylight
  ctx.fillStyle = '#e8f0f8';
  ctx.fillRect(W * 0.62, H * 0.1, W * 0.28, H * 0.42);
  ctx.strokeStyle = '#5a5248';
  ctx.lineWidth = 6;
  ctx.strokeRect(W * 0.62, H * 0.1, W * 0.28, H * 0.42);
  ctx.beginPath();
  ctx.moveTo(W * 0.76, H * 0.1);
  ctx.lineTo(W * 0.76, H * 0.52);
  ctx.moveTo(W * 0.62, H * 0.31);
  ctx.lineTo(W * 0.9, H * 0.31);
  ctx.stroke();

  // white mug on a table — the neutral reference
  ctx.fillStyle = '#6a5138';
  ctx.fillRect(W * 0.08, H * 0.58, W * 0.34, H * 0.05);
  ctx.fillRect(W * 0.12, H * 0.63, W * 0.04, H * 0.16);
  ctx.fillRect(W * 0.34, H * 0.63, W * 0.04, H * 0.16);

  ctx.fillStyle = '#f4f4f2';
  ctx.beginPath();
  ctx.roundRect(W * 0.2, H * 0.47, W * 0.09, H * 0.11, 4);
  ctx.fill();
  ctx.strokeStyle = '#f4f4f2';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(W * 0.3, H * 0.525, 9, -Math.PI / 2, Math.PI / 2);
  ctx.stroke();

  // grey cat sleeping on the floor
  ctx.fillStyle = '#8d8d90';
  ctx.beginPath();
  ctx.ellipse(W * 0.52, H * 0.84, 44, 20, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(W * 0.585, H * 0.79, 15, 0, Math.PI * 2);
  ctx.fill();
  // ears
  ctx.beginPath();
  ctx.moveTo(W * 0.575, H * 0.745);
  ctx.lineTo(W * 0.583, H * 0.72);
  ctx.lineTo(W * 0.595, H * 0.745);
  ctx.moveTo(W * 0.6, H * 0.745);
  ctx.lineTo(W * 0.61, H * 0.722);
  ctx.lineTo(W * 0.617, H * 0.75);
  ctx.fill();
  ctx.strokeStyle = '#6e6e72';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(W * 0.47, H * 0.85, 18, Math.PI * 0.2, Math.PI * 1.1);
  ctx.stroke();

  // framed picture
  ctx.fillStyle = '#4a4440';
  ctx.fillRect(W * 0.13, H * 0.14, W * 0.18, H * 0.22);
  ctx.fillStyle = '#d8d2c6';
  ctx.fillRect(W * 0.145, H * 0.165, W * 0.15, H * 0.17);
  ctx.fillStyle = '#7a8a6a';
  ctx.beginPath();
  ctx.moveTo(W * 0.145, H * 0.335);
  ctx.lineTo(W * 0.19, H * 0.24);
  ctx.lineTo(W * 0.23, H * 0.3);
  ctx.lineTo(W * 0.26, H * 0.26);
  ctx.lineTo(W * 0.295, H * 0.335);
  ctx.closePath();
  ctx.fill();
}

export default function WbCanvas({ kelvin }) {
  const canvasRef = useRef(null);
  const baseRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    if (!baseRef.current) {
      drawRoom(ctx);
      baseRef.current = ctx.getImageData(0, 0, W, H);
    }

    // Tint = ratio between the light's colour and neutral daylight (6500K).
    const [lr, lg, lb] = kelvinToRGB(kelvin);
    const [nr, ng, nb] = kelvinToRGB(6500);
    const fr = lr / nr;
    const fg = lg / ng;
    const fb = lb / nb;

    const out = ctx.createImageData(W, H);
    const src = baseRef.current.data;
    const dst = out.data;
    for (let i = 0; i < src.length; i += 4) {
      dst[i] = clamp255(src[i] * fr);
      dst[i + 1] = clamp255(src[i + 1] * fg);
      dst[i + 2] = clamp255(src[i + 2] * fb);
      dst[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }, [kelvin]);

  return <canvas ref={canvasRef} width={W} height={H} className="h-full w-full" />;
}
