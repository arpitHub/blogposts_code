import { useEffect, useRef } from 'react';

const W = 480;
const H = 300;

// City lights behind the subject — fixed layout so the frame is stable.
const CITY_LIGHTS = [
  { x: 0.06, y: 0.3, r: 4, c: '#ffd27a' }, { x: 0.12, y: 0.22, r: 3, c: '#8ecbff' },
  { x: 0.2, y: 0.34, r: 5, c: '#ff9d6c' }, { x: 0.27, y: 0.18, r: 3, c: '#ffe9b0' },
  { x: 0.33, y: 0.28, r: 4, c: '#ffd27a' }, { x: 0.45, y: 0.2, r: 3, c: '#8ecbff' },
  { x: 0.55, y: 0.32, r: 5, c: '#ff9d6c' }, { x: 0.62, y: 0.16, r: 3, c: '#ffd27a' },
  { x: 0.7, y: 0.26, r: 4, c: '#ffe9b0' }, { x: 0.78, y: 0.34, r: 5, c: '#8ecbff' },
  { x: 0.86, y: 0.2, r: 3, c: '#ff9d6c' }, { x: 0.93, y: 0.3, r: 4, c: '#ffd27a' },
  { x: 0.4, y: 0.12, r: 2.5, c: '#8ecbff' }, { x: 0.85, y: 0.1, r: 2.5, c: '#ffe9b0' },
];

function drawBackground(ctx, blurPx) {
  ctx.save();
  ctx.filter = blurPx > 0.3 ? `blur(${blurPx}px)` : 'none';

  const sky = ctx.createLinearGradient(0, 0, 0, H * 0.75);
  sky.addColorStop(0, '#131c30');
  sky.addColorStop(1, '#2c3a55');
  ctx.fillStyle = sky;
  ctx.fillRect(-20, -20, W + 40, H * 0.78 + 20);

  // skyline blocks
  ctx.fillStyle = '#0e1422';
  const buildings = [
    [0, 0.42, 0.1, 0.33], [0.1, 0.36, 0.08, 0.39], [0.19, 0.46, 0.12, 0.29],
    [0.32, 0.32, 0.09, 0.43], [0.42, 0.44, 0.1, 0.31], [0.53, 0.38, 0.08, 0.37],
    [0.62, 0.48, 0.11, 0.27], [0.74, 0.34, 0.09, 0.41], [0.84, 0.44, 0.09, 0.31],
    [0.93, 0.4, 0.07, 0.35],
  ];
  for (const [x, y, w, h] of buildings) {
    ctx.fillRect(x * W, y * H, w * W + 1, h * H);
  }

  // city bokeh lights — the size grows as blur grows, like real bokeh balls
  for (const l of CITY_LIGHTS) {
    const r = l.r * (1 + blurPx * 0.22);
    const grad = ctx.createRadialGradient(l.x * W, l.y * H, 0, l.x * W, l.y * H, r);
    grad.addColorStop(0, l.c);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(l.x * W, l.y * H, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawSubject(ctx) {
  // person, mid-distance, always sharp — this is the focal plane
  const x = W * 0.5;
  const baseY = H * 0.88;
  ctx.save();
  ctx.translate(x, baseY);

  ctx.fillStyle = '#e8b48a';
  ctx.beginPath();
  ctx.arc(0, -118, 16, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#2e2620';
  ctx.beginPath();
  ctx.arc(0, -124, 15, Math.PI, 0);
  ctx.fill();
  // coat
  ctx.fillStyle = '#b8452f';
  ctx.beginPath();
  ctx.moveTo(-20, -100);
  ctx.quadraticCurveTo(-24, -40, -18, 0);
  ctx.lineTo(18, 0);
  ctx.quadraticCurveTo(24, -40, 20, -100);
  ctx.quadraticCurveTo(0, -112, -20, -100);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = '#8f3320';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, -96);
  ctx.lineTo(0, -8);
  ctx.stroke();
  ctx.restore();

  // ground
  const ground = ctx.createLinearGradient(0, H * 0.78, 0, H);
  ground.addColorStop(0, '#20242b');
  ground.addColorStop(1, '#101216');
  ctx.fillStyle = ground;
  ctx.fillRect(0, H * 0.88, W, H * 0.12);
  // re-draw feet zone shadow
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.beginPath();
  ctx.ellipse(W * 0.5, H * 0.885, 26, 5, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawForeground(ctx, blurPx) {
  ctx.save();
  ctx.filter = blurPx > 0.3 ? `blur(${blurPx}px)` : 'none';
  // fairy lights strung across the near foreground, classic "foreground bokeh"
  ctx.strokeStyle = 'rgba(40,50,40,0.9)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(-10, H * 0.9);
  ctx.quadraticCurveTo(W * 0.3, H * 1.02, W * 0.65, H * 0.86);
  ctx.quadraticCurveTo(W * 0.85, H * 0.78, W + 10, H * 0.84);
  ctx.stroke();

  const bulbs = [
    [0.04, 0.92], [0.14, 0.95], [0.25, 0.97], [0.37, 0.955], [0.48, 0.92],
    [0.58, 0.885], [0.68, 0.845], [0.78, 0.815], [0.88, 0.815], [0.97, 0.835],
  ];
  for (const [bx, by] of bulbs) {
    const r = 5 * (1 + blurPx * 0.28);
    const grad = ctx.createRadialGradient(bx * W, by * H, 0, bx * W, by * H, r);
    grad.addColorStop(0, '#ffe08a');
    grad.addColorStop(1, 'rgba(255,224,138,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(bx * W, by * H, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export default function DofSceneCanvas({ bgBlurPx, fgBlurPx }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    drawBackground(ctx, bgBlurPx);
    drawSubject(ctx);
    drawForeground(ctx, fgBlurPx);
  }, [bgBlurPx, fgBlurPx]);

  return <canvas ref={canvasRef} width={W} height={H} className="h-full w-full" />;
}
