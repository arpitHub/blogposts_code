import { useEffect, useRef } from 'react';

const W = 480;
const H = 300;

// --- Waterfall scene -------------------------------------------------------
// Water particles fall continuously; each is rendered as a streak whose length
// equals distance travelled during the exposure — the real physics of motion blur.

function makeDrops() {
  const drops = [];
  for (let i = 0; i < 130; i++) {
    drops.push({
      x: 0.38 + Math.random() * 0.24,
      y: Math.random(),
      speed: 0.55 + Math.random() * 0.35, // fraction of fall height per second
      drift: (Math.random() - 0.5) * 0.015,
    });
  }
  return drops;
}

function drawWaterfall(ctx, drops, dt, exposureSeconds) {
  // backdrop: cliff and pool
  const sky = ctx.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, '#22303f');
  sky.addColorStop(1, '#101820');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = '#2c3a30';
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(W * 0.36, 0);
  ctx.lineTo(W * 0.33, H * 0.2);
  ctx.lineTo(W * 0.38, H * 0.75);
  ctx.lineTo(0, H * 0.8);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#26332a';
  ctx.beginPath();
  ctx.moveTo(W, 0);
  ctx.lineTo(W * 0.64, 0);
  ctx.lineTo(W * 0.67, H * 0.2);
  ctx.lineTo(W * 0.62, H * 0.75);
  ctx.lineTo(W, H * 0.8);
  ctx.closePath();
  ctx.fill();

  // pool
  ctx.fillStyle = '#1b2c38';
  ctx.fillRect(0, H * 0.78, W, H * 0.22);

  const fallTop = H * 0.02;
  const fallBottom = H * 0.8;
  const fallHeight = fallBottom - fallTop;

  for (const d of drops) {
    d.y += d.speed * dt;
    d.x += d.drift * dt;
    if (d.y > 1) {
      d.y -= 1;
      d.x = 0.38 + Math.random() * 0.24;
    }

    const px = d.x * W;
    const py = fallTop + d.y * fallHeight;
    // distance travelled during the exposure, in px
    const streak = Math.min(fallHeight * 0.9, d.speed * exposureSeconds * fallHeight);

    if (streak < 2.5) {
      ctx.fillStyle = 'rgba(210,235,255,0.9)';
      ctx.beginPath();
      ctx.arc(px, py, 1.6, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // long exposure: translucent streak — many drops overlap into "silk"
      const alpha = Math.max(0.06, 0.5 - streak / 260);
      ctx.strokeStyle = `rgba(210,235,255,${alpha})`;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(px, Math.max(fallTop, py - streak));
      ctx.lineTo(px - d.drift * 60, py);
      ctx.stroke();
    }
  }

  // pool splash zone gets misty with long exposures
  const mist = Math.min(0.5, exposureSeconds * 0.6);
  if (mist > 0.03) {
    const g = ctx.createRadialGradient(W * 0.5, H * 0.8, 5, W * 0.5, H * 0.8, 90);
    g.addColorStop(0, `rgba(220,240,255,${mist})`);
    g.addColorStop(1, 'rgba(220,240,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(W * 0.3, H * 0.62, W * 0.4, H * 0.36);
  }
}

// --- Pinwheel scene --------------------------------------------------------
// The wheel is drawn repeatedly across the arc it sweeps during the exposure.

const SPOKE_COLORS = ['#e2483d', '#ffb020', '#52c97a', '#3987e5', '#b06ce8', '#ff8a5c'];

function drawPinwheel(ctx, rotation, exposureSeconds, angularSpeed) {
  const sky = ctx.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, '#28303d');
  sky.addColorStop(1, '#151a22');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H * 0.46;
  const R = H * 0.34;

  // stick
  ctx.strokeStyle = '#4a3b2c';
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx, H);
  ctx.stroke();

  const sweep = angularSpeed * exposureSeconds; // radians swept during exposure
  const ghosts = Math.max(1, Math.min(40, Math.ceil(sweep / 0.05)));

  for (let g = ghosts - 1; g >= 0; g--) {
    const a = rotation - (g / ghosts) * sweep;
    const alpha = ghosts === 1 ? 1 : Math.min(1, 1.6 / ghosts + (g === 0 ? 0.25 : 0));
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.translate(cx, cy);
    ctx.rotate(a);
    for (let i = 0; i < 6; i++) {
      ctx.rotate(Math.PI / 3);
      ctx.fillStyle = SPOKE_COLORS[i];
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.quadraticCurveTo(R * 0.45, -R * 0.28, R, -R * 0.12);
      ctx.quadraticCurveTo(R * 0.6, R * 0.05, 0, 0);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  ctx.fillStyle = '#f3f2ee';
  ctx.beginPath();
  ctx.arc(cx, cy, 7, 0, Math.PI * 2);
  ctx.fill();
}

export default function MotionSceneCanvas({ scene, exposureSeconds }) {
  const canvasRef = useRef(null);
  const dropsRef = useRef(null);
  const rotationRef = useRef(0);
  const paramsRef = useRef({ scene, exposureSeconds });
  paramsRef.current = { scene, exposureSeconds };

  useEffect(() => {
    if (!dropsRef.current) dropsRef.current = makeDrops();
    const ctx = canvasRef.current.getContext('2d');
    let last = performance.now();
    let raf;

    const tick = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const { scene, exposureSeconds } = paramsRef.current;
      const angularSpeed = Math.PI * 1.2; // rad/s

      if (scene === 'waterfall') {
        drawWaterfall(ctx, dropsRef.current, dt, exposureSeconds);
      } else {
        rotationRef.current += angularSpeed * dt;
        drawPinwheel(ctx, rotationRef.current, exposureSeconds, angularSpeed);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={canvasRef} width={W} height={H} className="h-full w-full" />;
}
