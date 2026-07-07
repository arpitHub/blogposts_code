import { useEffect, useRef } from 'react';

const W = 480;
const H = 300;

// Three subjects at three depths. Each AF point maps to one of them; picking a
// point racks focus to that depth and blurs the others.
export const SUBJECTS = {
  near: { label: 'Flowers (near)', depth: 0 },
  mid: { label: 'Person (middle)', depth: 1 },
  far: { label: 'Tree (far)', depth: 2 },
};

// 3x3 AF grid: which subject each point covers.
export const AF_GRID = [
  ['far', 'far', 'far'],
  ['mid', 'mid', 'far'],
  ['near', 'mid', 'near'],
];

function blurFor(depth, focusDepth) {
  return Math.abs(depth - focusDepth) * 5.5;
}

export default function FocusCanvas({ focusedSubject, selectedPoint, mode, carX }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d');
    const focusDepth = SUBJECTS[focusedSubject].depth;

    ctx.clearRect(0, 0, W, H);

    // --- far layer: sky + tree ---
    ctx.save();
    const farBlur = blurFor(2, focusDepth);
    ctx.filter = farBlur > 0.3 ? `blur(${farBlur}px)` : 'none';
    const sky = ctx.createLinearGradient(0, 0, 0, H * 0.72);
    sky.addColorStop(0, '#8fb4d8');
    sky.addColorStop(1, '#d8cdb8');
    ctx.fillStyle = sky;
    ctx.fillRect(-15, -15, W + 30, H * 0.75 + 15);

    // distant tree, right side
    ctx.strokeStyle = '#4a3826';
    ctx.lineWidth = 9;
    ctx.beginPath();
    ctx.moveTo(W * 0.8, H * 0.7);
    ctx.lineTo(W * 0.8, H * 0.4);
    ctx.stroke();
    ctx.fillStyle = '#3e5c34';
    for (const [dx, dy, r] of [[0, 0, 34], [-0.06, 0.08, 26], [0.06, 0.08, 27], [0, 0.15, 30]]) {
      ctx.beginPath();
      ctx.arc(W * (0.8 + dx), H * (0.3 + dy), r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // --- ground (kept mid-ish, subtle) ---
    ctx.save();
    ctx.filter = 'none';
    const ground = ctx.createLinearGradient(0, H * 0.72, 0, H);
    ground.addColorStop(0, '#7a8a5a');
    ground.addColorStop(1, '#3c4a2c');
    ctx.fillStyle = ground;
    ctx.fillRect(0, H * 0.72, W, H * 0.28);
    ctx.restore();

    // --- mid layer: person (walks in AF-C demo) ---
    ctx.save();
    const midBlur = blurFor(1, focusDepth);
    ctx.filter = midBlur > 0.3 ? `blur(${midBlur}px)` : 'none';
    const px = W * (carX ?? 0.45);
    const footY = H * 0.8;
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath();
    ctx.ellipse(px, footY + 2, 18, 4, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#345e8a';
    ctx.beginPath();
    ctx.moveTo(px - 13, footY - 62);
    ctx.quadraticCurveTo(px - 16, footY - 24, px - 11, footY);
    ctx.lineTo(px + 11, footY);
    ctx.quadraticCurveTo(px + 16, footY - 24, px + 13, footY - 62);
    ctx.quadraticCurveTo(px, footY - 70, px - 13, footY - 62);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#e8b48a';
    ctx.beginPath();
    ctx.arc(px, footY - 77, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#4a3020';
    ctx.beginPath();
    ctx.arc(px, footY - 81, 10, Math.PI, 0);
    ctx.fill();
    ctx.restore();

    // --- near layer: flowers along the bottom ---
    ctx.save();
    const nearBlur = blurFor(0, focusDepth);
    ctx.filter = nearBlur > 0.3 ? `blur(${nearBlur}px)` : 'none';
    const flowers = [
      [0.06, 0.97], [0.16, 0.99], [0.27, 0.965], [0.38, 1.0], [0.5, 0.975],
      [0.62, 1.0], [0.73, 0.97], [0.85, 0.99], [0.95, 0.965],
    ];
    for (const [fx, fy] of flowers) {
      drawFlower(ctx, fx * W, fy * H);
    }
    ctx.restore();
  }, [focusedSubject, carX]);

  return <canvas ref={canvasRef} width={W} height={H} className="h-full w-full" />;
}

function drawFlower(ctx, x, y) {
  ctx.strokeStyle = '#2c4020';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y - 26);
  ctx.stroke();
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    ctx.fillStyle = '#e86a8a';
    ctx.beginPath();
    ctx.ellipse(x + Math.cos(a) * 7, y - 32 + Math.sin(a) * 7, 5.5, 5.5, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = '#ffd24a';
  ctx.beginPath();
  ctx.arc(x, y - 32, 4.5, 0, Math.PI * 2);
  ctx.fill();
}
