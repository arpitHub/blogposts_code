import { useEffect, useRef, useCallback } from 'react';

const W = 480;
const H = 300;

function drawScene(ctx, subject, horizonY) {
  // sky above the horizon
  const sky = ctx.createLinearGradient(0, 0, 0, horizonY * H);
  sky.addColorStop(0, '#2b3a55');
  sky.addColorStop(1, '#7a6a70');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, horizonY * H);

  // setting sun hugging the horizon
  const sunGrad = ctx.createRadialGradient(W * 0.72, horizonY * H - 14, 2, W * 0.72, horizonY * H - 14, 30);
  sunGrad.addColorStop(0, '#ffd9a0');
  sunGrad.addColorStop(1, 'rgba(255,217,160,0)');
  ctx.fillStyle = sunGrad;
  ctx.beginPath();
  ctx.arc(W * 0.72, horizonY * H - 14, 30, 0, Math.PI * 2);
  ctx.fill();

  // sea below the horizon
  const sea = ctx.createLinearGradient(0, horizonY * H, 0, H);
  sea.addColorStop(0, '#3a4a5c');
  sea.addColorStop(1, '#1b2530');
  ctx.fillStyle = sea;
  ctx.fillRect(0, horizonY * H, W, H - horizonY * H);

  // gentle wave lines
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1.5;
  for (let i = 1; i <= 5; i++) {
    const y = horizonY * H + (H - horizonY * H) * (i / 6);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(W * 0.3, y - 4, W * 0.7, y + 4, W, y);
    ctx.stroke();
  }

  // sun reflection column
  ctx.fillStyle = 'rgba(255,217,160,0.14)';
  ctx.fillRect(W * 0.69, horizonY * H, W * 0.06, H - horizonY * H);

  // the lighthouse — the draggable subject
  const sx = subject.x * W;
  const groundY = horizonY * H;
  const height = 78;
  const topY = groundY - height;

  ctx.save();
  // rock base
  ctx.fillStyle = '#232a30';
  ctx.beginPath();
  ctx.ellipse(sx, groundY + 4, 26, 8, 0, 0, Math.PI * 2);
  ctx.fill();
  // tower with stripes
  const tw = 16;
  const stripes = 5;
  for (let i = 0; i < stripes; i++) {
    ctx.fillStyle = i % 2 === 0 ? '#d8d3c8' : '#c0392b';
    const y0 = topY + (height * i) / stripes;
    const t0 = 1 - (i / stripes) * 0.25;
    const t1 = 1 - ((i + 1) / stripes) * 0.25;
    ctx.beginPath();
    ctx.moveTo(sx - (tw * t1) / 2, y0 + height / stripes);
    ctx.lineTo(sx - (tw * t0) / 2 - 1.4, y0);
    ctx.lineTo(sx + (tw * t0) / 2 + 1.4, y0);
    ctx.lineTo(sx + (tw * t1) / 2, y0 + height / stripes);
    ctx.closePath();
    ctx.fill();
  }
  // lamp room
  ctx.fillStyle = '#2a2d33';
  ctx.fillRect(sx - 7, topY - 10, 14, 10);
  ctx.fillStyle = '#ffe08a';
  ctx.fillRect(sx - 5, topY - 8, 10, 6);
  ctx.beginPath();
  ctx.moveTo(sx - 8, topY - 10);
  ctx.lineTo(sx, topY - 18);
  ctx.lineTo(sx + 8, topY - 10);
  ctx.closePath();
  ctx.fillStyle = '#2a2d33';
  ctx.fill();
  ctx.restore();
}

function drawGrid(ctx, grid) {
  if (grid === 'off') return;
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.lineWidth = 1;

  if (grid === 'thirds') {
    for (const f of [1 / 3, 2 / 3]) {
      ctx.beginPath();
      ctx.moveTo(f * W, 0);
      ctx.lineTo(f * W, H);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, f * H);
      ctx.lineTo(W, f * H);
      ctx.stroke();
    }
    // power points
    ctx.fillStyle = 'rgba(255,176,32,0.85)';
    for (const fx of [1 / 3, 2 / 3]) {
      for (const fy of [1 / 3, 2 / 3]) {
        ctx.beginPath();
        ctx.arc(fx * W, fy * H, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  } else if (grid === 'center') {
    ctx.beginPath();
    ctx.moveTo(W / 2, 0);
    ctx.lineTo(W / 2, H);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, H / 2);
    ctx.lineTo(W, H / 2);
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,176,32,0.85)';
    ctx.beginPath();
    ctx.arc(W / 2, H / 2, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export default function CompositionCanvas({ subject, horizonY, grid, onDrag }) {
  const canvasRef = useRef(null);
  const dragRef = useRef(null); // 'subject' | 'horizon' | null

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    drawScene(ctx, subject, horizonY);
    drawGrid(ctx, grid);
  }, [subject, horizonY, grid]);

  const toLocal = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return { x: cx / rect.width, y: cy / rect.height };
  }, []);

  const handleDown = useCallback(
    (e) => {
      const p = toLocal(e);
      // near the lighthouse?
      if (Math.abs(p.x - subject.x) < 0.09 && p.y > horizonY - 0.35 && p.y < horizonY + 0.08) {
        dragRef.current = 'subject';
      } else if (Math.abs(p.y - horizonY) < 0.08) {
        dragRef.current = 'horizon';
      }
    },
    [subject.x, horizonY, toLocal]
  );

  const handleMove = useCallback(
    (e) => {
      if (!dragRef.current) return;
      e.preventDefault();
      const p = toLocal(e);
      if (dragRef.current === 'subject') {
        onDrag({ subjectX: Math.min(0.94, Math.max(0.06, p.x)) });
      } else {
        onDrag({ horizonY: Math.min(0.85, Math.max(0.15, p.y)) });
      }
    },
    [onDrag, toLocal]
  );

  const handleUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={W}
      height={H}
      className="h-full w-full cursor-grab touch-none active:cursor-grabbing"
      onMouseDown={handleDown}
      onMouseMove={handleMove}
      onMouseUp={handleUp}
      onMouseLeave={handleUp}
      onTouchStart={handleDown}
      onTouchMove={handleMove}
      onTouchEnd={handleUp}
    />
  );
}
