import { useEffect, useRef } from 'react';

const W = 480;
const H = 300;

// The photographer "walks back" as focal length increases so the subject stays
// the same size in frame — this isolates the real lesson: the background does
// NOT stay the same. Long lenses magnify the background relative to the subject
// (compression); wide lenses shrink and push it away.
export default function FocalLengthCanvas({ focalLength }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d');
    draw(ctx, focalLength);
  }, [focalLength]);

  return <canvas ref={canvasRef} width={W} height={H} className="h-full w-full" />;
}

function draw(ctx, fl) {
  // background magnification relative to a 50mm "normal" view
  const bgScale = fl / 50;
  // mountains grow with focal length but stay in frame so the peaks remain readable;
  // the sun keeps growing unbounded — the classic "giant sun" telephoto effect
  const mountainH = Math.min(110 * bgScale, H * 0.52);
  const mountainW = 190 * Math.min(bgScale, 3);
  const sunR = Math.min(26 * bgScale, 150);

  // sky
  const sky = ctx.createLinearGradient(0, 0, 0, H * 0.7);
  sky.addColorStop(0, '#31435e');
  sky.addColorStop(1, '#8a7a80');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, W, H * 0.7);

  const horizonY = H * 0.7;

  // sun — parked left of the subject, riding just above the ridge line
  const sunX = W * 0.3;
  const sunY = Math.max(H * 0.16, horizonY - mountainH * 0.55 - sunR * 0.35);
  const sunGrad = ctx.createRadialGradient(sunX, sunY, 2, sunX, sunY, Math.max(6, sunR));
  sunGrad.addColorStop(0, '#ffe0a8');
  sunGrad.addColorStop(0.7, '#ffcf8a');
  sunGrad.addColorStop(1, 'rgba(255,207,138,0)');
  ctx.fillStyle = sunGrad;
  ctx.beginPath();
  ctx.arc(sunX, sunY, Math.max(6, sunR), 0, Math.PI * 2);
  ctx.fill();

  // mountain range, centered behind the subject
  drawMountain(ctx, W * 0.62, horizonY, mountainW, mountainH, '#3e4a58');
  drawMountain(ctx, W * 0.62 - mountainW * 0.55, horizonY, mountainW * 0.8, mountainH * 0.72, '#333e4a');
  drawMountain(ctx, W * 0.62 + mountainW * 0.5, horizonY, mountainW * 0.7, mountainH * 0.6, '#333e4a');

  // ground
  const ground = ctx.createLinearGradient(0, horizonY, 0, H);
  ground.addColorStop(0, '#4a4a3c');
  ground.addColorStop(1, '#26261e');
  ctx.fillStyle = ground;
  ctx.fillRect(0, horizonY, W, H - horizonY);

  // path converging to the horizon — perspective exaggeration on wide lenses
  const pathSpread = 150 / Math.sqrt(bgScale);
  ctx.fillStyle = '#5a5344';
  ctx.beginPath();
  ctx.moveTo(W * 0.5 - pathSpread, H);
  ctx.lineTo(W * 0.5 - 6 * bgScale, horizonY);
  ctx.lineTo(W * 0.5 + 6 * bgScale, horizonY);
  ctx.lineTo(W * 0.5 + pathSpread, H);
  ctx.closePath();
  ctx.fill();

  // subject: same size at every focal length (photographer moved to keep it framed)
  const px = W * 0.5;
  const footY = H * 0.88;
  ctx.fillStyle = '#20242c';
  ctx.beginPath();
  ctx.ellipse(px, footY + 3, 24, 6, 0, 0, Math.PI * 2);
  ctx.fill();
  // body
  ctx.fillStyle = '#2e6e5e';
  ctx.beginPath();
  ctx.moveTo(px - 16, footY - 78);
  ctx.quadraticCurveTo(px - 20, footY - 30, px - 14, footY);
  ctx.lineTo(px + 14, footY);
  ctx.quadraticCurveTo(px + 20, footY - 30, px + 16, footY - 78);
  ctx.quadraticCurveTo(px, footY - 88, px - 16, footY - 78);
  ctx.closePath();
  ctx.fill();
  // head
  ctx.fillStyle = '#e8b48a';
  ctx.beginPath();
  ctx.arc(px, footY - 96, 13, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#3a2c20';
  ctx.beginPath();
  ctx.arc(px, footY - 101, 12, Math.PI, 0);
  ctx.fill();

  // vignette hint at the widest settings
  if (fl < 24) {
    const v = ctx.createRadialGradient(W / 2, H / 2, H * 0.5, W / 2, H / 2, H * 0.95);
    v.addColorStop(0, 'rgba(0,0,0,0)');
    v.addColorStop(1, `rgba(0,0,0,${(24 - fl) * 0.02})`);
    ctx.fillStyle = v;
    ctx.fillRect(0, 0, W, H);
  }
}

function drawMountain(ctx, cx, baseY, w, h, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(cx - w / 2, baseY);
  ctx.lineTo(cx - w * 0.1, baseY - h);
  ctx.lineTo(cx + w * 0.08, baseY - h * 0.82);
  ctx.lineTo(cx + w / 2, baseY);
  ctx.closePath();
  ctx.fill();
  // snow cap
  ctx.fillStyle = 'rgba(240,244,248,0.85)';
  ctx.beginPath();
  ctx.moveTo(cx - w * 0.16, baseY - h * 0.82);
  ctx.lineTo(cx - w * 0.1, baseY - h);
  ctx.lineTo(cx + w * 0.02, baseY - h * 0.86);
  ctx.lineTo(cx - w * 0.02, baseY - h * 0.78);
  ctx.closePath();
  ctx.fill();
}
