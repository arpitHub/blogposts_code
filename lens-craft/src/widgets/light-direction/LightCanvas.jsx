import { useEffect, useRef, useCallback } from 'react';

const W = 480;
const H = 300;
const CX = W / 2;
const CY = H * 0.52;
const FACE_R = 62;
const ORBIT_R = 118;

// angle: 0 = light directly in front (photographer's position, bottom of canvas),
// PI/2 = camera-left, PI = behind the subject, 3PI/2 = camera-right.
export default function LightCanvas({ angle, softness, onAngleChange }) {
  const canvasRef = useRef(null);
  const draggingRef = useRef(false);

  useEffect(() => {
    const ctx = canvasRef.current.getContext('2d');
    draw(ctx, angle, softness);
  }, [angle, softness]);

  const handleDown = useCallback((e) => {
    draggingRef.current = true;
    onAngleChange(pointerAngle(e, canvasRef.current));
  }, [onAngleChange]);

  const handleMove = useCallback((e) => {
    if (!draggingRef.current) return;
    e.preventDefault();
    onAngleChange(pointerAngle(e, canvasRef.current));
  }, [onAngleChange]);

  const handleUp = useCallback(() => {
    draggingRef.current = false;
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

// Angle of the pointer around the subject centre, 0 = straight down (front).
function pointerAngle(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const px = (((e.touches ? e.touches[0].clientX : e.clientX) - rect.left) / rect.width) * W;
  const py = (((e.touches ? e.touches[0].clientY : e.clientY) - rect.top) / rect.height) * H;
  return Math.atan2(px - CX, py - CY);
}

function draw(ctx, angle, softness) {
  // The light orbits in the horizontal plane; we render a top-lit studio look.
  // lightX/lightZ: unit vector in the plane, z>0 means in front of subject.
  const lightX = Math.sin(angle);
  const lightZ = Math.cos(angle);
  const behind = lightZ < -0.2;

  // backdrop
  const bg = ctx.createRadialGradient(CX, CY, 40, CX, CY, 320);
  bg.addColorStop(0, behind ? '#2e3442' : '#262b35');
  bg.addColorStop(1, '#14161c');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // rim glow behind subject when backlit
  if (behind) {
    const rim = ctx.createRadialGradient(CX, CY, FACE_R * 0.6, CX, CY, FACE_R * 2.2);
    rim.addColorStop(0, 'rgba(255,236,200,0.55)');
    rim.addColorStop(1, 'rgba(255,236,200,0)');
    ctx.fillStyle = rim;
    ctx.beginPath();
    ctx.arc(CX, CY, FACE_R * 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // cast shadow on the floor, opposite the light
  const shadowLen = 40 + (1 - Math.abs(lightZ)) * 30;
  ctx.save();
  ctx.translate(CX - lightX * shadowLen, CY + FACE_R + 26);
  ctx.scale(1, 0.3);
  const shBlur = 6 + softness * 22;
  ctx.filter = `blur(${shBlur}px)`;
  ctx.fillStyle = `rgba(0,0,0,${behind ? 0.5 : 0.55 - softness * 0.2})`;
  ctx.beginPath();
  ctx.arc(0, 0, FACE_R * 0.85, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // ---- the head ----
  // base skin, darker when backlit
  const baseTone = behind ? '#6b5648' : '#a9836b';
  ctx.fillStyle = baseTone;
  ctx.beginPath();
  ctx.ellipse(CX, CY, FACE_R * 0.82, FACE_R, 0, 0, Math.PI * 2);
  ctx.fill();

  if (!behind) {
    // lit-side gradient: highlight centred where the light hits
    const hx = CX + lightX * FACE_R * 0.55;
    const litR = FACE_R * (0.9 + softness * 0.9);
    const lit = ctx.createRadialGradient(hx, CY - FACE_R * 0.15, 4, hx, CY - FACE_R * 0.15, litR);
    const litStrength = 0.55 + lightZ * 0.25;
    lit.addColorStop(0, `rgba(255,226,190,${litStrength})`);
    lit.addColorStop(softness * 0.5 + 0.35, `rgba(255,226,190,${litStrength * 0.35})`);
    lit.addColorStop(1, 'rgba(255,226,190,0)');
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(CX, CY, FACE_R * 0.82, FACE_R, 0, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = lit;
    ctx.fillRect(CX - FACE_R * 1.2, CY - FACE_R * 1.2, FACE_R * 2.4, FACE_R * 2.4);

    // shadow side
    const shx = CX - lightX * FACE_R * 0.7;
    const shadowGrad = ctx.createRadialGradient(shx, CY, FACE_R * (0.2 + softness * 0.35), shx, CY, FACE_R * 1.5);
    const shadowDepth = 0.5 - softness * 0.3;
    shadowGrad.addColorStop(0, `rgba(30,22,18,${shadowDepth})`);
    shadowGrad.addColorStop(1, 'rgba(30,22,18,0)');
    ctx.fillStyle = shadowGrad;
    ctx.fillRect(CX - FACE_R * 1.2, CY - FACE_R * 1.2, FACE_R * 2.4, FACE_R * 2.4);
    ctx.restore();
  } else {
    // backlit: bright rim along the edges
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(CX, CY, FACE_R * 0.82, FACE_R, 0, 0, Math.PI * 2);
    ctx.clip();
    ctx.strokeStyle = `rgba(255,240,210,${0.85 - softness * 0.25})`;
    ctx.lineWidth = 5 + softness * 6;
    ctx.filter = `blur(${2 + softness * 5}px)`;
    ctx.beginPath();
    ctx.ellipse(CX, CY, FACE_R * 0.8, FACE_R * 0.97, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  // simple features, kept dim so shading reads first
  ctx.fillStyle = 'rgba(35,25,20,0.65)';
  ctx.beginPath();
  ctx.ellipse(CX - FACE_R * 0.3, CY - FACE_R * 0.12, 5, 7, 0, 0, Math.PI * 2);
  ctx.ellipse(CX + FACE_R * 0.3, CY - FACE_R * 0.12, 5, 7, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(35,25,20,0.55)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(CX, CY + FACE_R * 0.42, FACE_R * 0.3, 0.25 * Math.PI, 0.75 * Math.PI);
  ctx.stroke();
  // nose hint
  ctx.beginPath();
  ctx.moveTo(CX, CY - 4);
  ctx.lineTo(CX - 4, CY + FACE_R * 0.16);
  ctx.stroke();

  // ---- orbit ring & draggable light ----
  ctx.strokeStyle = 'rgba(255,255,255,0.14)';
  ctx.setLineDash([4, 6]);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(CX, CY, ORBIT_R, ORBIT_R * 0.92, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  const lx = CX + Math.sin(angle) * ORBIT_R;
  const ly = CY + Math.cos(angle) * ORBIT_R * 0.92;
  const lampGrad = ctx.createRadialGradient(lx, ly, 2, lx, ly, 26);
  lampGrad.addColorStop(0, 'rgba(255,236,180,1)');
  lampGrad.addColorStop(1, 'rgba(255,236,180,0)');
  ctx.fillStyle = lampGrad;
  ctx.beginPath();
  ctx.arc(lx, ly, 26, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#fff3cf';
  ctx.beginPath();
  ctx.arc(lx, ly, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(lx, ly, 13, 0, Math.PI * 2);
  ctx.stroke();

  // camera marker at the bottom (the viewer's position)
  ctx.fillStyle = 'var(--color-ink-3)';
  ctx.fillStyle = '#8a8894';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('📷 you are here', CX, H - 8);
}
