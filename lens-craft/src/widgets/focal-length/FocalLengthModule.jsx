import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import WidgetShell, { Stat } from '../../components/WidgetShell.jsx';
import FocalLengthCanvas from './FocalLengthCanvas.jsx';

// log-scale slider over the classic zoom range
const MIN_FL = 16;
const MAX_FL = 400;

function sliderToFl(v) {
  const t = v / 100;
  return Math.round(MIN_FL * Math.pow(MAX_FL / MIN_FL, t));
}
function flToSlider(fl) {
  return (Math.log(fl / MIN_FL) / Math.log(MAX_FL / MIN_FL)) * 100;
}

function lensClass(fl) {
  if (fl < 24) return { name: 'Ultra-wide', use: 'Architecture, drama, cramped interiors' };
  if (fl < 40) return { name: 'Wide', use: 'Landscapes, environmental portraits' };
  if (fl < 70) return { name: 'Normal', use: 'Street, documentary — sees like your eye' };
  if (fl < 135) return { name: 'Short tele', use: 'Portraits — flattering compression' };
  if (fl < 300) return { name: 'Telephoto', use: 'Sports, events, isolating details' };
  return { name: 'Super-tele', use: 'Wildlife, birds, distant action' };
}

export default function FocalLengthModule({ module }) {
  const [slider, setSlider] = useState(flToSlider(50));
  const fl = sliderToFl(slider);
  const cls = lensClass(fl);

  // full-frame horizontal FOV
  const fovDeg = Math.round((2 * Math.atan(18 / fl) * 180) / Math.PI);

  return (
    <ModulePage
      module={module}
      intro="In this demo the photographer walks backward as the lens gets longer, keeping the person exactly the same size in the frame. Watch what happens to the mountains behind them."
      explanation={
        <p>
          This is <strong className="text-ink">lens compression</strong>: the subject is
          identical in every frame, but at 400mm the mountains tower over them while at 16mm
          they shrink to the horizon. Long lenses appear to "pull the background closer";
          wide lenses push it away and exaggerate depth. That's why portrait photographers
          love 85–135mm (flattering, tidy backgrounds) and landscape photographers carry
          wide glass (depth and scale).
        </p>
      }
      challenge="Slide from 16mm to 400mm slowly while watching only the sun. The person never changes — the world behind them does. Now think about your last phone photo: the main camera is ~24mm equivalent, which is why faces look distorted up close. Portrait mode switches to ~77mm for a reason."
    >
      <WidgetShell
        preview={<FocalLengthCanvas focalLength={fl} />}
        previewFooter={
          <div className="flex items-center justify-between font-mono text-xs">
            <span className="text-accent">{fl}mm</span>
            <span className="text-ink-2">field of view ≈ {fovDeg}°</span>
            <span className="text-ink-3">{cls.name}</span>
          </div>
        }
        controls={
          <>
            <div>
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm font-medium text-ink-2">Focal length</span>
                <span className="rounded-md border border-accent px-2 py-0.5 font-mono text-sm text-accent">
                  {fl}mm
                </span>
              </div>
              <input
                type="range"
                className="dial"
                style={{ '--slider-color': '#ffb020', '--slider-fill': `${slider}%` }}
                min={0}
                max={100}
                step={0.5}
                value={slider}
                onChange={(e) => setSlider(Number(e.target.value))}
              />
              <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-3">
                <span>16mm</span>
                <span>50mm</span>
                <span>400mm</span>
              </div>
            </div>

            {/* FOV wedge */}
            <div className="flex items-center justify-center">
              <svg viewBox="0 0 200 90" className="w-48">
                <circle cx="100" cy="78" r="5" fill="var(--color-ink-2)" />
                <path
                  d={fovWedge(100, 78, 68, fovDeg)}
                  fill="rgba(255,176,32,0.14)"
                  stroke="var(--color-accent)"
                  strokeWidth="1.5"
                />
                <text x="100" y="14" textAnchor="middle" fontSize="10" fill="var(--color-ink-3)" fontFamily="var(--font-mono)">
                  what the lens sees
                </text>
              </svg>
            </div>

            <div className="grid grid-cols-2 gap-2 border-t border-line pt-4">
              <Stat label="Lens class" value={cls.name} />
              <Stat label="Background feels" value={fl < 40 ? 'Far, small' : fl < 100 ? 'Natural' : 'Pulled close'} />
            </div>

            <p className="text-xs leading-relaxed text-ink-3">{cls.use}</p>
          </>
        }
      />
    </ModulePage>
  );
}

function fovWedge(cx, cy, r, deg) {
  const half = (deg / 2) * (Math.PI / 180);
  const x1 = cx - Math.sin(half) * r;
  const y1 = cy - Math.cos(half) * r;
  const x2 = cx + Math.sin(half) * r;
  const y2 = cy - Math.cos(half) * r;
  return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${deg > 180 ? 1 : 0} 1 ${x2} ${y2} Z`;
}
