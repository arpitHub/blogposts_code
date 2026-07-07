import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import WidgetShell, { Stat } from '../../components/WidgetShell.jsx';
import DofSceneCanvas from './DofSceneCanvas.jsx';
import DofDiagram from './DofDiagram.jsx';
import { APERTURE_STOPS } from '../exposure-triangle/exposureMath.js';

export default function ApertureDofModule({ module }) {
  const [apertureIndex, setApertureIndex] = useState(1); // f/2

  const fNumber = APERTURE_STOPS[apertureIndex];
  // wide open = index 0 = maximum blur
  const blurNorm = 1 - apertureIndex / (APERTURE_STOPS.length - 1);
  const bgBlurPx = blurNorm * 15;
  const fgBlurPx = blurNorm * 11;

  // aperture blade opening size for the iris graphic
  const irisOpen = 0.25 + blurNorm * 0.75;

  return (
    <ModulePage
      module={module}
      intro="Depth of field is the slice of the scene that's acceptably sharp. The aperture — an iris inside your lens — sets how thick that slice is."
      explanation={
        <p>
          A wide aperture like <strong className="text-ink">f/1.4</strong> gives a paper-thin
          slice of focus: your subject pops and everything else melts into bokeh. A narrow
          aperture like <strong className="text-ink">f/16</strong> extends the sharp zone from
          near foreground to distant background — which is why landscape photographers stop
          down and portrait photographers open up. Notice the diagram: the sharp zone always
          extends further <em>behind</em> your subject than in front of it.
        </p>
      }
      challenge="Slide to f/1.4 and look at the string of fairy lights in the foreground — they blur too, not just the city behind. Depth of field is a slice, and it cuts both ways. Find the widest aperture where the lights are still recognizable as bulbs."
    >
      <WidgetShell
        preview={<DofSceneCanvas bgBlurPx={bgBlurPx} fgBlurPx={fgBlurPx} />}
        previewFooter={<DofDiagram fNumber={fNumber} />}
        controls={
          <>
            {/* iris graphic mirrors the f-number */}
            <div className="flex items-center justify-center py-2">
              <svg viewBox="0 0 120 120" className="h-28 w-28">
                <circle cx="60" cy="60" r="54" fill="var(--color-panel-2)" stroke="var(--color-line)" />
                {[...Array(8)].map((_, i) => {
                  const angle = (i / 8) * Math.PI * 2;
                  const bladeReach = 54 * (1 - irisOpen);
                  return (
                    <path
                      key={i}
                      d={describeBlade(60, 60, 54, angle, bladeReach)}
                      fill="var(--color-panel-3)"
                      stroke="var(--color-line)"
                      strokeWidth="1"
                    />
                  );
                })}
                <circle cx="60" cy="60" r={54 * irisOpen} fill="rgba(255,176,32,0.12)" stroke="var(--color-accent)" strokeWidth="1.5" />
                <text x="60" y="65" textAnchor="middle" fontSize="16" fill="var(--color-accent)" fontFamily="var(--font-mono)" fontWeight="600">
                  f/{fNumber}
                </text>
              </svg>
            </div>

            <Slider
              label="Aperture"
              color="#ffb020"
              value={apertureIndex}
              min={0}
              max={APERTURE_STOPS.length - 1}
              display={`f/${fNumber}`}
              onChange={setApertureIndex}
            />

            <div className="grid grid-cols-2 gap-2 border-t border-line pt-4">
              <Stat label="Depth of field" value={blurNorm > 0.7 ? 'Razor thin' : blurNorm > 0.4 ? 'Shallow' : blurNorm > 0.15 ? 'Moderate' : 'Deep'} />
              <Stat label="Light gathered" value={blurNorm > 0.7 ? 'Maximum' : blurNorm > 0.4 ? 'Plenty' : blurNorm > 0.15 ? 'Less' : 'Very little'} />
            </div>

            <p className="text-xs leading-relaxed text-ink-3">
              Rule of thumb: portraits f/1.4–f/2.8 · everyday f/4–f/8 · landscapes f/8–f/16.
              Beyond f/16 diffraction starts to soften the whole image.
            </p>
          </>
        }
      />
    </ModulePage>
  );
}

// One aperture blade: a wedge that slides inward as the iris closes.
function describeBlade(cx, cy, R, angle, reach) {
  if (reach <= 0.5) return 'M 0 0';
  const a1 = angle - 0.5;
  const a2 = angle + 0.5;
  const x1 = cx + Math.cos(a1) * R;
  const y1 = cy + Math.sin(a1) * R;
  const x2 = cx + Math.cos(a2) * R;
  const y2 = cy + Math.sin(a2) * R;
  const tipX = cx + Math.cos(angle) * (R - reach);
  const tipY = cy + Math.sin(angle) * (R - reach);
  return `M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x2} ${y2} L ${tipX} ${tipY} Z`;
}
