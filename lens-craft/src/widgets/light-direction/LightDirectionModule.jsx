import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import WidgetShell, { Stat } from '../../components/WidgetShell.jsx';
import LightCanvas from './LightCanvas.jsx';

function describeDirection(angle) {
  // normalize to [0, 2PI); 0 = front (at the camera), PI = directly behind
  const a = ((angle % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
  const deg = (a * 180) / Math.PI;
  if (deg < 25 || deg > 335) return { name: 'Front light', note: 'Flat and even — safe, but can look like a passport photo.' };
  if (deg < 70 || deg > 290) return { name: '45° light', note: 'The classic portrait angle — shape without harshness. "Rembrandt" territory.' };
  if (deg < 115 || deg > 245) return { name: 'Side light', note: 'Maximum drama: one half lit, one half in shadow. Texture jumps out.' };
  if (deg < 155 || deg > 205) return { name: 'Rim / back-side light', note: 'The subject edge glows, separating them from the background.' };
  return { name: 'Backlight', note: 'Silhouette or halo — expose for the mood you want, not the meter.' };
}

export default function LightDirectionModule({ module }) {
  const [angle, setAngle] = useState(Math.PI / 4); // 45°, classic
  const [softness, setSoftness] = useState(0.35);

  const dir = describeDirection(angle);
  const deg = Math.round((((angle % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2)) * 180 / Math.PI);

  return (
    <ModulePage
      module={module}
      intro="Photography is drawing with light, and the two questions that matter most are: where is it coming from, and how big is the source? Drag the lamp around the subject and see for yourself."
      explanation={
        <p>
          <strong className="text-ink">Direction</strong> creates shape: front light flattens,
          side light sculpts, backlight silhouettes.{' '}
          <strong className="text-ink">Softness</strong> comes from the size of the light source
          relative to the subject — a bare bulb or the midday sun is small and hard; a big
          window, a cloudy sky, or a softbox is large and soft. Hard light makes crisp, deep
          shadows; soft light wraps around and fades them.
        </p>
      }
      challenge="Drag the light directly behind the subject (top of the ring) and watch the rim of light appear around the head. Then bring it to 90° with softness near zero — this is the classic 'split light' used in dramatic portraits. Which mood suits a villain, and which a wedding?"
    >
      <WidgetShell
        preview={<LightCanvas angle={angle} softness={softness} onAngleChange={setAngle} />}
        previewFooter={
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium" style={{ color: 'var(--color-accent)' }}>
              {dir.name} · {deg}°
            </span>
            <span className="max-w-[55%] text-right text-xs leading-snug text-ink-2">{dir.note}</span>
          </div>
        }
        controls={
          <>
            <div className="rounded-lg bg-panel-2 px-3 py-2.5 text-xs leading-relaxed text-ink-2">
              <span className="mb-1 block font-mono text-[10px] uppercase tracking-wide text-ink-3">
                How to use
              </span>
              Drag the <strong className="text-ink">glowing lamp</strong> around the ring — it
              orbits the subject while your camera stays put at the bottom.
            </div>

            <Slider
              label="Light softness"
              color="#ffb020"
              value={Math.round(softness * 100)}
              min={0}
              max={100}
              display={softness < 0.25 ? 'Hard (bare bulb)' : softness < 0.6 ? 'Medium' : 'Soft (softbox)'}
              onChange={(v) => setSoftness(v / 100)}
            />

            <div className="grid grid-cols-2 gap-2 border-t border-line pt-4">
              <Stat label="Shadow edge" value={softness < 0.25 ? 'Crisp' : softness < 0.6 ? 'Feathered' : 'Wrapped'} />
              <Stat label="Mood" value={deg > 115 && deg < 245 ? 'Ethereal' : softness < 0.3 ? 'Dramatic' : 'Flattering'} />
            </div>

            <p className="text-xs leading-relaxed text-ink-3">
              No gear needed to practise: a desk lamp and an apple on your table teach the same
              physics as a $2000 strobe. Move the lamp, watch the shadows.
            </p>
          </>
        }
      />
    </ModulePage>
  );
}
