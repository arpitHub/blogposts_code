import { useEffect, useRef, useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import ToggleGroup from '../../components/ToggleGroup.jsx';
import WidgetShell, { Stat } from '../../components/WidgetShell.jsx';
import FocusCanvas, { AF_GRID, SUBJECTS } from './FocusCanvas.jsx';

export default function FocusingModule({ module }) {
  const [selected, setSelected] = useState([2, 1]); // row, col -> mid
  const [mode, setMode] = useState('afs');
  const [carX, setCarX] = useState(0.45);
  const rafRef = useRef();

  const focusedSubject = AF_GRID[selected[0]][selected[1]];

  // AF-C demo: the person strolls back and forth; in AF-C the camera would
  // keep re-focusing on them — we animate to make "continuous" tangible.
  useEffect(() => {
    if (mode !== 'afc') {
      setCarX(0.45);
      return;
    }
    let t0 = performance.now();
    const tick = (now) => {
      const t = (now - t0) / 1000;
      setCarX(0.45 + Math.sin(t * 0.7) * 0.22);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [mode]);

  return (
    <ModulePage
      module={module}
      intro="Your camera can only focus on one distance at a time — the AF point tells it where to look, and the focus mode tells it whether to keep looking."
      explanation={
        <p>
          Each <strong className="text-ink">AF point</strong> samples one spot in the frame:
          put it on the flowers and the person behind them goes soft.{' '}
          <strong className="text-ink">AF-S</strong> (single) locks focus once when you
          half-press — perfect for stills. <strong className="text-ink">AF-C</strong>{' '}
          (continuous) re-focuses every instant while you hold the button — essential the
          moment your subject moves. The most common beginner miss: letting the camera
          auto-pick a point, and it chooses the flowers when you meant the face.
        </p>
      }
      challenge="Select the bottom-left AF point (flowers) and notice the person blur out — this is exactly what happens when 'auto area' AF grabs the nearest object. Then switch to AF-C and watch the subject walk: in real AF-S they'd drift out of focus after the first lock."
    >
      <WidgetShell
        preview={
          <div className="relative h-full w-full">
            <FocusCanvas focusedSubject={focusedSubject} mode={mode} carX={mode === 'afc' ? carX : undefined} />
            {/* AF point overlay */}
            <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 p-8">
              {AF_GRID.flatMap((row, r) =>
                row.map((subj, c) => {
                  const isSel = selected[0] === r && selected[1] === c;
                  return (
                    <button
                      key={`${r}-${c}`}
                      onClick={() => setSelected([r, c])}
                      className="group grid place-items-center"
                      aria-label={`AF point ${r},${c} (${SUBJECTS[subj].label})`}
                    >
                      <span
                        className={`h-5 w-7 rounded-[3px] border-2 transition-all ${
                          isSel
                            ? 'scale-110 border-[#ff5a4d] shadow-[0_0_8px_rgba(255,90,77,0.6)]'
                            : 'border-white/35 group-hover:border-white/70'
                        }`}
                      />
                    </button>
                  );
                })
              )}
            </div>
          </div>
        }
        previewFooter={
          <div className="flex items-center justify-between text-xs">
            <span className="font-mono text-accent-2">
              ● {SUBJECTS[focusedSubject].label}
            </span>
            <span className="text-ink-3">click any AF point in the frame</span>
          </div>
        }
        controls={
          <>
            <div>
              <div className="mb-2 text-sm font-medium text-ink-2">Focus mode</div>
              <ToggleGroup
                value={mode}
                onChange={setMode}
                options={[
                  { value: 'afs', label: 'AF-S · single' },
                  { value: 'afc', label: 'AF-C · continuous' },
                ]}
              />
            </div>

            <div className="rounded-lg bg-panel-2 px-3 py-2.5 text-xs leading-relaxed text-ink-2">
              {mode === 'afs'
                ? 'AF-S: focus locks once on the selected point. If the subject moves after the lock, they leave the focal plane and blur. Best for portraits, landscapes, products.'
                : 'AF-C: the camera re-focuses continuously on whatever is under the point — watch it hold the walking subject. Best for kids, pets, sports, birds.'}
            </div>

            <div className="grid grid-cols-2 gap-2 border-t border-line pt-4">
              <Stat label="Focused on" value={SUBJECTS[focusedSubject].label.split(' ')[0]} />
              <Stat label="Subject moving?" value={mode === 'afc' ? 'Tracked ✓' : 'Would blur'} />
            </div>

            <p className="text-xs leading-relaxed text-ink-3">
              On most cameras: single-point AF + the joystick/d-pad to place it. Modern
              mirrorless adds eye-detect AF, which is AF-C with automatic point placement on a
              face — magical, but know this manual fallback for when it guesses wrong.
            </p>
          </>
        }
      />
    </ModulePage>
  );
}
