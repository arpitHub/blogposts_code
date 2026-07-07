import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import ToggleGroup from '../../components/ToggleGroup.jsx';
import WidgetShell from '../../components/WidgetShell.jsx';
import CompositionCanvas from './CompositionCanvas.jsx';

const THIRD = 1 / 3;
const TOL = 0.045;

export default function CompositionModule({ module }) {
  const [subjectX, setSubjectX] = useState(0.5);
  const [horizonY, setHorizonY] = useState(0.5);
  const [grid, setGrid] = useState('thirds');

  const subjectOnThird = Math.abs(subjectX - THIRD) < TOL || Math.abs(subjectX - 2 * THIRD) < TOL;
  const horizonOnThird = Math.abs(horizonY - THIRD) < TOL || Math.abs(horizonY - 2 * THIRD) < TOL;
  const deadCenter = Math.abs(subjectX - 0.5) < TOL && Math.abs(horizonY - 0.5) < TOL;

  let feedback, feedbackColor;
  if (subjectOnThird && horizonOnThird) {
    feedback = 'Classic rule-of-thirds composition — subject and horizon both on the lines.';
    feedbackColor = 'var(--color-level-beginner)';
  } else if (deadCenter) {
    feedback = 'Dead-centre with a bisecting horizon — usually static, but powerful for symmetry.';
    feedbackColor = 'var(--color-accent)';
  } else if (horizonOnThird) {
    feedback = 'Horizon sits on a third — now try the lighthouse on a vertical line.';
    feedbackColor = 'var(--color-accent)';
  } else if (subjectOnThird) {
    feedback = 'Subject on a third — now drag the horizon off centre. High = sea story, low = sky story.';
    feedbackColor = 'var(--color-accent)';
  } else {
    feedback = 'Drag the lighthouse and the horizon line. Watch how the frame\'s balance changes.';
    feedbackColor = 'var(--color-ink-3)';
  }

  return (
    <ModulePage
      module={module}
      intro="Composition is deciding where things go in the frame. There are no laws — but there are patterns that reliably feel balanced, and you should know them before you break them."
      explanation={
        <p>
          The <strong className="text-ink">rule of thirds</strong> places subjects on the grid
          lines and their intersections ("power points") rather than dead centre — it gives the
          eye somewhere to travel. The <strong className="text-ink">horizon</strong> choice
          decides what the photo is about: on the lower third, the sky is the story; on the
          upper third, the sea is. Perfect centring isn't wrong — it's a deliberate choice for
          symmetry and stillness.
        </p>
      }
      challenge="Compose three different photos of the same scene: (1) lighthouse on the right third with a low horizon — sky drama, (2) lighthouse on the left third with a high horizon — ocean texture, (3) dead centre with the horizon bisecting — formal symmetry. Feel how differently the same lighthouse reads."
    >
      <WidgetShell
        preview={
          <CompositionCanvas
            subject={{ x: subjectX }}
            horizonY={horizonY}
            grid={grid}
            onDrag={({ subjectX: sx, horizonY: hy }) => {
              if (sx !== undefined) setSubjectX(sx);
              if (hy !== undefined) setHorizonY(hy);
            }}
          />
        }
        previewFooter={
          <p className="text-sm leading-snug transition-colors" style={{ color: feedbackColor }}>
            {feedback}
          </p>
        }
        controls={
          <>
            <div>
              <div className="mb-2 text-sm font-medium text-ink-2">Grid overlay</div>
              <ToggleGroup
                value={grid}
                onChange={setGrid}
                options={[
                  { value: 'thirds', label: 'Thirds' },
                  { value: 'center', label: 'Centre' },
                  { value: 'off', label: 'Off' },
                ]}
              />
            </div>

            <div className="rounded-lg bg-panel-2 px-3 py-2.5 text-xs leading-relaxed text-ink-2">
              <span className="mb-1 block font-mono text-[10px] uppercase tracking-wide text-ink-3">
                How to use
              </span>
              Drag the <strong className="text-ink">lighthouse</strong> left/right and the{' '}
              <strong className="text-ink">horizon</strong> up/down, directly on the photo.
            </div>

            <div className="flex flex-col gap-2 border-t border-line pt-4 font-mono text-xs">
              <ReadoutRow ok={subjectOnThird} label="Subject on a third line" />
              <ReadoutRow ok={horizonOnThird} label="Horizon on a third line" />
              <ReadoutRow ok={deadCenter} label="Symmetric / centred" />
            </div>

            <p className="text-xs leading-relaxed text-ink-3">
              Most cameras can show this exact grid in the viewfinder — turn it on in your
              display settings and it becomes second nature within a week.
            </p>
          </>
        }
      />
    </ModulePage>
  );
}

function ReadoutRow({ ok, label }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className="grid h-4 w-4 place-items-center rounded-full text-[10px]"
        style={{
          background: ok ? 'rgba(82,201,122,0.18)' : 'var(--color-panel-2)',
          color: ok ? 'var(--color-level-beginner)' : 'var(--color-ink-3)',
        }}
      >
        {ok ? '✓' : '·'}
      </span>
      <span style={{ color: ok ? 'var(--color-ink)' : 'var(--color-ink-3)' }}>{label}</span>
    </div>
  );
}
