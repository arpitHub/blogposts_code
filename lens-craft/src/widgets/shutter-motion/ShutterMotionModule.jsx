import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import ToggleGroup from '../../components/ToggleGroup.jsx';
import WidgetShell, { Stat } from '../../components/WidgetShell.jsx';
import MotionSceneCanvas from './MotionSceneCanvas.jsx';
import { SHUTTER_STOPS } from '../exposure-triangle/exposureMath.js';

export default function ShutterMotionModule({ module }) {
  const [shutterIndex, setShutterIndex] = useState(2); // 1/1000
  const [scene, setScene] = useState('waterfall');

  const stop = SHUTTER_STOPS[shutterIndex];
  const handholdable = stop.seconds <= 1 / 60;

  return (
    <ModulePage
      module={module}
      intro="The shutter is a time machine. It decides how much of a moment gets stacked into one frame — a thousandth of a second freezes a droplet mid-air; a full second turns the same waterfall into silk."
      explanation={
        <p>
          Motion blur isn't a flaw — it's the physical record of movement during the exposure.
          Fast shutter speeds (<strong className="text-ink">1/500 and up</strong>) freeze sports
          and wildlife. Slow ones (<strong className="text-ink">1/15 to several seconds</strong>)
          communicate motion: silky water, light trails, spinning wheels. Below about 1/60 your
          own hands shake enough to blur <em>everything</em>, not just the moving subject —
          that's when you need a tripod.
        </p>
      }
      challenge="On the waterfall, find the exact shutter speed where individual droplets stop reading as dots and start becoming streaks (it's around 1/125–1/60). Then switch to the pinwheel and find the speed where you can no longer count the six blades."
    >
      <WidgetShell
        preview={<MotionSceneCanvas scene={scene} exposureSeconds={stop.seconds} />}
        previewFooter={
          <div className="flex items-center justify-between">
            <ToggleGroup
              value={scene}
              onChange={setScene}
              options={[
                { value: 'waterfall', label: 'Waterfall' },
                { value: 'pinwheel', label: 'Pinwheel' },
              ]}
            />
            <span
              className="font-mono text-xs"
              style={{ color: handholdable ? 'var(--color-level-beginner)' : 'var(--color-accent-2)' }}
            >
              {handholdable ? '✓ hand-holdable' : '⚠ tripod needed'}
            </span>
          </div>
        }
        controls={
          <>
            <Slider
              label="Shutter speed"
              color="#35d0ba"
              value={shutterIndex}
              min={0}
              max={SHUTTER_STOPS.length - 1}
              display={stop.label + 's'}
              onChange={setShutterIndex}
            />

            {/* time-scale visual: how long the shutter stays open */}
            <div>
              <div className="mb-1.5 font-mono text-[10px] uppercase tracking-wide text-ink-3">
                Time the shutter is open
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-panel-2">
                <div
                  className="h-full rounded-full bg-[#35d0ba] transition-all duration-200"
                  style={{ width: `${Math.max(1.2, Math.pow(shutterIndex / (SHUTTER_STOPS.length - 1), 1.6) * 100)}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-3">
                <span>blink of a hummingbird</span>
                <span>full second</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 border-t border-line pt-4">
              <Stat
                label="Moving subject"
                value={stop.seconds <= 1 / 500 ? 'Frozen' : stop.seconds <= 1 / 60 ? 'Mostly sharp' : stop.seconds <= 1 / 8 ? 'Blurred' : 'Silky / streaked'}
              />
              <Stat label="Typical use" value={stop.seconds <= 1 / 500 ? 'Sports, birds' : stop.seconds <= 1 / 60 ? 'Everyday' : stop.seconds <= 1 / 8 ? 'Panning shots' : 'Waterfalls, night'} />
            </div>

            <p className="text-xs leading-relaxed text-ink-3">
              Rule of thumb: to freeze motion, use at least 1/(2× subject speed feel) — 1/250 for
              walking people, 1/1000 for sports, 1/2000+ for birds in flight.
            </p>
          </>
        }
      />
    </ModulePage>
  );
}
