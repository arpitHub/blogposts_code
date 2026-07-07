import { useState } from 'react';
import ModulePage from '../../components/ModulePage.jsx';
import Slider from '../../components/Slider.jsx';
import { Stat } from '../../components/WidgetShell.jsx';

// Sensor sizes in mm (width x height) and their crop factor vs full frame.
const SENSORS = [
  { id: 'ff', name: 'Full frame', w: 36, h: 24, crop: 1.0, color: '#ffb020', note: 'The 35mm-film reference. Best low light and shallowest DOF per f-stop; biggest, priciest bodies and lenses.' },
  { id: 'apsc', name: 'APS-C', w: 23.6, h: 15.7, crop: 1.5, color: '#35d0ba', note: 'The enthusiast sweet spot — most beginner DSLRs/mirrorless. Your 50mm behaves like a 75mm.' },
  { id: 'm43', name: 'Micro 4/3', w: 17.3, h: 13, crop: 2.0, color: '#52c97a', note: 'Half the frame, double the reach: tiny bodies, tiny lenses, wildlife-friendly crop.' },
  { id: 'one', name: '1-inch', w: 13.2, h: 8.8, crop: 2.7, color: '#b06ce8', note: 'Premium compacts and drones. Big step up from a phone, pocketable.' },
  { id: 'phone', name: 'Phone (1/1.3")', w: 9.8, h: 7.3, crop: 3.7, color: '#ff5a6a', note: "Why phones use computational tricks: physics gives this sensor ~7% of a full frame's light-gathering area." },
];

const SCALE = 7.4; // px per mm at full size

export default function GearModule({ module }) {
  const [selected, setSelected] = useState('apsc');
  const [focal, setFocal] = useState(50);

  const sensor = SENSORS.find((s) => s.id === selected);
  const equivalent = Math.round(focal * sensor.crop);

  return (
    <ModulePage
      module={module}
      intro="Ignore megapixels — sensor size is the spec that actually shapes your photos. It sets how much light you collect, how shallow your focus can go, and what your lenses 'really' are."
      explanation={
        <p>
          A smaller sensor sees a <strong className="text-ink">cropped view</strong> of what the
          lens projects, which is why the same 50mm lens frames like 75mm on APS-C and 100mm on
          Micro 4/3 — that multiplier is the <strong className="text-ink">crop factor</strong>.
          Smaller sensors also collect less total light (more noise at the same ISO) and need
          wider apertures to match full-frame background blur. None of this makes small sensors
          bad: it makes them <em>different tools</em> — reach and portability vs. low-light
          headroom.
        </p>
      }
      challenge="Your kit lens is probably an 18–55mm on APS-C. Use the calculator: what full-frame view does it really cover? (27–82mm — which is why it feels 'normal' and never truly wide.) Now check what 400mm of wildlife reach costs on full frame vs Micro 4/3."
    >
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.3fr_1fr]">
        {/* nested sensor rectangles, true to scale */}
        <div className="rounded-2xl border border-line bg-panel p-6">
          <span className="mb-4 block font-mono text-[10px] uppercase tracking-wider text-ink-3">
            Physical sensor sizes — true relative scale
          </span>
          <div className="relative mx-auto" style={{ width: 36 * SCALE, height: 24 * SCALE }}>
            {SENSORS.map((s) => {
              const isSel = s.id === selected;
              return (
                <button
                  key={s.id}
                  onClick={() => setSelected(s.id)}
                  className="absolute bottom-0 left-0 rounded-md border-2 transition-all"
                  style={{
                    width: s.w * SCALE,
                    height: s.h * SCALE,
                    borderColor: s.color,
                    background: isSel ? `${s.color}22` : 'transparent',
                    boxShadow: isSel ? `0 0 14px ${s.color}55` : 'none',
                    zIndex: Math.round(40 - s.w),
                  }}
                  aria-label={s.name}
                >
                  <span
                    className="absolute left-1.5 top-1 font-mono text-[10px]"
                    style={{ color: s.color }}
                  >
                    {s.name}
                  </span>
                </button>
              );
            })}
          </div>
          <p className="mt-5 text-xs leading-relaxed text-ink-2">{sensor.note}</p>
        </div>

        {/* crop factor calculator */}
        <div className="flex flex-col gap-5 rounded-2xl border border-line bg-panel p-5">
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-3">
            Crop factor calculator
          </span>

          <Slider
            label="Lens focal length"
            color={sensor.color}
            value={focal}
            min={8}
            max={400}
            display={`${focal}mm`}
            onChange={setFocal}
          />

          <div className="rounded-xl bg-black/40 px-4 py-4 text-center font-mono">
            <div className="text-[11px] uppercase tracking-wide text-ink-3">
              {focal}mm on {sensor.name}
            </div>
            <div className="my-1 text-2xl" style={{ color: sensor.color }}>
              ≈ {equivalent}mm
            </div>
            <div className="text-[11px] text-ink-3">full-frame equivalent view</div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <Stat label="Crop factor" value={`${sensor.crop.toFixed(1)}×`} />
            <Stat
              label="Light-gathering area"
              value={`${Math.round(((sensor.w * sensor.h) / (36 * 24)) * 100)}% of FF`}
            />
          </div>

          <p className="text-xs leading-relaxed text-ink-3">
            Tap the rectangles to switch sensors. "Equivalent" refers to field of view — the
            lens's physical focal length never changes.
          </p>
        </div>
      </div>
    </ModulePage>
  );
}
