import { Suspense, lazy } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import { getModule } from '../data/modules.js';
import ModulePage from '../components/ModulePage.jsx';
import PlaceholderPanel from '../components/PlaceholderPanel.jsx';

// Every widget is lazy-loaded so each module page ships as its own chunk —
// keeps the landing page light (Recharts alone would triple it).
const BUILT_MODULES = {
  'exposure-triangle': lazy(() => import('../widgets/exposure-triangle/ExposureTriangleModule.jsx')),
  'aperture-depth-of-field': lazy(() => import('../widgets/aperture-dof/ApertureDofModule.jsx')),
  'shutter-speed-motion': lazy(() => import('../widgets/shutter-motion/ShutterMotionModule.jsx')),
  'iso-noise': lazy(() => import('../widgets/iso-noise/IsoNoiseModule.jsx')),
  'composition': lazy(() => import('../widgets/composition/CompositionModule.jsx')),
  'light-and-direction': lazy(() => import('../widgets/light-direction/LightDirectionModule.jsx')),
  'white-balance': lazy(() => import('../widgets/white-balance/WhiteBalanceModule.jsx')),
  'focal-length': lazy(() => import('../widgets/focal-length/FocalLengthModule.jsx')),
  'histogram-reading': lazy(() => import('../widgets/histogram/HistogramModule.jsx')),
  'focusing-modes': lazy(() => import('../widgets/focusing/FocusingModule.jsx')),
  'raw-vs-jpeg': lazy(() => import('../widgets/raw-jpeg/RawJpegModule.jsx')),
  'post-processing-basics': lazy(() => import('../widgets/post-processing/PostProcessingModule.jsx')),
  'genre-guides': lazy(() => import('../widgets/genre-guides/GenreGuidesModule.jsx')),
  'gear-explainer': lazy(() => import('../widgets/gear/GearModule.jsx')),
};

export default function ModuleRoute() {
  const { slug } = useParams();
  const module = getModule(slug);

  if (!module) return <Navigate to="/" replace />;

  const Built = BUILT_MODULES[slug];
  if (Built) {
    return (
      <Suspense
        fallback={
          <div className="grid min-h-[60vh] place-items-center font-mono text-xs text-ink-3">
            loading module…
          </div>
        }
      >
        <Built module={module} />
      </Suspense>
    );
  }

  return (
    <ModulePage
      module={module}
      intro="This module is on the roadmap and hasn't been built yet."
    >
      <PlaceholderPanel title={module.title} />
    </ModulePage>
  );
}
