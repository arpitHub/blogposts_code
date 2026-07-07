import { useParams, Navigate } from 'react-router-dom';
import { getModule } from '../data/modules.js';
import ModulePage from '../components/ModulePage.jsx';
import PlaceholderPanel from '../components/PlaceholderPanel.jsx';
import ExposureTriangleModule from '../widgets/exposure-triangle/ExposureTriangleModule.jsx';
import ApertureDofModule from '../widgets/aperture-dof/ApertureDofModule.jsx';
import ShutterMotionModule from '../widgets/shutter-motion/ShutterMotionModule.jsx';
import IsoNoiseModule from '../widgets/iso-noise/IsoNoiseModule.jsx';
import CompositionModule from '../widgets/composition/CompositionModule.jsx';
import LightDirectionModule from '../widgets/light-direction/LightDirectionModule.jsx';
import WhiteBalanceModule from '../widgets/white-balance/WhiteBalanceModule.jsx';
import FocalLengthModule from '../widgets/focal-length/FocalLengthModule.jsx';
import HistogramModule from '../widgets/histogram/HistogramModule.jsx';
import FocusingModule from '../widgets/focusing/FocusingModule.jsx';

// Maps a module slug to its fully-built module page component.
// Anything not in this registry falls back to the placeholder.
const BUILT_MODULES = {
  'exposure-triangle': ExposureTriangleModule,
  'aperture-depth-of-field': ApertureDofModule,
  'shutter-speed-motion': ShutterMotionModule,
  'iso-noise': IsoNoiseModule,
  'composition': CompositionModule,
  'light-and-direction': LightDirectionModule,
  'white-balance': WhiteBalanceModule,
  'focal-length': FocalLengthModule,
  'histogram-reading': HistogramModule,
  'focusing-modes': FocusingModule,
};

export default function ModuleRoute() {
  const { slug } = useParams();
  const module = getModule(slug);

  if (!module) return <Navigate to="/" replace />;

  const Built = BUILT_MODULES[slug];
  if (Built) return <Built module={module} />;

  return (
    <ModulePage
      module={module}
      intro="This module is on the roadmap and hasn't been built yet."
    >
      <PlaceholderPanel title={module.title} />
    </ModulePage>
  );
}
